import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class Adafuse(nn.Module):
    """
    Adafuse module that fuses textual and visual information using 
    Adaptive Gating and a Semantic-aware Mixture-of-Experts (MoE) mechanism.
    """

    def __init__(self, feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param feature_dim: Input feature dimension (e.g., CLIP/BLIP embedding size)
        :param projection_dim: Dimension for the initial projection layers
        :param hidden_dim: Hidden dimension for the MoE experts and gating
        """
        super(Adafuse, self).__init__()
        
        # Modality Dropout probabilities (for training robustness)
        self.text_drop_prob = 0.0
        self.image_drop_prob = 0.0

        # 1. Initial Projection Layers
        # Projects raw features to a higher-dimensional space
        self.text_projection_layer = nn.Linear(feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

        # 2. Adaptive Gating Branch (Lambda)
        # dynamically computes a scalar weight to balance text and image reliability
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )

        # 3. Logit Scale (Temperature parameter for contrastive loss)
        # Initialized to approx log(50.0) -> 3.91
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(50.0))

        # 4. Semantic-aware Mixture-of-Experts (MoE) Branch
        self.num_experts = 4
        input_dim = projection_dim * 2
        
        # A. Experts: A list of independent MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ) for _ in range(self.num_experts)
        ])
        
        # B. Router: Predicts routing weights for each expert
        self.moe_router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, self.num_experts) 
        )
        
        # C. Shared Output Layer: Maps MoE output back to original feature dimension
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor) -> Tuple[torch.tensor, Dict]:
        """
        Forward pass for training. 
        Returns scaled logits and auxiliary info for logging.
        """
        # 1. Combine features to get the predicted embedding
        predicted_features, aux_info = self.combine_features(image_features, text_features)

        # 2. Normalize target features
        target_features = F.normalize(target_features, dim=-1)
        
        # 3. Compute similarity logits (scaled by temperature)
        logits = self.logit_scale.exp() * predicted_features @ target_features.T
        
        return logits, aux_info

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> Tuple[torch.tensor, Dict]:
        """
        Fuses image and text features.
        Returns:
            - output: The fused embedding (normalized)
            - aux_info: Dictionary containing gating values and router probabilities
        """
        # --- Modality Dropout (Training Only) ---
        if self.training:
            batch_size = text_features.size(0)
            
            # Generate masks
            keep_text = (torch.rand(batch_size, 1, device=text_features.device) > self.text_drop_prob).float()
            keep_image = (torch.rand(batch_size, 1, device=image_features.device) > self.image_drop_prob).float()
            
            # Safety: Prevent dropping both modalities simultaneously
            both_dropped = (1.0 - keep_text) * (1.0 - keep_image)
            keep_image = keep_image + both_dropped 
            
            # Apply masks
            text_features = text_features * keep_text
            image_features = image_features * keep_image

        # 1. Projection
        text_projected = self.dropout1(F.gelu(self.text_projection_layer(text_features)))
        image_projected = self.dropout2(F.gelu(self.image_projection_layer(image_features)))

        # 2. Concatenation (Joint Context)
        raw_combined_features = torch.cat((text_projected, image_projected), -1)

        # 3. MoE Branch Execution
        
        # A. Calculate Routing Weights (Softmax)
        router_logits = self.moe_router(raw_combined_features)
        router_probs = F.softmax(router_logits, dim=-1) # [Batch, Num_Experts]

        # B. Compute Expert Outputs
        # Stack outputs: [Batch, Num_Experts, Hidden_Dim]
        expert_outputs = torch.stack([
            expert(raw_combined_features) for expert in self.experts
        ], dim=1)

        # C. Weighted Fusion
        # [Batch, Num_Experts, 1] * [Batch, Num_Experts, Hidden] -> Sum over experts -> [Batch, Hidden]
        router_probs_unsqueezed = router_probs.unsqueeze(-1)
        fused_hidden = (router_probs_unsqueezed * expert_outputs).sum(dim=1)
        
        # D. Final Projection (Residual Term)
        residual_features = self.dropout3(self.output_layer(fused_hidden))

        # 4. Adaptive Gating Branch Execution
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)

        # 5. Final Fusion
        # Formula: Output = λ * Text + (1 - λ) * Image + MoE_Residual
        output = dynamic_scalar * text_features + \
                 (1 - dynamic_scalar) * image_features + \
                 residual_features 

        aux_info = {
            "scalar_values": dynamic_scalar,  # For logging Lambda (Gating)
            "router_probs": router_probs      # For logging Expert utilization
        }
        
        return F.normalize(output, dim=-1), aux_info