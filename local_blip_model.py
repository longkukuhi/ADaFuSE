import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import numpy as np
from typing import Optional
from models.med import BertConfig, BertModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

class LocalBlipForRetrieval(nn.Module):

    def __init__(self,
                 med_config: str,
                 image_size: int,
                 vit: str,
                 embed_dim: int,
                 pretrained_path: str,
                 vit_grad_ckpt: bool = False,
                 vit_ckpt_layer: int = 0,
                 vit_drop_path_rate: float = 0.1,
                 bert_attention_dropout: float = 0.1,
                 bert_hidden_dropout: float = 0.1,
                 ):
        super().__init__()

        # Initialize Visual Encoder (ViT)
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer,
            drop_path_rate=vit_drop_path_rate
        )
        self.vit_drop_path_rate = vit_drop_path_rate
        self.tokenizer = init_tokenizer()
        
        # Initialize Text Encoder (BERT) with custom config
        med_config_obj = BertConfig.from_json_file(med_config)
        med_config_obj.encoder_width = vision_width
        med_config_obj.attention_probs_dropout_prob = bert_attention_dropout
        med_config_obj.hidden_dropout_prob = bert_hidden_dropout
        
        self.text_encoder = BertModel(config=med_config_obj, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # Initialize logit scale (temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if pretrained_path:
            # Load pretrained weights
            model, msg = load_checkpoint(self, pretrained_path)

            keys_to_ignore = {"logit_scale"}
            missing = [k for k in msg.missing_keys if k not in keys_to_ignore]
            unexpected = [k for k in msg.unexpected_keys if k not in keys_to_ignore]
            
            if missing:
                print("Missing keys during loading:", missing)
            if unexpected:
                # Note: Extra keys (like vision_proj_m) are expected if loading original MoCo weights
                print("Unexpected keys (expected/ignored):", unexpected)
        else:
            print("Warning: No pretrained_path provided, using random initialization.")

    def get_image_features(self, pixel_values: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """
        Extract and normalize image features.
        """
        # Visual encoder forward pass
        image_embeds = self.visual_encoder(pixel_values)
        # Take [CLS] token (index 0) and project
        image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat

    def get_text_features(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None, **kwargs) -> torch.FloatTensor:
        """
        Extract and normalize text features.
        """
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            mode='text'
        )
        # Take [CLS] token (index 0) and project
        text_feat = normalize(self.text_proj(text_outputs.last_hidden_state[:, 0, :]), dim=-1)
        return text_feat

    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay_params = {
            'visual_encoder.cls_token', 
            'visual_encoder.pos_embed',
            'text_encoder.embeddings.position_embeddings.weight',
            'logit_scale' 
        }
        return no_decay_params