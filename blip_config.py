from pathlib import Path

class Config:
    
    # --- 1. Experiment & Path Configuration ---
    log_base_dir = "experiments_blip"
    experiment_name = "your_project_name"  # Experiment name (WandB Run Name)

    dialogue_format = "Summarized"   # Options: 'Summarized' or 'VisDial'
    dialogue_round = 10
    use_random_rounds = True
    use_caption_masking = False
    caption_masking_prob = 0.2
    
    # Dataset Paths
    train_json_path = './dataset/visdial_1.0_train_sum_all.json'
    val_corpus_json_path = './ChatIR_Protocol/Search_Space_val_50k.json'
    val_queries_path = 'dialogues/VD-reformulated.json'
    val_generated_image_dir = './data/generated_images'

    # --- 2. Data & Model Configuration ---
    # Training modes: 'blip_only', 'end_to_end', 'Adafuse_only'
    training_mode = 'end_to_end' 
    
    # Path to BLIP pretrained weights (e.g., fine-tuned on Retrieval or COCO)
    blip_model = './blip_models/chatir_weights.ckpt'

    # Caching paths for validation features (speeds up evaluation)
    val_cache_corpus_path = "./temp/chatir_val_corpus_features.pt"
    val_cache_gen_path = "./temp/chatir_val_gen_features.pt"
    val_cache_force_rebuild = False
    
    # Fusion Strategy: 'add' (Simple addition) or 'Adafuse' (Gating + MoE)
    fusion_strategy = 'Adafuse'

    # --- 3. ADaFuSE Architecture Parameters ---
    Adafuse_lr = 5e-5
    projection_dim = 512
    hidden_dim = 512 * 4  # 2048

    # --- 4. Training Hyperparameters ---
    num_epochs = 50
    batch_size = 128
    update_freq = 16   # Gradient accumulation steps
    blip_lr = 1e-5

    warmup_epochs = 5     
    validation_frequency = 1
    weight_decay = 0.05   
    layer_decay = 0.9     # Layer-wise learning rate decay factor
    clip_grad = 3.0       # Gradient clipping threshold
    model_ema = False     # Enable Model EMA
    model_ema_decay = 0.999 

    # Dropout settings
    vit_drop_path_rate = 0.1      # DropPath rate for ViT
    bert_attention_dropout = 0.1  # Attention dropout for BERT
    bert_hidden_dropout = 0.1     # Hidden layer dropout for BERT
    
    # --- 5. Training Control ---
    resume_from = None  # Path to checkpoint to resume from, e.g., "saved_models/blip_12.pt"
    save_training = True # Whether to save checkpoints during training

    # --- 6. Image Processing & Loss Configuration ---
    input_size = 224  # Input image resolution
    
    # Available components: "ref_tgt", "text_tgt", "fused_tgt", "ref_text"
    loss_components =  ["fused_tgt"]  
    loss_weights = [1.0]  # Weights for each loss component

    # --- 7. WandB Configuration ---
    wandb_entity = "your_user_name"  # Username or Team name (set to None if not used)
    wandb_project = "your_project_name"
    wandb_mode = "online" # Options: 'online', 'offline', 'disabled'

    # Checkpoint path for Adafuse (if loading separately)
    Adafuse_checkpoint_path = None