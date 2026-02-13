import wandb
import json
from typing import List, Dict, Tuple, Optional
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from pathlib import Path
from transformers import get_cosine_schedule_with_warmup
from transformers import BertTokenizer
from local_blip_model import LocalBlipForRetrieval
import torch.nn.functional as F
from argparse import Namespace
from timm.utils import ModelEma
 # from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner, get_num_layer_for_vit
from blip_optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from blip_optim_factory import get_num_layer_for_blip # 导入我们的新函数

from adafuse import Adafuse
from my_dataset import build_blip_transform,ComposedRetrievalDataset, CorpusDataset,ValidationQueriesDataset, get_blip_transform, QueryImageDataset
from my_utils import update_train_running_results, set_train_bar_description
from blip_config import Config

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def save_checkpoint(name: str, cur_epoch: int, model: nn.Module, optimizer: optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler, best_metric: float, training_path: Path,
                    scheduler,training_mode: str,Adafuse: Adafuse = None, 
                    model_ema: ModelEma = None,Adafuse_ema: ModelEma = None):

    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)

    checkpoint = {
        'epoch': cur_epoch,
        # 'model': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_recall_at_10': float(best_metric),
        'training_mode': training_mode # 记录一下当时的模式，方便后续排查
    }
    if training_mode != 'Adafuse_only':
    # 非 Adafuse_only 模式 (即 blip_only 或 end_to_end)，需要保存 BLIP
        model_state_dict = model.state_dict()
        
        # 处理可能的 DataParallel 包装
        model_ref = model.module if hasattr(model, 'module') else model
        if hasattr(model_ref, 'text_encoder') and hasattr(model_ref.text_encoder, 'config'):
            text_width = model_ref.text_encoder.config.hidden_size

        # 注入随机初始化的 ITM Head 参数 (保持你原有的逻辑)
        if 'itm_head.weight' not in model_state_dict:
            model_state_dict['itm_head.weight'] = torch.randn(2, text_width)
            model_state_dict['itm_head.bias'] = torch.zeros(2)
        
        checkpoint['model'] = model_state_dict
        
        # 同样，只有在非 Adafuse_only 时才保存 BLIP 的 EMA
        if model_ema is not None:
            checkpoint['model_ema_state_dict'] = model_ema.ema.state_dict()
    else:
        # Adafuse_only 模式：不保存 BLIP 权重
        pass


    if Adafuse:
        checkpoint['Adafuse_state_dict'] = Adafuse.state_dict()
    if Adafuse_ema is not None:
        checkpoint['Adafuse_ema_state_dict'] = Adafuse_ema.ema.state_dict()
        
    torch.save(checkpoint, str(models_path / f'{name}.pt'))

def extract_corpus_features(
    corpus_dataset: CorpusDataset,
    blip_model: nn.Module,
    batch_size: int,
    device: torch.device,
    cache_path: str = "",
    cache_force_rebuild: bool = False,
    meta: Optional[dict] = None,
):
    if cache_path:
        cache_path = str(cache_path)
        if (not cache_force_rebuild) and os.path.exists(cache_path):
            ckpt = torch.load(cache_path, map_location="cpu")
            if meta is None or ckpt.get("meta", {}) == meta:
                corpus_ids = ckpt["corpus_ids"].to(device, non_blocking=True)
                corpus_vectors = ckpt["corpus_vectors"].to(device, non_blocking=True)
                return corpus_ids, corpus_vectors
            else:
                print(f"[Cache] corpus cache meta mismatch, rebuild: {cache_path}")

    def corpus_collate_fn(batch):
        image_paths, images_tensors = zip(*batch)
        pixel_values = torch.stack(images_tensors)
        ids = [corpus_dataset.path_to_id_map[p] for p in image_paths]
        return torch.tensor(ids), pixel_values

    corpus_loader = DataLoader(
        dataset=corpus_dataset,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=corpus_collate_fn,
        pin_memory=True
    )

    corpus_vectors_list = []
    corpus_ids_list = []

    with torch.no_grad():
        for batch_ids, pixel_values in tqdm(corpus_loader, desc="为大规模语料库提取特征 (BLIP)"):
            pixel_values = pixel_values.to(device, non_blocking=True)
            batch_vectors = blip_model.get_image_features(pixel_values)
            corpus_vectors_list.append(batch_vectors.cpu().half())  # 存盘用 half 更省空间
            corpus_ids_list.append(batch_ids.cpu())

    corpus_vectors = torch.cat(corpus_vectors_list)
    corpus_ids = torch.cat(corpus_ids_list)
    arg_ids = torch.argsort(corpus_ids)
    corpus_vectors = corpus_vectors[arg_ids]
    corpus_ids = corpus_ids[arg_ids]

    # 2) 保存缓存
    if cache_path:
        Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"corpus_ids": corpus_ids, "corpus_vectors": corpus_vectors, "meta": meta or {}},
            cache_path
        )
        print(f"[Cache] saved corpus cache -> {cache_path}")

    return corpus_ids.to(device), corpus_vectors.to(device)

def _create_or_load_generated_cache_blip(
    model: nn.Module,
    queries_path: str,
    gen_image_dir: str,
    num_eval_rounds: int,
    device: torch.device,
    input_size: int,
    batch_size: int,
    feature_dim: int,
    cache_path: str = "",
    cache_force_rebuild: bool = False,
    meta: Optional[dict] = None,
):
    # 0) load
    if cache_path:
        cache_path = str(cache_path)
        if (not cache_force_rebuild) and os.path.exists(cache_path):
            ckpt = torch.load(cache_path, map_location="cpu")
            if meta is None or ckpt.get("meta", {}) == meta:
                gen_feats = ckpt["gen_feats"].to(device, non_blocking=True)
                print(f"[Cache] loaded gen cache <- {cache_path}")
                return gen_feats
            else:
                print(f"[Cache] gen cache meta mismatch, rebuild: {cache_path}")

    # 1) build
    transform = get_blip_transform(input_size=input_size)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    query_dataset = QueryImageDataset(
        queries=queries,
        gen_image_dir=gen_image_dir,
        num_rounds=num_eval_rounds,
        transform=transform
    )

    query_loader = DataLoader(
        dataset=query_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False
    )

    gen_feats = torch.empty((len(queries), num_eval_rounds, feature_dim), dtype=torch.float16)

    model.eval()
    with torch.no_grad():
        for filenames, images in tqdm(query_loader, desc="Caching Generated Features (Batch)"):
            images = images.to(device, non_blocking=True)
            batch_feats = model.get_image_features(images).detach().cpu().half()  # CPU half 存入大 tensor

            for i, filename in enumerate(filenames):
                # filename: "{query_idx}_{round_idx}.jpg"
                q, rjpg = filename.split("_")
                r = int(rjpg.split(".")[0])
                gen_feats[int(q), r] = batch_feats[i]

    # 2) save
    if cache_path:
        Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
        torch.save({"gen_feats": gen_feats, "meta": meta or {}}, cache_path)
        print(f"[Cache] saved gen cache -> {cache_path}")

    return gen_feats.to(device, non_blocking=True)

def _calculate_fused_scores_blip(method: str, text_features: torch.Tensor, gen_features: torch.Tensor, 
                                  corpus_features: torch.Tensor, dialog_length: int,
                                  fusion_strategy: str, Adafuse_model: nn.Module):

    corpus_features_T = corpus_features.T
    aux_info = {}
    if method == 'text':
        return text_features @ corpus_features_T, aux_info

    if method == 'image':
        return gen_features @ corpus_features_T, aux_info

    if method == 'dar':
        text_scores = text_features @ corpus_features_T
        gen_scores = gen_features @ corpus_features_T

        if dialog_length < 2:
            w_text, w_img = 0.8, 0.2
        else:
            w_text, w_img = 0.5, 0.5
        return w_text * text_scores + w_img * gen_scores, aux_info
    if method == 'fused_feature':
        fused_features = None
        if fusion_strategy == 'Adafuse':
            with torch.no_grad(): 
                fused_features, aux_info = Adafuse_model.combine_features(gen_features, text_features)
        elif fusion_strategy == 'add':
            fused_features = F.normalize(gen_features + text_features, dim=-1, eps=1e-6)
        return fused_features @ corpus_features_T, aux_info
    
    raise ValueError(f"未知的融合方法: {method}")

def _calculate_ranks(ranked_indices, target_ids):

    mask = (ranked_indices == target_ids)
    hits = mask.nonzero(as_tuple=False) 
    if hits.size(0) != ranked_indices.size(0):
        ranks = torch.full((ranked_indices.size(0),), float('inf'), device=ranked_indices.device)
        ranks[hits[:, 0]] = hits[:, 1].float()
        return ranks
        
    return hits[:, 1]  

def get_first_hitting_time(target_recall, num_rounds, hitting_recall=10):
    if len(target_recall) == 0:
        return torch.tensor([])
    target_recalls = target_recall.view(num_rounds, -1).T
    hits = (target_recalls < hitting_recall)
    final_hits = torch.inf * torch.ones(target_recalls.shape[0])
    hitting_times = []
    for ro_i in range(num_rounds):
        rh = hits[:, ro_i]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())
    return torch.stack(hitting_times)

def cumulative_hits_per_round(target_recall, num_rounds, hitting_recall=10):


    if len(target_recall) == 0:
        return [0.0] * num_rounds
    ht_times = get_first_hitting_time(target_recall, num_rounds, hitting_recall)
    if ht_times.numel() == 0:
        return [0.0] * num_rounds
    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times.shape[1])

def run_eval4_validation(
    blip_model: nn.Module,
    Adafuse: nn.Module,
    hyper_params: dict,
    epoch: int,
    device: torch.device,
    is_ema: bool = False
):

    blip_model.eval()
    if Adafuse:
        Adafuse.eval()

    training_mode = hyper_params.get("training_mode", "")
    enable_cache = (training_mode == "Adafuse_only")
    cache_force = bool(hyper_params.get("val_cache_force_rebuild", False))

    cache_corpus_path = hyper_params.get("val_cache_corpus_path", "") if enable_cache else ""
    cache_gen_path = hyper_params.get("val_cache_gen_path", "") if enable_cache else ""

    val_queries_dataset = ValidationQueriesDataset(
        queries_path=hyper_params["val_queries_path"],
        generated_image_dir=hyper_params["val_generated_image_dir"]
    )

    val_transform = get_blip_transform(input_size=hyper_params["input_size"])
    corpus_val_dataset = CorpusDataset(
        json_file_path=hyper_params["val_corpus_json_path"],
        pil_transform=val_transform
    )
    path_to_id_map = corpus_val_dataset.path_to_id_map

    corpus_meta = {
        "input_size": hyper_params["input_size"],
        "val_corpus_json_path": hyper_params["val_corpus_json_path"],
        "blip_model_path": hyper_params.get("blip_model", ""),
        "dialogue_format": hyper_params.get("dialogue_format", ""),
    }

    corpus_ids, corpus_vectors = extract_corpus_features(
        corpus_dataset=corpus_val_dataset,
        blip_model=blip_model,
        batch_size=hyper_params["batch_size"],
        device=device,
        cache_path=cache_corpus_path,
        cache_force_rebuild=cache_force,
        meta=corpus_meta
    )

    num_eval_rounds = 11
    feature_dim = blip_model.text_proj.out_features  # 256

    gen_meta = {
        "input_size": hyper_params["input_size"],
        "val_queries_path": hyper_params["val_queries_path"],
        "val_generated_image_dir": hyper_params["val_generated_image_dir"],
        "num_eval_rounds": num_eval_rounds,
        "blip_model_path": hyper_params.get("blip_model", ""),
        "dialogue_format": hyper_params.get("dialogue_format", ""),
    }

    gen_feats = _create_or_load_generated_cache_blip(
        model=blip_model,
        queries_path=hyper_params["val_queries_path"],
        gen_image_dir=hyper_params["val_generated_image_dir"],
        num_eval_rounds=num_eval_rounds,
        device=device,
        input_size=hyper_params["input_size"],
        batch_size=hyper_params["batch_size"],
        feature_dim=feature_dim,
        cache_path=cache_gen_path,
        cache_force_rebuild=cache_force,
        meta=gen_meta
    )
    # gen_feats: [num_queries, num_eval_rounds, feature_dim] on device

    experiments = {
        "BLIP_Text_Only": "text",
        "BLIP_Image_Only": "image",
        "BLIP_DAR": "dar",
        "BLIP_Fused_Feature": "fused_feature"
    }
    experiments_names = list(experiments.keys())
    all_rounds_recalls = {name: [] for name in experiments_names}

    recall_k_for_excel = 10
    model_prefix = "EMA" if is_ema else "Reg"
    wandb_val_logs = {}

    for dl in range(num_eval_rounds):
        val_queries_dataset.set_dialog_length(dl)

        val_loader = DataLoader(
            val_queries_dataset,
            batch_size=hyper_params["batch_size"],
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        round_scalar_values = []
        round_expert_probs = []

        exp_recalls_per_round = {name: [] for name in experiments_names}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val Round {dl}"):

                # ---- target ids ----
                target_ids = [path_to_id_map.get(p, -1) for p in batch["target_path"]]
                target_ids = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(1)

                # ---- text features ----
                text_inputs = blip_model.tokenizer(
                    text=list(batch["text"]),
                    padding="longest",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    text_features = blip_model.get_text_features(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"]
                    )

                    qidx = torch.tensor(batch["query_idx"], dtype=torch.long, device=device)
                    gen_features = gen_feats[qidx, dl]  # [B, D]

                    for name, method_type in experiments.items():
                        total_scores , aux_info= _calculate_fused_scores_blip(
                            method=method_type,
                            text_features=text_features,
                            gen_features=gen_features,
                            corpus_features=corpus_vectors,
                            dialog_length=dl,
                            fusion_strategy=hyper_params["fusion_strategy"],
                            Adafuse_model=Adafuse
                        )
                        arg_ranks = torch.argsort(total_scores, descending=True, dim=1).long()
                        target_recall = _calculate_ranks(arg_ranks, target_ids)
                        exp_recalls_per_round[name].append(target_recall)
                    if method_type == 'fused_feature' and aux_info:
                        if "scalar_values" in aux_info:

                            round_scalar_values.append(aux_info["scalar_values"].cpu())
                        if "router_probs" in aux_info:
                            round_expert_probs.append(aux_info["router_probs"].cpu())

        for name in experiments.keys():
            if exp_recalls_per_round[name]:
                all_rounds_recalls[name].append(torch.cat(exp_recalls_per_round[name]))
            else:
                all_rounds_recalls[name].append(torch.tensor([], dtype=torch.long, device=device))
        if len(round_scalar_values) > 0:

            all_scalars = torch.cat(round_scalar_values).float() 
            all_probs = torch.cat(round_expert_probs).float()  
            
            avg_scalar = all_scalars.mean().item()
            std_scalar = all_scalars.std().item()
            avg_expert_usage = all_probs.mean(dim=0) # [4]
            
            monitor_prefix = f"val_monitor_{model_prefix.lower()}" # e.g., val_monitor_reg
            
            wandb_val_logs[f"{monitor_prefix}/R{dl}_scalar_mean"] = avg_scalar
            wandb_val_logs[f"{monitor_prefix}/R{dl}_scalar_std"] = std_scalar
            
            for exp_i, usage in enumerate(avg_expert_usage):
                wandb_val_logs[f"{monitor_prefix}/R{dl}_expert_{exp_i}_load"] = usage.item()

    epoch_results_for_excel = {}


    for name, results_per_round in all_rounds_recalls.items():
        indep_r10_list = []

        print(f"  --- Independent Per Round (R@10) ---")
        for dl, recalls in enumerate(results_per_round):
            total_queries = len(recalls)
            rate = 0.0
            if total_queries > 0:
                num_hits = (recalls < recall_k_for_excel).sum().item()
                rate = (num_hits * 100.0 / total_queries)
            indep_r10_list.append(rate)
            print(f"\tRound {dl}: {rate:.2f}%")
            # wandb_val_logs[f"z_{model_prefix.lower()}_{name}_Indep_R{dl}_R@10"] = rate

        epoch_results_for_excel[f"{model_prefix}_{name}_Indep"] = indep_r10_list

        print(f"  --- Cumulative (R@10) ---")
        all_recalls_flat = torch.cat([r.detach().cpu() for r in results_per_round if r.numel() > 0]) \
            if any(r.numel() > 0 for r in results_per_round) else torch.tensor([], dtype=torch.long)

        if all_recalls_flat.numel() > 0:
            cumulative_results_tensor = cumulative_hits_per_round(
                all_recalls_flat,
                num_rounds=num_eval_rounds,
                hitting_recall=recall_k_for_excel
            )
            cumulative_results = cumulative_results_tensor.tolist()
        else:
            cumulative_results = [0.0] * num_eval_rounds

        epoch_results_for_excel[f"{model_prefix}_{name}_Cumul"] = cumulative_results
        
        for dl, rate in enumerate(cumulative_results):
            print(f"\tUp to Round {dl}: {rate:.2f}%")
            # wandb_val_logs[f"z_{model_prefix.lower()}_{name}_Cumul_R{dl}_R@10"] = rate

    best_metric_for_checkpoint = 0.0
    fusion_strategy = hyper_params["fusion_strategy"]

    fused_metric_name = f"{model_prefix}_BLIP_Fused_Feature_Indep"
    dar_metric_name = f"{model_prefix}_BLIP_DAR_Indep"

    if (fusion_strategy in ["Adafuse", "add"]) and fused_metric_name in epoch_results_for_excel:
        best_metric_for_checkpoint = epoch_results_for_excel[fused_metric_name][-1]
    elif dar_metric_name in epoch_results_for_excel:
        best_metric_for_checkpoint = epoch_results_for_excel[dar_metric_name][-1]


    return best_metric_for_checkpoint, epoch_results_for_excel, wandb_val_logs

def blip_collate_fn(batch: list, tokenizer: BertTokenizer):

    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        return None, None, None

    ref_imgs_1, target_imgs, captions = zip(*batch)
    
    ref_pixel_values_1 = torch.stack(ref_imgs_1)
    
    target_pixel_values = torch.stack(target_imgs)

    text_inputs = tokenizer(
            text=list(captions), 
            padding="longest", 
            truncation=True,    
            max_length=256,     
            return_tensors="pt"
        )
    
    return ref_pixel_values_1, target_pixel_values, text_inputs
    
def train_blip_finetune(
    **hyper_params):

    train_json_path = hyper_params['train_json_path']
    # val_json_path = hyper_params['val_json_path']
    projection_dim = hyper_params['projection_dim']
    hidden_dim = hyper_params['hidden_dim']
    num_epochs = hyper_params['num_epochs']
    Adafuse_lr = hyper_params['Adafuse_lr']
    batch_size = hyper_params['batch_size']
    validation_frequency = hyper_params['validation_frequency']
    save_training = hyper_params['save_training']
    resume_from = hyper_params.get('resume_from')


    training_mode = hyper_params['training_mode']
    fusion_strategy = hyper_params['fusion_strategy']

    if training_mode == 'blip_only' and fusion_strategy == 'Adafuse' and not hyper_params.get('Adafuse_checkpoint_path'):
        raise ValueError("在'blip_only'模式下使用'Adafuse'策略时, 必须通过 --Adafuse-checkpoint-path 提供一个预训练好的Adafuse模型路径。")
    if training_mode == 'end_to_end' and fusion_strategy == 'add':
        raise ValueError("'end_to_end'模式只能与'Adafuse'策略配合使用，不能使用'add'策略。")

    if training_mode == 'end_to_end':
        model_name_prefix = 'e2e'
    else:  
        model_name_prefix = 'blip'

    blip_model = LocalBlipForRetrieval(
        med_config='blip_my/configs/med_config.json', 
        image_size=hyper_params['input_size'],                  
        vit='base',                         
        embed_dim=256,                       
        pretrained_path=hyper_params['blip_model'], 
        vit_grad_ckpt=True,                   
        vit_ckpt_layer=12,
        vit_drop_path_rate=hyper_params['vit_drop_path_rate'],
        bert_attention_dropout=hyper_params['bert_attention_dropout'],
        bert_hidden_dropout=hyper_params['bert_hidden_dropout']
    )

    print(f"  ViT DropPath Rate: {blip_model.vit_drop_path_rate}")
    print(f"  BERT Attention Dropout: {blip_model.text_encoder.config.attention_probs_dropout_prob}")
    print(f"  BERT Hidden Dropout: {blip_model.text_encoder.config.hidden_dropout_prob}")
    print("----------------------")

    blip_model.text_encoder.encoder.gradient_checkpointing = True
    blip_model.to(device) 

    feature_dim = blip_model.text_proj.out_features 

    training_path = Path(hyper_params['log_base_dir']) / hyper_params['experiment_name']
    training_path.mkdir(exist_ok=True, parents=True)
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(hyper_params, file, sort_keys=True, indent=4)
    train_log_path = training_path / 'train_metrics.csv'
    if resume_from and train_log_path.exists():
        training_log_frame = pd.read_csv(train_log_path)
    else:

        training_log_frame = pd.DataFrame()
    val_excel_path = training_path / 'validation_summary.xlsx'
    validation_dataframes = {}
    if resume_from and val_excel_path.exists():
        validation_dataframes = pd.read_excel(str(val_excel_path), sheet_name=None, index_col=0)

    loss_components_to_use = set(hyper_params['loss_components'])
    loss_weights = dict(zip(hyper_params['loss_components'], hyper_params['loss_weights']))

    enable_ref_ref = "ref_ref" in loss_components_to_use
    train_transform = build_blip_transform(is_train=True, input_size=hyper_params['input_size'])
    train_dataset = ComposedRetrievalDataset(
        json_file_path=train_json_path, 
        pil_transform=train_transform,
        dialogue_format=hyper_params['dialogue_format'], 
        dialogue_round=hyper_params['dialogue_round'],
        use_random_rounds=hyper_params['use_random_rounds'],
        use_caption_masking=hyper_params['use_caption_masking'],
        caption_masking_prob=hyper_params['caption_masking_prob'],
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        collate_fn=partial(blip_collate_fn, tokenizer=blip_model.tokenizer), 
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )

    Adafuse = None
    if training_mode == 'end_to_end' or fusion_strategy == 'Adafuse':

        Adafuse = Adafuse(feature_dim, hyper_params['projection_dim'], hyper_params['hidden_dim']).to(device)
        
        Adafuse_ckpt_path = hyper_params.get('Adafuse_checkpoint_path')
        if Adafuse_ckpt_path and Path(Adafuse_ckpt_path).exists():

            state_dict = torch.load(Adafuse_ckpt_path, map_location=device)

            if 'Adafuse_state_dict' in state_dict:
                Adafuse.load_state_dict(state_dict['Adafuse_state_dict'])
            elif 'model_state_dict' in state_dict:
                 Adafuse.load_state_dict(state_dict['model_state_dict'])
            else:
                 Adafuse.load_state_dict(state_dict)

    if training_mode == 'blip_only' and Adafuse:

        Adafuse.eval()
        for param in Adafuse.parameters():
            param.requires_grad = False

    elif training_mode == 'end_to_end' and Adafuse:

        Adafuse.train()
        for param in Adafuse.parameters(): 
            param.requires_grad = True
        
        blip_model.eval()
        for param in blip_model.parameters():
            param.requires_grad = False

        if Adafuse:
            Adafuse.train()
            for param in Adafuse.parameters():
                param.requires_grad = True
        else:
            raise ValueError("error")
        
    model_ema = None
    if hyper_params.get('model_ema', False):
        model_ema = ModelEma(
        blip_model,
        decay=hyper_params['model_ema_decay'],
        device='cpu' if False else '',
        resume=''
        )
    Adafuse_ema = None

    if Adafuse and hyper_params.get('model_ema', False):

        Adafuse_ema = ModelEma(
            Adafuse,
            decay=hyper_params['model_ema_decay'],
            device='cpu' if False else '',
            resume=''
        )

    params_to_optimize = []
    if training_mode != 'Adafuse_only':

        blip_lr = hyper_params['blip_lr']
        layer_decay = hyper_params['layer_decay']
        if layer_decay < 1.0:
            print(f"Enabling layer-wise learning rate decay for BLIP: {layer_decay}")
            num_logical_layers = blip_model.text_encoder.config.num_hidden_layers + 2
            num_layers = blip_model.text_encoder.config.num_hidden_layers

            lr_scales = list(layer_decay ** (num_logical_layers - 1 - i) for i in range(num_logical_layers))

            print(f"BLIP logical layers: {num_logical_layers} (V/T both have {num_layers} layers)")
            print(f"LR Scales: [{lr_scales[0]:.2e} (Embed) ... {lr_scales[-1]:.2e} (Head)]")


            # 3d. Create assigner, use lambda to pass extra layer count parameter
            layer_decay_assigner = LayerDecayValueAssigner(
                lr_scales, 
                scale_handler=get_num_layer_for_blip 
            )

            blip_param_groups = get_parameter_groups(
                blip_model, 
                hyper_params['weight_decay'], 
                blip_model.no_weight_decay(), 
                get_num_layer=layer_decay_assigner.get_layer_id,
                get_layer_scale=layer_decay_assigner.get_scale
            )

        else:
            print(f"Layer-wise learning rate decay disabled (layer_decay = {layer_decay}).")
            blip_param_groups = get_parameter_groups(
                blip_model, 
                hyper_params['weight_decay'], 
                blip_model.no_weight_decay()
            )

        for group in blip_param_groups:
            group['lr'] = blip_lr * group.get('lr_scale', 1.0)
        params_to_optimize.extend(blip_param_groups)
    else:
        print("Note: BLIP model parameters not added to optimizer (Adafuse_only mode)")

    if training_mode in ['end_to_end', 'Adafuse_only']:
        Adafuse_lr = hyper_params['Adafuse_lr'] # Use separate learning rate here
        print(f"Adding Adafuse parameters to optimizer, learning rate: {Adafuse_lr}")
        
        Adafuse_param_groups = get_parameter_groups(Adafuse, hyper_params['weight_decay'])
        for group in Adafuse_param_groups:
            group['lr'] = Adafuse_lr
        params_to_optimize.extend(Adafuse_param_groups)

    optimizer = optim.AdamW(params_to_optimize, eps=1e-8, betas=(0.9, 0.999))

    warmup_epochs = hyper_params.get('warmup_epochs', 2) # Default to 2 epochs
    update_freq = hyper_params['update_freq']
    steps_per_epoch = len(train_loader) // update_freq
    
    num_training_steps = steps_per_epoch * num_epochs
    num_warmup_steps = steps_per_epoch * warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_recall_at_10 = 0.0

    if resume_from and Path(resume_from).exists():
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            
            # Define placeholder keys to exclude
            keys_to_ignore = ["itm_head.weight", "itm_head.bias"]
            for key in keys_to_ignore:
                if key in state_dict:
                    del state_dict[key]

            blip_model.load_state_dict(state_dict, strict=True)
            print("BLIP model weights loaded.")
        else:
            # If 'model' key is missing
            if training_mode == 'Adafuse_only':
                print("Info: Checkpoint does not contain BLIP weights (Normal for Adafuse_only). BLIP remains in original pretrained state.")
            else:
                # Warning if end_to_end mode loads a checkpoint without BLIP weights
                print("[Warning] Current mode requires BLIP weights, but they are missing in checkpoint! Continuing with initialized pretrained weights.")
        
        # 2. Handle EMA
        if model_ema is not None:
            if 'model_ema_state_dict' in checkpoint:
                model_ema.ema.load_state_dict(checkpoint['model_ema_state_dict'])
            elif training_mode == 'Adafuse_only':
                 print("Info: Checkpoint does not contain BLIP EMA weights.")
        if Adafuse and 'Adafuse_state_dict' in checkpoint:
            Adafuse.load_state_dict(checkpoint['Adafuse_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state restored successfully!")
        else:
            print("Warning: Scheduler state not found in checkpoint, restarting scheduler.")
        if model_ema is not None and 'model_ema_state_dict' in checkpoint:
            model_ema.ema.load_state_dict(checkpoint['model_ema_state_dict'])
        
        if Adafuse_ema is not None and 'Adafuse_ema_state_dict' in checkpoint:
            Adafuse_ema.ema.load_state_dict(checkpoint['Adafuse_ema_state_dict'])
            print("Adafuse EMA state restored successfully!")
        start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
        
        best_recall_at_10 = checkpoint.get('best_recall_at_10', 0.0)      
        print(f"Resume successful! Training starts from Epoch {start_epoch}. Best Recall@10: {best_recall_at_10:.2f}%")

    if start_epoch == 0:
        print(f"\n--- Running Zero-Shot Validation (Epoch -1) ---")
        
        blip_model.eval() 
        all_epoch_metrics = {}
        

        with torch.no_grad():
            current_r10, regular_metrics_log, wandb_zs_logs = run_eval4_validation(
                blip_model=blip_model,
                Adafuse=Adafuse,
                hyper_params=hyper_params,
                epoch=-1,  # Use -1 as Epoch number
                device=device,
                is_ema=False
            ) 
        wandb_zs_logs['epoch'] = -1
        wandb.log(wandb_zs_logs)
        all_epoch_metrics.update(regular_metrics_log)
        print(f"Zero-shot Base Model R@10 (Round 10, DAR): {current_r10:.2f}%")

 
        col_names = [f'Round {i}' for i in range(11)] 
        for sheet_name, new_data_list in all_epoch_metrics.items():
            
            # 3. Convert new data to DataFrame, index by current epoch
            new_row_df = pd.DataFrame([new_data_list], columns=col_names, index=[-1])
            new_row_df.index.name = "Epoch"
            
            # 4. Check if sheet exists in validation_dataframes
            if sheet_name in validation_dataframes:
                # Append new row if exists
                existing_df = validation_dataframes[sheet_name]
                # Check if epoch exists, overwrite
                if -1 in existing_df.index: 
                    existing_df.loc[-1] = new_data_list 
                else:
                    validation_dataframes[sheet_name] = pd.concat([existing_df, new_row_df])
            else:
                # Create new table if not exists
                validation_dataframes[sheet_name] = new_row_df 
        
        with pd.ExcelWriter(str(val_excel_path), engine='openpyxl') as writer:
                for sheet_name, df in validation_dataframes.items():
                    df.to_excel(writer, sheet_name=sheet_name) 

        print(f"--- Zero-Shot Validation completed, results saved ---")

    print('Training loop started')
    for epoch in range(start_epoch, num_epochs):
        if training_mode == 'Adafuse_only':
            blip_model.eval()          # Key: Always disable dropout/droppath
            if Adafuse:
                Adafuse.train()       # Key: Switch back after validation
        elif training_mode == 'end_to_end':
            blip_model.train()
            if Adafuse:
                Adafuse.train()
        elif training_mode == 'blip_only':
            blip_model.train()
            if Adafuse:
                Adafuse.eval()
        else:
            raise ValueError(f"Unknown training_mode: {training_mode}")

        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        for loss_name in hyper_params['loss_components']:
            train_running_results[f'accumulated_loss_{loss_name}'] = 0.0
        train_bar = tqdm(train_loader, ncols=150)
        
        for i, batch in enumerate(train_bar):
            # 1. Safety check

            ref_pixel_values_1, target_pixel_values, text_inputs = batch

            images_in_batch = ref_pixel_values_1.size(0)
            step = len(train_bar) * epoch + train_bar.n
            
            # 3. Data transfer
            ref_pixel_values = ref_pixel_values_1.to(device)
            target_pixel_values = target_pixel_values.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            with torch.cuda.amp.autocast():
                # --- A. Feature Extraction ---
                # 1. Extract base features (Ref1, Target, Text)
                if training_mode == "Adafuse_only":
                    with torch.no_grad():
                        reference_features = blip_model.get_image_features(ref_pixel_values)
                        target_features = blip_model.get_image_features(target_pixel_values)
                        text_features = blip_model.get_text_features(**text_inputs)
                else:
                    reference_features = blip_model.get_image_features(ref_pixel_values)
                    target_features = blip_model.get_image_features(target_pixel_values)
                    text_features = blip_model.get_text_features(**text_inputs)
                
                # --- B. Normalization and Preparation ---
                blip_logit_scale_exp = blip_model.logit_scale.exp()
                fused_logit_scale_exp = blip_logit_scale_exp

                # Switch to Adafuse's own scale only in 'Adafuse_only' mode
                if training_mode == 'Adafuse_only' and Adafuse is not None and hasattr(Adafuse, 'logit_scale'):
                    fused_logit_scale_exp = Adafuse.logit_scale.exp()
                ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                
                # L2 normalization
                reference_features_norm = F.normalize(reference_features, dim=-1, eps=1e-6)
                target_features_norm = F.normalize(target_features, dim=-1, eps=1e-6)
                text_features_norm = F.normalize(text_features, dim=-1, eps=1e-6)
                
                # --- C. Compute Fused Features (if needed) ---
                fused_features_norm = None
                aux_info = {}
                if 'fused_tgt' in loss_components_to_use:
                    if hyper_params['fusion_strategy'] == 'Adafuse':
                        # Adafuse handles fusion internally
                        fused_raw, aux_info = Adafuse.combine_features(reference_features, text_features)
                    else: 
                        # 'add' strategy: Simple addition (Note: using unnormalized features to match BEiT3 logic)
                        fused_raw = reference_features + text_features
                    
                    fused_features_norm = F.normalize(fused_raw, dim=-1, eps=1e-6)

                # --- D. Component-wise Loss Calculation ---
                all_losses = {}
                criterion = nn.CrossEntropyLoss()

                # 1. [Ref_Image <-> Target_Image]
                if 'ref_tgt' in loss_components_to_use:
                    l_r2t = blip_logit_scale_exp * reference_features_norm @ target_features_norm.T
                    l_t2r = blip_logit_scale_exp * target_features_norm @ reference_features_norm.T
                    all_losses['ref_tgt'] = (criterion(l_r2t, ground_truth) + criterion(l_t2r, ground_truth)) / 2

                # 2. [Text <-> Target_Image]
                if 'text_tgt' in loss_components_to_use:
                    l_txt2t = blip_logit_scale_exp * text_features_norm @ target_features_norm.T
                    l_t2txt = blip_logit_scale_exp * target_features_norm @ text_features_norm.T
                    all_losses['text_tgt'] = (criterion(l_txt2t, ground_truth) + criterion(l_t2txt, ground_truth)) / 2

                # 3. [Fused <-> Target_Image]
                if 'fused_tgt' in loss_components_to_use:
                    l_f2t = fused_logit_scale_exp * fused_features_norm @ target_features_norm.T
                    l_t2f = fused_logit_scale_exp * target_features_norm @ fused_features_norm.T
                    all_losses['fused_tgt'] = (criterion(l_f2t, ground_truth) + criterion(l_t2f, ground_truth)) / 2

                # 4. [Ref_Image <-> Text] (Cross-modal alignment)
                if 'ref_text' in loss_components_to_use:
                    l_r2txt = blip_logit_scale_exp * reference_features_norm @ text_features_norm.T
                    l_txt2r = blip_logit_scale_exp * text_features_norm @ reference_features_norm.T
                    all_losses['ref_text'] = (criterion(l_r2txt, ground_truth) + criterion(l_txt2r, ground_truth)) / 2

                # --- E. Weighted Sum ---
                total_loss = 0.0
                if not all_losses:
                    raise ValueError(f"Config Error: No loss computed! loss_components: {loss_components_to_use}")

                for name, val in all_losses.items():
                    weight = loss_weights.get(name, 1.0)
                    total_loss += val * weight

                loss = total_loss
                unscaled_loss = loss.detach() # For logging

                # Log component losses
                for name, val in all_losses.items():
                    key = f'accumulated_loss_{name}'
                    if key in train_running_results:
                        train_running_results[key] += val.item() * images_in_batch

                loss = loss / hyper_params['update_freq'] # Gradient accumulation normalization


            scaler.scale(loss).backward()
            if (i + 1) % hyper_params['update_freq'] == 0:
                max_norm = hyper_params['clip_grad']
                if max_norm > 0:
                    # Must unscale gradients before step
                    scaler.unscale_(optimizer)
                # Clip BLIP gradients only if training BLIP
                if training_mode != 'Adafuse_only':
                    torch.nn.utils.clip_grad_norm_(blip_model.parameters(), max_norm)
                
                # Clip Adafuse gradients if involved in training
                if training_mode in ['end_to_end', 'Adafuse_only']: 
                    torch.nn.utils.clip_grad_norm_(Adafuse.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                if model_ema is not None:
                    model_ema.update(blip_model)
                if Adafuse_ema is not None:
                    Adafuse_ema.update(Adafuse)
                optimizer.zero_grad() 
                scheduler.step()


            # --- Modified Code ---
            current_step_logs = {
                "train/step": step,
                "train/epoch": epoch + i / len(train_loader),
                "train/step_loss": unscaled_loss.detach().item()
            }

            # 1. Log base learning rate (usually BLIP)
            if len(optimizer.param_groups) > 0:
                current_step_logs["train/lr_blip_base"] = optimizer.param_groups[0]['lr']

            # 2. Log Adafuse learning rate (added last to optimizer)
            # In end_to_end mode, the last one [-1] is Adafuse
            if training_mode in ['end_to_end', 'Adafuse_only']:
                current_step_logs["train/lr_Adafuse"] = optimizer.param_groups[-1]['lr']
            
            # Log component losses (optional, detailed)
            for name, val in all_losses.items():
                current_step_logs[f"train/loss_component_{name}"] = val.item()
            # 1. Log Adafuse temperature scale (if exists)
            if Adafuse is not None and hasattr(Adafuse, 'logit_scale'):
                # [A] Log actual effective scale (e.g., approx 14.28 initially)
                current_step_logs['train/Adafuse_scale_exp'] = Adafuse.logit_scale.exp().item()
                
                # [B] Log raw parameter value (e.g., approx 2.65 initially)
                current_step_logs['train/Adafuse_scale_raw'] = Adafuse.logit_scale.item()

            # 2. (Optional) Log BLIP Scale for comparison
            if hasattr(blip_model, 'logit_scale'):
                 current_step_logs['train/blip_scale_exp'] = blip_model.logit_scale.exp().item()
            # 1. Monitor Dynamic Scalar (Lambda)
            if "scalar_values" in aux_info:
                # Detach and convert to float
                scalar_val = aux_info["scalar_values"].detach().float()
                
                # Calculate batch mean (overall tendency: text or image)
                current_step_logs["train/scalar_mean"] = scalar_val.mean().item()
                
                # Calculate batch std (dynamism: check for collapse)
                # If Std stays near 0, gating might have collapsed
                current_step_logs["train/scalar_std"] = scalar_val.std().item()

            # 2. Monitor MoE Expert Load (Router Probs)
            if "router_probs" in aux_info:
                probs = aux_info["router_probs"].detach().float() # [Batch, 4]
                
                # Calculate average usage of each expert in current batch
                # Average over Batch dim -> [4]
                expert_loads = probs.mean(dim=0) 
                
                for idx, load in enumerate(expert_loads):
                    current_step_logs[f"train/expert_{idx}_load"] = load.item()

            wandb.log(current_step_logs) # <--- WandB Core Call

            update_train_running_results(train_running_results, unscaled_loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)
        
        train_epoch_loss = train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']
        epoch_log_data = {
            "epoch": epoch, 
            "train/epoch_loss": train_epoch_loss
        }
        

        epoch_metrics_data = {'epoch': epoch, 'train_epoch_loss': train_epoch_loss}
        for loss_name in hyper_params['loss_components']:
            key = f'accumulated_loss_{loss_name}'
            if key in train_running_results and train_running_results['images_in_epoch'] > 0:
                val = train_running_results[key] / train_running_results['images_in_epoch']
                epoch_metrics_data[f'loss_{loss_name}'] = val
                epoch_log_data[f"train/loss_{loss_name}_epoch"] = val
        wandb.log(epoch_log_data) # Upload Epoch Summary

        training_log_frame = pd.concat([
            training_log_frame,
            pd.DataFrame(data=epoch_metrics_data, index=[0])
        ])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            all_epoch_metrics = {}
            
            with torch.no_grad():
                current_r10, regular_metrics_log, wandb_reg_logs = run_eval4_validation(
                    blip_model=blip_model,
                    Adafuse=Adafuse,
                    hyper_params=hyper_params,
                    epoch=epoch,
                    device=device,
                    is_ema=False
                )
            wandb_reg_logs['epoch'] = epoch
            wandb.log(wandb_reg_logs)
            all_epoch_metrics.update(regular_metrics_log)
            print(f"Base Model R@10 (Round 10, DAR): {current_r10:.2f}%")
            current_best_metric = current_r10

            if model_ema is not None:
                print("\n--- Evaluating EMA Model ---")
                model_ema.ema.eval() # Ensure EMA model is in eval mode
                
                ema_eval_Adafuse = None
                if Adafuse_ema is not None:
                    Adafuse_ema.ema.eval()
                    ema_eval_Adafuse = Adafuse_ema.ema

                with torch.no_grad():
                    ema_r10, ema_metrics_log, wandb_ema_logs = run_eval4_validation(
                        blip_model=model_ema.ema,
                        Adafuse=ema_eval_Adafuse,
                        hyper_params=hyper_params,
                        epoch=epoch,
                        device=device,
                        is_ema=True
                    )
                wandb_ema_logs['epoch'] = epoch
                wandb.log(wandb_ema_logs)
                all_epoch_metrics.update(ema_metrics_log)
                print(f"EMA Model R@10 (Round 10, DAR): {ema_r10:.2f}%")
                current_best_metric = ema_r10 
            col_names = [f'Round {i}' for i in range(11)]
            for sheet_name, new_data_list in all_epoch_metrics.items():
                
                # 3. Convert new data to DataFrame, index by current epoch
                new_row_df = pd.DataFrame([new_data_list], columns=col_names, index=[epoch])
                new_row_df.index.name = "Epoch"
                
                # 4. Check if sheet exists in validation_dataframes
                if sheet_name in validation_dataframes:
                    # Append new row if exists
                    existing_df = validation_dataframes[sheet_name]
                    # Check if epoch exists, overwrite
                    if epoch in existing_df.index:
                        existing_df.loc[epoch] = new_data_list
                    else:
                        validation_dataframes[sheet_name] = pd.concat([existing_df, new_row_df])
                else:
                    # Create new table if not exists
                    validation_dataframes[sheet_name] = new_row_df
            with pd.ExcelWriter(str(val_excel_path), engine='openpyxl') as writer:
                    for sheet_name, df in validation_dataframes.items():
                        df.to_excel(writer, sheet_name=sheet_name)

            if save_training:
                if current_best_metric > best_recall_at_10:
                    best_recall_at_10 = current_best_metric
                    print(f"New best model found (R@10: {best_recall_at_10:.2f}%), saving 'best' checkpoint...")
                    save_checkpoint(f'{model_name_prefix}_best', epoch, blip_model, optimizer, scaler, best_recall_at_10, training_path,scheduler, training_mode,Adafuse, model_ema, Adafuse_ema)

                print(f"Saving checkpoint for Epoch {epoch}... (Best R@10: {best_recall_at_10:.2f}%)")
                save_checkpoint(f'{model_name_prefix}_{epoch}', epoch, blip_model, optimizer, scaler, best_recall_at_10, training_path, scheduler, training_mode,Adafuse, model_ema, Adafuse_ema)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default=Config.experiment_name, help="Experiment name (WandB Run ID)")
    parser.add_argument("--wandb_project", type=str, default=Config.wandb_project, help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=Config.wandb_entity, help="WandB entity (username or team name)")
    parser.add_argument("--wandb_mode", type=str, default=Config.wandb_mode, 
                        choices=['online', 'offline', 'disabled'], 
                        help="WandB running mode")
    parser.add_argument("--log_base_dir", type=str, default=Config.log_base_dir, help="Root directory for all experiment logs")
    
    # --- Data and Model Path Parameters ---
    parser.add_argument("--dialogue_format", type=str, default=Config.dialogue_format, choices=['Summarized', 'VisDial'],
                        help="Dialogue format to use (Summarized or VisDial)")
    parser.add_argument("--dialogue_round", type=int, default=Config.dialogue_round,
                        help="Dialogue rounds to use (0-10)")
    parser.add_argument('--use_caption_masking', action='store_true', default=Config.use_caption_masking, 
                        help="Enable random caption masking")
    parser.add_argument("--caption_masking_prob", type=float, default=Config.caption_masking_prob, 
                        help="Probability of masking round 0")
    parser.add_argument('--use_random_rounds', action='store_true',default=Config.use_random_rounds, help="Enable random dialogue rounds (R3 strategy)")
    parser.add_argument("--train_json_path", type=str, default=Config.train_json_path)
    parser.add_argument("--val_corpus_json-path", type=str, default=Config.val_corpus_json_path, help="Path to the large-scale validation corpus")
    parser.add_argument("--val_queries_path", type=str, default=Config.val_queries_path)
    parser.add_argument("--val_generated_image_dir", type=str, default=Config.val_generated_image_dir)
    parser.add_argument("--blip_model", type=str, default=Config.blip_model, help="BLIP model name or path")

    # --- ADaFuSE Architecture Parameters ---
    parser.add_argument("--projection_dim", default=Config.projection_dim, type=int, help='ADaFuSE projection dimension')
    parser.add_argument("--hidden_dim", default=Config.hidden_dim, type=int, help="ADaFuSE hidden dimension")
    parser.add_argument("--Adafuse_checkpoint_path", type=str, default=Config.Adafuse_checkpoint_path)
    
    # --- Training Hyperparameters ---
    parser.add_argument("--training_mode", type=str, default=Config.training_mode, choices=['blip_only', 'end_to_end', 'Adafuse_only'])
    parser.add_argument("--num_epochs", default=Config.num_epochs, type=int)
    parser.add_argument("--blip_lr", default=Config.blip_lr, type=float)
    parser.add_argument("--Adafuse_lr", default=Config.Adafuse_lr, type=float)
    parser.add_argument("--weight_decay", default=Config.weight_decay, type=float, help="Weight decay value (not applied to bias and LayerNorm)")
    parser.add_argument("--warmup_epochs", default=Config.warmup_epochs, type=int, help="Number of warmup epochs")
    parser.add_argument("--layer_decay", default=Config.layer_decay, type=float, help="Layer-wise learning rate decay factor (enabled if < 1.0)")
    parser.add_argument("--batch_size", default=Config.batch_size, type=int)
    parser.add_argument("--update_freq", default=Config.update_freq, type=int)
    parser.add_argument("--clip_grad", default=Config.clip_grad, type=float)
    parser.add_argument("--model_ema", action='store_true', default=Config.model_ema)
    parser.add_argument("--model_ema-decay", type=float, default=Config.model_ema_decay)
    parser.add_argument("--validation_frequency", default=Config.validation_frequency, type=int)

    parser.add_argument("--vit_drop_path_rate", type=float, default=Config.vit_drop_path_rate,
                        help="DropPath rate for ViT")
    parser.add_argument("--bert_attention_dropout", type=float, default=Config.bert_attention_dropout,
                        help="Dropout rate for BERT attention layers")
    parser.add_argument("--bert_hidden_dropout", type=float, default=Config.bert_hidden_dropout,
                        help="Dropout rate for BERT hidden layers")

    parser.add_argument("--resume_from", type=str, default=Config.resume_from, help="Resume training from a specific checkpoint")
    parser.add_argument("--save_training", action='store_true', default=Config.save_training)
    
    parser.add_argument("--fusion_strategy", type=str, default=Config.fusion_strategy, choices=['add', 'Adafuse'])
    # parser.add_argument("--target-ratio", type=float, default=Config.target_ratio)
    # parser.add_argument("--loss-type", type=str, default=Config.loss_type, choices=['crossentropy', 'dual_symmetric_ce','triple_symmetric_ce','quad_symmetric_ce'])
    parser.add_argument("--loss_components", nargs='+', default=Config.loss_components, 
                        help="List of loss components to use")
    parser.add_argument("--loss_weights", nargs='+', type=float, default=Config.loss_weights,
                        help="List of weights corresponding to loss_components")
    parser.add_argument("--input_size", type=int, default=Config.input_size, 
                        help="Input image resolution (e.g., 224 or 384)")
    parser.add_argument("--val_cache_corpus_path", type=str, default=Config.val_cache_corpus_path,
                    help="Path to validation corpus feature cache (.pt); enabled only for Adafuse_only")
    parser.add_argument("--val_cache_gen_path", type=str, default=Config.val_cache_gen_path,
                        help="Path to validation query feature cache (.pt); enabled only for Adafuse_only")
    parser.add_argument("--val_cache_force_rebuild", action="store_true", default=Config.val_cache_force_rebuild,
                        help="Force rebuild validation cache (ignore existing cache files)")


    args = parser.parse_args()

    # Package all arguments into a dictionary for easier management and saving
    training_hyper_params = vars(args)

    if len(training_hyper_params['loss_components']) != len(training_hyper_params['loss_weights']):
        raise ValueError("The number of loss_components and loss_weights must match!")

    run_id = args.experiment_name # Use experiment name as ID for easy resumption
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.experiment_name,
        id=run_id,         # Force specific ID
        resume="allow",    # Resume logging if ID exists
        config=training_hyper_params, # Automatically log all hyperparameters
        mode=args.wandb_mode
    )

    # Start training
    train_blip_finetune(**training_hyper_params)