import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import torch
import random
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from blip_config import Config

BLIP_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def build_blip_transform(is_train, input_size=Config.input_size):
    mean = BLIP_CLIP_MEAN
    std  = BLIP_CLIP_STD
    if is_train:
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform

class ComposedRetrievalDataset(Dataset):

    TRAIN_GEN_ROOT = Path("dataset/query_images") 

    def __init__(self, json_file_path: str, pil_transform: callable = None, 
                 dialogue_format: str = "VisDial", dialogue_round: int = 0,
                 use_random_rounds: bool = False, use_caption_masking: bool = False, 
                 caption_masking_prob: float = 0.0, 
                 **kwargs):

        super().__init__()
        self.json_file_path = Path(json_file_path)
        
        if pil_transform is None:
            self.pil_transform = build_blip_transform(is_train=True, input_size=Config.input_size)
        else:
            self.pil_transform = pil_transform

        self.dialogue_format = dialogue_format
        self.dialogue_round = dialogue_round
        self.use_random_rounds = use_random_rounds
        self.use_caption_masking = use_caption_masking
        self.caption_masking_prob = caption_masking_prob

        self.reference_image_dir = self.TRAIN_GEN_ROOT
        self.reference_filename_prefix = "train-" 

        if not self.json_file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.json_file_path}")
            
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data: List[Dict] = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def _load_ref_image(self, target_filename_stem: str, round_idx: int):

        reference_filename = f"{self.reference_filename_prefix}{target_filename_stem}_{round_idx}.jpg"
        round_folder_name = f"round{round_idx}"
        reference_path = self.reference_image_dir / round_folder_name / reference_filename
        
        # 绝对不使用 try-except 跳过
        if not reference_path.exists():
            raise FileNotFoundError(f"[Critical] Missing training image: {reference_path}. "
                                    f"Please check dataset integrity.")
            
        image = Image.open(reference_path).convert("RGB")
        if self.pil_transform:
            image = self.pil_transform(image)
        return image

    def __getitem__(self, index: int) -> Tuple:
        item_info = self.data[index]
        target_path_str = item_info['img']
        dialog_list = item_info['dialog'] 
        
        max_allowed_round = self.dialogue_round

        # --- 1. 确定当前参考图的轮次 (Current View) ---
        if self.use_random_rounds:
            current_round_index = random.randint(0, max_allowed_round)
        else:
            current_round_index = max_allowed_round

        # --- 3. 构建文本 Caption ---
        caption = ""
        if self.dialogue_format == 'Summarized':
            if current_round_index >= len(dialog_list):
                caption = dialog_list[-1]
            else:
                caption = dialog_list[current_round_index]
        elif self.dialogue_format == 'VisDial':
            max_index = current_round_index + 1
            relevant_dialogs = dialog_list[:max_index]
            
            # Caption Masking (随机丢弃首句以增强鲁棒性)
            if self.use_caption_masking and \
               random.random() < self.caption_masking_prob and \
               len(relevant_dialogs) > 1:
                relevant_dialogs = relevant_dialogs[1:] 
            
            caption = ", ".join(relevant_dialogs)

        # --- 4. 加载图片 (Strict Loading) ---
        
        # 加载 Target Image (GT)
        if not Path(target_path_str).exists():
             raise FileNotFoundError(f"[Critical] Missing target image: {target_path_str}")
        
        target_image = Image.open(target_path_str).convert("RGB")
        if self.pil_transform:
            target_image = self.pil_transform(target_image)
            
        # 加载 Reference Image View 1
        target_filename_stem = Path(target_path_str).stem
        ref_image_1 = self._load_ref_image(target_filename_stem, current_round_index)
        
        return ref_image_1, target_image, caption

def get_blip_transform(input_size=Config.input_size):

    return build_blip_transform(is_train=False, input_size=input_size)

class CorpusDataset(Dataset):

    def __init__(self, json_file_path: str, pil_transform: callable = None):
        super().__init__()
        self.json_file_path = Path(json_file_path)
        
        if pil_transform is None:
            self.pil_transform = build_blip_transform(is_train=False, input_size=Config.input_size)
        else:
            self.pil_transform = pil_transform

        if not self.json_file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.json_file_path}")

        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.image_paths: List[str] = json.load(f)

        self.path_to_id_map: Dict[str, int] = {path: i for i, path in enumerate(self.image_paths)}
        print(f"CorpusDataset: Successfully loaded {len(self.image_paths)} images from {json_file_path}.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        image_path = self.image_paths[index]
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Missing corpus image: {image_path}")
            
        image = Image.open(image_path).convert("RGB")

        if self.pil_transform:
            image = self.pil_transform(image)
            
        return image_path, image
    
class ValidationQueriesDataset(Dataset):
  
    def __init__(self, queries_path: str, generated_image_dir: str):
        self.queries_path = Path(queries_path)
        self.generated_image_dir = Path(generated_image_dir)
        
        if not self.queries_path.exists():
             raise FileNotFoundError(f"Queries file not found: {self.queries_path}")
        
        with open(self.queries_path, 'r', encoding='utf-8') as f:
            self.queries = json.load(f)
            
        self.dialog_length = 0 
        self.dialogue_format = Config.dialogue_format
        self.sep_token = ", "

    def __len__(self) -> int:
        return len(self.queries)

    def set_dialog_length(self, dialog_length: int):
        self.dialog_length = dialog_length

    def __getitem__(self, i: int) -> Dict:
        target_path = self.queries[i]['img']
        
        if self.dialogue_format == 'Summarized':
            text = self.queries[i]['dialog'][self.dialog_length]
        elif self.dialogue_format == 'VisDial':
            text = self.sep_token.join(self.queries[i]['dialog'][:self.dialog_length + 1])

        gen_image_filename = f"{i}_{self.dialog_length}.jpg"
        gen_image_path = (self.generated_image_dir / gen_image_filename).as_posix()
        
        if not Path(gen_image_path).exists():
           
            raise FileNotFoundError(f"Validation dataset missing generated image: {gen_image_path}")

        return {
            'query_idx': i,
            'text': text,
            'target_path': target_path,
            'gen_path': gen_image_path
        }
    
class QueryImageDataset(Dataset):
    def __init__(self, queries: List[Dict], gen_image_dir: str, num_rounds: int, transform: callable):

        self.samples = []
        self.transform = transform
        gen_dir = Path(gen_image_dir)
        
        for query_idx in range(len(queries)):
            for round_idx in range(num_rounds):
                filename = f"{query_idx}_{round_idx}.jpg"
                filepath = gen_dir / filename
                
                if not filepath.exists():
                    raise FileNotFoundError(f"[Strict Mode] Critical Data Missing: {filepath} does not exist!")
                
                self.samples.append((filename, str(filepath)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        filename, filepath = self.samples[idx]
       
        image = Image.open(filepath).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return filename, image