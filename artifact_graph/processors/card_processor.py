import os
import json
from typing import Dict

class CardProcessor:
    def __init__(self, data_dir: str, metadata_dir: str, dataset_json: str):
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.dataset_json = dataset_json
    
    def prepare_cards(self) -> tuple[Dict, Dict]:
        """准备模型和数据集卡片信息"""
        model_cards = {}
        dataset_cards = {}
        
        # 加载数据集卡片
        with open(self.dataset_json, 'r', encoding='utf-8') as f:
            ds_info = json.load(f)
        for ds in ds_info:
            ds_name = ds['id'].split('/')[-1].lower()
            dataset_cards[ds_name] = {
                'description': ds.get('description', ''),
                'downloads': ds.get('downloads', 0),
                'tags': ds.get('tags', [])
            }
        
        # 加载模型卡片
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".json"):
                continue
            model_id = fname[:-5]
            try:
                with open(os.path.join(self.metadata_dir, f"{model_id}.json"), 'r', encoding='utf-8') as f:
                    md = json.load(f)
                model_cards[model_id] = {
                    'description': md.get('description', ''),
                    'downloads': md.get('downloads', 0),
                    'tags': md.get('tags', [])
                }
            except FileNotFoundError:
                continue
        
        return model_cards, dataset_cards 