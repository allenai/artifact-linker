import torch
from typing import Dict, Any

def format_card_info(card_info: Dict[str, Any]) -> str:
    """Format card information as string"""
    if not card_info:
        return "No information available"
    
    parts = []
    if card_info.get('description'):
        parts.append(f"Description: {card_info['description']}")
    if card_info.get('downloads'):
        parts.append(f"Downloads: {card_info['downloads']}")
    if card_info.get('tags'):
        parts.append(f"Tags: {', '.join(card_info['tags'])}")
    
    return "; ".join(parts) if parts else "No information available"

def validate_data_consistency(data: Any) -> bool:
    """Validate data consistency"""
    if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
        return False
    if data.x.size(0) != data.edge_index.max().item() + 1:
        return False
    return True 