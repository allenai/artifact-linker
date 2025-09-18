#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple, Any


class DownloadBasedBinaryPredictor:
    """
    Binary predictor that uses download counts as the decision criterion.
    Predicts 'connected' if model has high downloads, 'not connected' otherwise.
    """
    
    def __init__(self, download_threshold: int = 1000):
        self.download_threshold = download_threshold
    
    def predict(
        self,
        edge_pairs: List[Tuple[str, str]],
        G=None,
        mode="simple",
        summaries: dict | None = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Predict binary connections based on model download counts.
        
        Args:
            edge_pairs: List of (model, dataset) pairs
            G: Graph (used to get download info if available)
            mode: Prediction mode (ignored for baseline)
            summaries: Summary data (ignored for baseline)
        
        Returns:
            List of prediction results
        """
        results = []
        
        for model_name, dataset_name in edge_pairs:
            try:
                # Get download count from graph node attributes
                downloads = 0
                if G and model_name in G.nodes:
                    downloads = G.nodes[model_name].get('downloads', 0)
                
                # Simple threshold-based prediction
                prediction = downloads >= self.download_threshold
                
                reason = f"Model {model_name} has {downloads} downloads (threshold: {self.download_threshold})"
                
                result = {
                    "prediction": prediction,
                    "reason": reason,
                    "downloads": downloads
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error predicting for ({model_name}, {dataset_name}): {e}")
                results.append(None)
        
        return results


class DownloadBasedRanker:
    """
    Ranker that sorts models by their download counts (highest first).
    """
    
    def __init__(self):
        pass
    
    def rank_models_for_dataset(
        self,
        dataset_name: str,
        G,
        summaries: dict | None = None,
        num_negative_samples: int = 5,
        max_models_to_rank: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        Rank models for a dataset based on download counts.
        
        Args:
            dataset_name: Target dataset name
            G: Graph containing model-dataset connections
            summaries: Summary data (ignored for baseline)
            num_negative_samples: Number of negative samples
            max_models_to_rank: Maximum models to rank
        
        Returns:
            Dictionary with ranked models and metadata
        """
        try:
            # Find neighbor models (positive samples)
            neighbor_models = []
            all_models = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]
            
            for neighbor in G.neighbors(dataset_name):
                if G.nodes[neighbor].get("type") == "model":
                    neighbor_models.append(neighbor)
            
            # Sample negative models
            connected_models = set(neighbor_models)
            unconnected_models = [m for m in all_models if m not in connected_models]
            
            actual_negative_samples = min(num_negative_samples, len(unconnected_models))
            negative_models = random.sample(unconnected_models, actual_negative_samples)
            
            # Combine models
            all_models_to_rank = neighbor_models + negative_models
            if len(all_models_to_rank) > max_models_to_rank:
                if len(neighbor_models) >= max_models_to_rank:
                    all_models_to_rank = neighbor_models[:max_models_to_rank]
                else:
                    remaining_slots = max_models_to_rank - len(neighbor_models)
                    negative_models = negative_models[:remaining_slots]
                    all_models_to_rank = neighbor_models + negative_models
            
            # Sort by download counts (highest first)
            models_with_downloads = []
            for model in all_models_to_rank:
                downloads = G.nodes[model].get('downloads', 0)
                models_with_downloads.append((model, downloads))
            
            # Sort by downloads descending
            models_with_downloads.sort(key=lambda x: x[1], reverse=True)
            ranked_models = [model for model, _ in models_with_downloads]
            
            return {
                "ranked_models": ranked_models,
                "reasoning": f"Ranked {len(ranked_models)} models by download counts (highest first)",
                "dataset_name": dataset_name,
                "neighbor_models": neighbor_models,
                "negative_models": negative_models,
                "total_models_ranked": len(all_models_to_rank),
                "download_counts": {model: downloads for model, downloads in models_with_downloads}
            }
            
        except Exception as e:
            print(f"Error ranking models for dataset {dataset_name}: {e}")
            return None


class DownloadBasedAttributeRanker:
    """
    Attribute ranker that ranks model-dataset pairs by model download counts.
    Higher downloads = higher predicted performance.
    """
    
    def __init__(self):
        pass
    
    def rank_edges_by_attribute(
        self,
        positive_edges: List[Tuple[str, str]],
        G,
        attribute_name: str,
        summaries: dict | None = None,
        max_edges_to_rank: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        Rank model-dataset pairs by model download counts.
        
        Args:
            positive_edges: List of (model, dataset) pairs
            G: Graph containing node information
            attribute_name: Target attribute (used for metadata)
            summaries: Summary data (ignored for baseline)
            max_edges_to_rank: Maximum edges to rank
        
        Returns:
            Dictionary with ranked pairs and metadata
        """
        try:
            # Limit edges if needed
            edges_to_rank = positive_edges[:max_edges_to_rank] if len(positive_edges) > max_edges_to_rank else positive_edges
            
            # Get download counts for each model
            edges_with_downloads = []
            for model, dataset in edges_to_rank:
                downloads = G.nodes[model].get('downloads', 0) if model in G.nodes else 0
                edges_with_downloads.append((model, dataset, downloads))
            
            # Sort by downloads descending
            edges_with_downloads.sort(key=lambda x: x[2], reverse=True)
            
            # Create ranked pairs with expected scores based on normalized downloads
            if edges_with_downloads:
                max_downloads = max(downloads for _, _, downloads in edges_with_downloads)
                min_downloads = min(downloads for _, _, downloads in edges_with_downloads)
                download_range = max_downloads - min_downloads if max_downloads > min_downloads else 1
            else:
                max_downloads = min_downloads = download_range = 1
            
            ranked_pairs = []
            for rank, (model, dataset, downloads) in enumerate(edges_with_downloads, 1):
                # Normalize downloads to 0.1-0.9 range for expected score
                if download_range > 0:
                    normalized_score = 0.1 + 0.8 * (downloads - min_downloads) / download_range
                else:
                    normalized_score = 0.5
                
                ranked_pairs.append({
                    "model": model,
                    "dataset": dataset,
                    "rank": rank,
                    "expected_score": round(normalized_score, 3),
                    "downloads": downloads
                })
            
            return {
                "ranked_pairs": ranked_pairs,
                "reasoning": f"Ranked {len(ranked_pairs)} model-dataset pairs by model download counts",
                "attribute_name": attribute_name,
                "total_edges_ranked": len(edges_to_rank),
                "original_edges_count": len(positive_edges),
                "download_stats": {
                    "max_downloads": max_downloads,
                    "min_downloads": min_downloads,
                    "avg_downloads": sum(downloads for _, _, downloads in edges_with_downloads) / len(edges_with_downloads) if edges_with_downloads else 0
                }
            }
            
        except Exception as e:
            print(f"Error ranking edges by attribute: {e}")
            return None


class RandomBaseline:
    """
    Random baseline for comparison. Makes random predictions/rankings.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def predict_binary(
        self,
        edge_pairs: List[Tuple[str, str]],
        G=None,
        mode="simple",
        summaries: dict | None = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Random binary predictions."""
        results = []
        for model_name, dataset_name in edge_pairs:
            prediction = self.rng.choice([True, False])
            result = {
                "prediction": prediction,
                "reason": "Random baseline prediction",
                "confidence": self.rng.uniform(0.1, 0.9)
            }
            results.append(result)
        return results
    
    def rank_models_for_dataset(
        self,
        dataset_name: str,
        G,
        summaries: dict | None = None,
        num_negative_samples: int = 5,
        max_models_to_rank: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Random model ranking."""
        try:
            # Get models like the download-based ranker
            neighbor_models = []
            all_models = [n for n, d in G.nodes(data=True) if d.get("type") == "model"]
            
            for neighbor in G.neighbors(dataset_name):
                if G.nodes[neighbor].get("type") == "model":
                    neighbor_models.append(neighbor)
            
            connected_models = set(neighbor_models)
            unconnected_models = [m for m in all_models if m not in connected_models]
            
            actual_negative_samples = min(num_negative_samples, len(unconnected_models))
            negative_models = self.rng.sample(unconnected_models, actual_negative_samples)
            
            all_models_to_rank = neighbor_models + negative_models
            if len(all_models_to_rank) > max_models_to_rank:
                all_models_to_rank = self.rng.sample(all_models_to_rank, max_models_to_rank)
            
            # Random shuffle
            ranked_models = all_models_to_rank.copy()
            self.rng.shuffle(ranked_models)
            
            return {
                "ranked_models": ranked_models,
                "reasoning": "Random baseline ranking",
                "dataset_name": dataset_name,
                "neighbor_models": neighbor_models,
                "negative_models": negative_models,
                "total_models_ranked": len(all_models_to_rank)
            }
            
        except Exception as e:
            print(f"Error in random ranking for dataset {dataset_name}: {e}")
            return None
    
    def rank_edges_by_attribute(
        self,
        positive_edges: List[Tuple[str, str]],
        G,
        attribute_name: str,
        summaries: dict | None = None,
        max_edges_to_rank: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Random attribute ranking."""
        try:
            edges_to_rank = positive_edges[:max_edges_to_rank] if len(positive_edges) > max_edges_to_rank else positive_edges
            
            # Random shuffle
            shuffled_edges = edges_to_rank.copy()
            self.rng.shuffle(shuffled_edges)
            
            ranked_pairs = []
            for rank, (model, dataset) in enumerate(shuffled_edges, 1):
                ranked_pairs.append({
                    "model": model,
                    "dataset": dataset,
                    "rank": rank,
                    "expected_score": self.rng.uniform(0.1, 0.9)
                })
            
            return {
                "ranked_pairs": ranked_pairs,
                "reasoning": "Random baseline ranking",
                "attribute_name": attribute_name,
                "total_edges_ranked": len(edges_to_rank),
                "original_edges_count": len(positive_edges)
            }
            
        except Exception as e:
            print(f"Error in random attribute ranking: {e}")
            return None
