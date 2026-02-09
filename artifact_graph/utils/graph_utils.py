#!/usr/bin/env python3
"""
Graph utility functions.

Data preparation functions are organized in:
- link_prediction_utils.py: prepare_link_predictor_dataset
- attribute_prediction_utils.py: prepare_attribute_predictor_dataset  
- ranking_utils.py: prepare_link_ranker_dataset, prepare_attribute_ranker_dataset
"""
from __future__ import annotations


def create_safe_filename(name: str) -> str:
    """Create a filesystem-safe filename from a string."""
    return name.replace("/", "_").replace(":", "_").replace("\\", "_")
