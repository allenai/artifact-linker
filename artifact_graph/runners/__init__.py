#!/usr/bin/env python3
"""Unified runners for prediction and ranking tasks."""
from .link_runner import run_link_prediction, run_link_ranking
from .attribute_runner import run_attribute_prediction, run_attribute_ranking

__all__ = [
    "run_link_prediction",
    "run_link_ranking",
    "run_attribute_prediction",
    "run_attribute_ranking",
]
