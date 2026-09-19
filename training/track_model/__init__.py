"""Ensemble track model package for 2023-2025 NA hurricanes.

Pipeline:
  1. read_files   – Parse ECMWF TIGGE XML, extract 51-member tracks
  2. build_pairs  – Build 6h velocity pairs from ensemble tracks
  3. train_markov – Fit per-step Gaussian Markov model
  4. sample_tracks– Generate synthetic ensemble tracks
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .read_files import run_read_tracks
from .build_pairs import run_build_pairs
from .train_markov import run_train_markov
from .sample_tracks import run_sample_tracks

__all__ = [
    "run_read_tracks", "run_build_pairs",
    "run_train_markov", "run_sample_tracks",
]
