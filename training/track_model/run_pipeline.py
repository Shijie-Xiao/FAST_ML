#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the full ensemble track-generation pipeline for all 2023-2025 NA storms.

Stages:
  1. read_files   -- parse ECMWF TIGGE, pick best init cycle, save 51-member raw
  2. build_pairs  -- 6h velocity pairs (75% coverage)
  3. train_markov -- per-step Gaussian Markov model
  4. sample_tracks-- sample 1000 synthetic members -> PINN/ensemble_tracks/{storm}/

Output synthetic tracks: config.SYNTH_TRACKS_DIR/{storm_lower}/synthetic_tracks_1000members.nc
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import read_files
import build_pairs
import train_markov
import sample_tracks


def main():
    print(f"=== Ensemble track pipeline: {len(config.ALL_STORMS)} NA storms "
          f"({config.YEARS}) ===", flush=True)
    print("[1/4] read TIGGE ensembles ...", flush=True)
    got = read_files.run_read_tracks()
    print("[2/4] build velocity pairs ...", flush=True)
    build_pairs.run_build_pairs()
    print("[3/4] train Markov model ...", flush=True)
    train_markov.run_train_markov()
    print("[4/4] sample synthetic tracks ...", flush=True)
    sample_tracks.run_sample_tracks()

    # Summary of produced NCs
    ncs = sorted(config.SYNTH_TRACKS_DIR.glob("*/synthetic_tracks_*members.nc"))
    print(f"\n=== DONE: {len(ncs)} synthetic-track NCs written to "
          f"{config.SYNTH_TRACKS_DIR} ===", flush=True)
    for nc in ncs:
        print(f"  {nc.parent.name}: {nc.name}", flush=True)


if __name__ == "__main__":
    main()
