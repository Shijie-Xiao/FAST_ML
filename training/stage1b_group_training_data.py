#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1b: group per-storm pkls into the training layout.

prepare_complete_training_data.py (Stage 1) writes

    data/{basin}/{year}/{SID}_{NAME}/{SID}_{NAME}_dataset.pkl

while the trainer / evaluator / FAST reference all read

    training_data/{year}/{STORM}/{STORM}_dataset.pkl

This script links (symlink when possible, copy with --copy) every finished
Stage-1 storm into the training layout. Re-running it is safe: existing
targets are refreshed only when the source is newer (--refresh) or always
(--force).

Usage:
    python stage1b_group_training_data.py [--data_root data]
        [--training_dir training_data] [--basins NA,EP]
        [--year_start 2003] [--year_end 2025]
        [--copy] [--refresh] [--force] [--dry_run]
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_root', default='data')
    p.add_argument('--training_dir', default='training_data')
    p.add_argument('--basins', default='NA,EP')
    p.add_argument('--year_start', type=int, default=2003)
    p.add_argument('--year_end', type=int, default=2025)
    p.add_argument('--copy', action='store_true',
                   help='Copy instead of symlink (needed on filesystems without symlink support)')
    p.add_argument('--refresh', action='store_true',
                   help='Relink targets whose source pkl is newer')
    p.add_argument('--force', action='store_true',
                   help='Relink all targets even if they exist')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    data_root = Path(args.data_root)
    training_dir = Path(args.training_dir)
    if not data_root.is_dir():
        p.error(f'data_root not found: {data_root} (run Stage 0/1 first)')

    basins = [b.strip().upper() for b in args.basins.split(',') if b.strip()]
    linked, skipped, missing = 0, 0, 0
    for basin in basins:
        for year in range(args.year_start, args.year_end + 1):
            ydir = data_root / basin / str(year)
            if not ydir.is_dir():
                continue
            for sdir in sorted(ydir.iterdir()):
                if not sdir.is_dir():
                    continue
                src = sdir / f'{sdir.name}_dataset.pkl'
                if not src.exists():
                    missing += 1
                    continue
                dst_dir = training_dir / str(year) / sdir.name
                dst = dst_dir / src.name
                if dst.exists() or dst.is_symlink():
                    if args.force or (args.refresh and dst.is_symlink()
                                      and dst.resolve() != src.resolve()):
                        pass  # fall through to (re)link
                    else:
                        skipped += 1
                        continue
                if args.dry_run:
                    print(f'[dry-run] {src} -> {dst}')
                else:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    if dst.is_symlink() or dst.exists():
                        dst.unlink() if dst.is_symlink() else dst.unlink()
                    if args.copy:
                        shutil.copy2(src, dst)
                    else:
                        try:
                            dst.symlink_to(src)
                        except OSError:
                            shutil.copy2(src, dst)
                    # bring the 6h track csv along: several tools read it next to the pkl
                    csv6 = sdir / 'track_intensity_6h.csv'
                    if csv6.exists():
                        dcsv = dst_dir / csv6.name
                        if not (dcsv.exists() or dcsv.is_symlink()):
                            if args.copy:
                                shutil.copy2(csv6, dcsv)
                            else:
                                try:
                                    dcsv.symlink_to(csv6)
                                except OSError:
                                    shutil.copy2(csv6, dcsv)
                linked += 1
    print(f'Done: linked={linked} skipped(existing)={skipped} '
          f'missing(Stage-1 pkl)={missing}')


if __name__ == '__main__':
    main()
