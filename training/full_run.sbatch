#!/bin/bash
#SBATCH -A m5011
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -J fml-full
#SBATCH -o /pscratch/sd/s/sixao74/Deepmind/FAST_ML/training/full_run_%j.log
#SBATCH -e /pscratch/sd/s/sixao74/Deepmind/FAST_ML/training/full_run_%j.log

# Full-data verification run for the migrated training pipeline.
# Uses the complete 820-storm archive (PINN/training_data, 2003-2024, NA+EP)
# via year-dir symlinks, then runs the two-stream trainer end to end over the
# paper split with short epochs (verification, not convergence).
set -euo pipefail

TRAIN=/pscratch/sd/s/sixao74/Deepmind/FAST_ML/training
PY=/pscratch/sd/s/sixao74/envs/fast_ml/bin/python
SRC=/pscratch/sd/s/sixao74/Deepmind/PINN/training_data

cd "$TRAIN"

# 1) link the complete archive year dirs into training/training_data
mkdir -p training_data
for y in $(ls "$SRC" | grep -E '^[0-9]+$'); do
    if [ ! -e "training_data/$y" ]; then
        ln -s "$SRC/$y" "training_data/$y"
        echo "linked $y"
    fi
done

echo "=== training_data inventory ==="
for y in $(ls training_data | grep -E '^[0-9]+$' | sort); do
    echo "$y: $(ls training_data/$y | wc -l)"
done

# 2) full-data trainer run: paper split, short epochs (verification, not convergence)
"$PY" SciML_Fast_TwoStream.py \
    --data_dir training_data \
    --train_years 2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \
    --val_years 2023,2024 \
    --epochs_a 2 --epochs_b 2 --epochs_d1 2 --epochs_d2 1 \
    --bs 2 --accum 2 \
    --skip_final_eval

echo "=== full-run smoke finished ==="
