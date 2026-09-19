# Training pipeline

Complete training and data-preparation code for the FAST-ML two-stream CNN,
as used in the JAMES submission. The inference-only release in the repository
root (`fastml/`, `scripts/`) consumes the checkpoint produced here.

Every command below has been executed end to end on the NERSC copy of the data
(Hurricane Beryl 2024 as the test case): best-track download → ERA5 training
data → two-stream training → GEFS vortex tracking → Markov track sampling →
per-member ERA5 prep → CNN chi/S prediction → FAST ODE ensemble.

## Layout

```
training/
├── IBtracs_datasets.py            Stage 0: download IBTrACS best tracks (NA/EP)
├── prepare_complete_training_data.py   Stage 1: 6h track -> 1h ERA5 -> *_dataset.pkl
├── stage1b_group_training_data.py Stage 1b: group pkls into training_data/{year}/
├── SciML_Fast_TwoStream.py        Stage 2: two-stream CNN training (Phases A/B/C/D)
├── SciML_Fast.py                  Model components, FAST physics engine (torch)
├── run_fast_physics_torch.py      Torch FAST ODE used in the training loss
├── run_fast_reference.py          FAST reference run + ERA5 ventilation baseline
├── eval_twostream_by_year.py      Per-year evaluation of the trained model
├── prep_bt_redo_2024_2025na.py    Utility: recompute 2024/2025NA best-track pkls
├── namelist.py, config.txt        Physics configuration
├── thermo/, util/, vortex_inversion/   Thermodynamics + vortex surgery support
├── precalc_data/                  Climatology fields (Cd, MLD, bathymetry, ...)
├── ckpt/twostream_final_d2.pth    Released weights (same file as ../ckpt)
├── track_model/                   Ensemble track generation
│   ├── read_files.py              ECMWF TIGGE XML -> 51-member tracks
│   ├── gefs_tracks.py             GEFS GRIB2 vortex tracking -> 31-member tracks
│   ├── run_gefs_tracks.py         One command: track -> pairs -> Markov -> sample
│   ├── build_pairs.py             6h velocity pairs
│   ├── train_markov.py            Per-step Gaussian Markov model
│   ├── sample_tracks.py           1000 synthetic members -> synthetic_tracks_*.nc
│   └── run_pipeline.py            TIGGE variant of the same four stages
├── gefs/
│   ├── download_gefs_storm.py     GEFS pgrb2a downloader (31 members x 0..240h)
│   └── gefs_storms_2024_2026.csv  Storm table used by the downloader
├── ensemble/                      Ensemble member preparation + forecast
│   ├── prepare_ensemble_storm.py  Best-track/synthetic track splicing helpers
│   ├── run_prep_div1000.py        Stage E3: per-member *_dataset.pkl (ERA5)
│   ├── predict_chi_s_div1000.py   Stage E4: CNN chi/S prediction -> chi_s.nc
│   └── run_ode_from_chis.py       Stage E5: per-member FAST ODE -> ode.nc
└── fast_ode/                      Vendored FAST ODE (Fast.py, env.py, track.py, ...)
```

## Prerequisites

* Python 3.10+ with `torch`, `numpy`, `pandas`, `scipy`, `xarray`, `einops`,
  `matplotlib`; plus `pyamg`, `pyshtools` (vortex surgery), `cfgrib` (GEFS
  GRIB2). Optional: `wandb` (experiment tracking; without credentials the
  trainer just prints a warning and continues).
* An ERA5 archive on disk, laid out as `{root}/{NA|EP}/{VAR}/` with hourly
  NetCDF files (e.g. `NA/T/828421.T.e5...2017010100_2017010123.nc`). The
  scripts default to the NERSC copy at `/global/cfs/cdirs/m5011/Jay/ERA5`;
  override with `--era5_root`, `$FASTML_ERA5_ROOT`, or `config.txt`.
* For the ensemble branch: GEFS GRIB2 input (`$GEFS_GRIB_ROOT`, downloadable
  via `gefs/download_gefs_storm.py`) and/or the ECMWF TIGGE XML archive
  (`$TIGGE_BASE_DIR`).

All `data/`, `training_data/`, `ensemble_*`, `runs/`, `ckpts/` outputs are
gitignored — only code and small static inputs are tracked.

## Stage 0 — IBTrACS best tracks

```bash
cd training
python IBtracs_datasets.py --config config.txt
```

`config.txt` controls scope (defaults: `basins=ALL`, `year_start=2003`,
`year_end=2025`, `output_dir=data`, `era5_root=...`). Output:
`data/{basin}/{year}/{SID}_{NAME}/track_intensity_6h.csv`. To scope down,
edit `config.txt` or copy it and pass `--config myconfig.txt`.

## Stage 1 — ERA5 training data (single-track preparation)

Converts each 6h best track to 1h and extracts ERA5 fields (72×72×0.25°
patches, 7 pressure levels, SST/MSLP, scalars, chi/S references) into a single
per-storm pkl:

```bash
python prepare_complete_training_data.py --data_root data \
    --era5_root $ERA5_ROOT --basins ALL --year_start 2003 --year_end 2024
# single storm / single year: --basins NA --year_start 2024 --year_end 2024
```

~30 min per storm on one core (skips storms whose pkl already exists; add
`--overwrite` to recompute). Output stays inside the storm dir:
`data/{basin}/{year}/{SID}_{NAME}/{SID}_{NAME}_dataset.pkl`
(~240 MB per storm, `spatial_3d` embedded).

## Stage 1b — group into the training layout

The trainer expects `training_data/{year}/{STORM}/{STORM}_dataset.pkl`, so link
the Stage-1 outputs in:

```bash
python stage1b_group_training_data.py --data_root data \
    --training_dir training_data --basins NA,EP --year_start 2003 --year_end 2025
```

Symlinks by default (`--copy` on filesystems without symlink support);
re-run any time after new Stage-1 storms appear (`--refresh` relinks newer
sources). Verified: the trainer loads these linked pkls identically to the
original PINN-era ones.

## Stage 2 — two-stream CNN training

Phases: A (S stream) → B (chi stream) → C (frozen physics check) →
D1 (head-only ODE fine-tune) → D2 (full end-to-end at small lr). Checkpoints
resume automatically from the latest phase file in the working directory.

```bash
# Full run (paper split: 2003-2022 train, 2023-2024 val)
python SciML_Fast_TwoStream.py --data_dir training_data \
    --train_years 2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014, \
2015,2016,2017,2018,2019,2020,2021,2022 --val_years 2023,2024 \
    --epochs_a 150 --epochs_b 150 --epochs_d1 200 --epochs_d2 100 --skip_final_eval

# Multi-GPU (A100 x4)
torchrun --nproc_per_node=4 SciML_Fast_TwoStream.py ...same args...

# ERA5 single-track training (smoke / debugging): one storm, one epoch
python SciML_Fast_TwoStream.py --data_dir training_data \
    --train_years 2024 --val_years '' --epochs_a 1 --epochs_b 0 \
    --epochs_d1 0 --epochs_d2 0 --skip_final_eval \
    --storm_include '2024/2024181N09320_BERYL'
```

Notes (verified behaviour):
* `--train_years/--val_years/--test_years` are comma-separated year lists;
  unset lists fall back to the defaults in `SciML_Fast.py`
  (train 2003-2022, val 2023-2024).
* `--storm_include 'year/STORM_DIR'` restricts training to given storms
  (comma-separate several); `--storm_exclude storm_exclude.txt` drops storms
  listed one per line.
* Filtering: peak ≥ 45 kt (`--min_vmax_kts`) and ≥ 120 h of track after the
  first 45-kt point (`--min_duration_h`); weak/short storms are counted and
  skipped automatically.
* Outputs land in the working directory: `ckpt_phase_{a,b,d,d1}.pth`,
  `twostream_final*.pth` (final models), `training_logs/` (phase CSV +
  monitor plots), `wandb/` if enabled. The release checkpoint is
  `ckpt/twostream_final_d2.pth`.

Evaluation:

```bash
python eval_twostream_by_year.py --ckpt ckpt/twostream_final_d2.pth \
    --data_dir training_data --out_dir runs/eval --split val --model_tag B_D2
```

### Full-scale run (verified)

The complete archive — 820 storms, 2003-2024, NA+EP, ~104 GB of pkls — was
loaded and trained through all phases (A/B/C/D1/D2) on one A100 node
(`full_run.sbatch`, a NERSC/Perlmutter submission template): peak host RAM
~85 GB during loading (the trainer lazy-loads spatial fields per batch, so
only the embedded single-pkl storms' fields stay resident), then 303 train /
28 val storms after the 45-kt + 120-h filters, ~5 min per 2-epoch phase with
`--bs 2 --accum 2`. Checkpoint save/load resumed across phases with
`miss=0 unexp=0`. For the converged paper model use the full epoch counts
(150/150/200/100) and, ideally, 4 GPUs via `torchrun`.

## Ensemble forecast (track sampling + inference)

Example: Hurricane Beryl 2024, GEFS initialisation 2024-06-28 12Z.

```bash
cd training

# E1 - GEFS GRIB2 download (31 members x 0..240h; skip if archive exists)
python gefs/download_gefs_storm.py --out_root $GEFS_GRIB_ROOT \
    --basin NA --year 2024 --name BERYL --members 31 --forecast_hours 0-240

# E2 - vortex tracking + Markov fit + sample 1000 synthetic tracks
cd track_model
GEFS_GRIB_ROOT=$GEFS_GRIB_ROOT python run_gefs_tracks.py --case 2024_BERYL_NA
cd ..
# -> ensemble_tracks/2024/beryl/synthetic_tracks_1000members.nc
# (TIGGE variant: python run_pipeline.py)

# E3 - per-member ERA5 extraction (2 min/member/core; PREP_* env config)
export PREP_SYNTH_NC=ensemble_tracks/2024/beryl/synthetic_tracks_1000members.nc
export PREP_BEST_TRACK=data/NA/2024/2024181N09320_BERYL/2024181N09320_BERYL_dataset.pkl
export PREP_HID=2024181N09320_BERYL
export PREP_OUT_ROOT=ensemble_runs/beryl_div1000
export PREP_YEAR=2024
export PREP_REF_TIME='2024-06-28 12:00:00'
export PREP_DURATION_H=120            # 5-day window; 264 for 11-day
export PREP_N_MEMBERS=1000
export PREP_WORKERS=32
export PREP_ERA5_ROOT=$ERA5_ROOT
python ensemble/run_prep_div1000.py

# E4 - CNN chi/S for every member -> chi_s.nc
python ensemble/predict_chi_s_div1000.py --ckpt ckpt/twostream_final_d2.pth \
    --ensemble_dir ensemble_runs/beryl_div1000 --year 2024 \
    --hurricane_id 2024181N09320_BERYL --train_data_dir training_data \
    --train_years 2024 --reference_time '2024-06-28 12:00:00' \
    --out_nc ensemble_runs/beryl_div1000/chi_s.nc

# E5 - per-member FAST ODE (fast + ml modes) -> ode.nc
python ensemble/run_ode_from_chis.py \
    --in_nc ensemble_runs/beryl_div1000/chi_s.nc \
    --out_nc ensemble_runs/beryl_div1000/ode.nc \
    --init_mode fhlo \
    --bt_pkl data/NA/2024/2024181N09320_BERYL/2024181N09320_BERYL_dataset.pkl
```

`ode.nc` carries `fast_vmax_kts` / `ml_vmax_kts` per member and is the format
consumed by the inference release (`scripts/plot_ensemble_vs_google.py`,
`scripts/plot_strike_probability.py`). Smoke result for the 2-member Beryl
test: FAST peak 145 kt, FAST-ML 126 kt, observed 145 kt.

Useful E5 flags: `--modes ml` (halve cost), `--vp_scale 1.1` (compensate PI
low bias), `--limit N` / `PREP_START/PREP_END` (subsets for testing),
`--no_forcing` (free-physics forecast). Without `--bt_pkl` the ODE falls back
from `fhlo` to `free` initialisation with a warning.

## Environment overrides

| Variable | Purpose | Default |
|---|---|---|
| `FASTML_ERA5_ROOT` | ERA5 archive root | NERSC CFS path |
| `TIGGE_BASE_DIR` | ECMWF TIGGE XML archive | NERSC CFS path |
| `GEFS_GRIB_ROOT` | GEFS GRIB2 archive | NERSC CFS path |
| `PREP_*` | E3 case configuration (see script header) | IRMA example |
| `WANDB_ENTITY/PROJECT/API_KEY` | Weights & Biases (never hardcode) | - |

## Notes

* `fast_ode/` is a vendored copy of the reference FAST implementation
  (`Fast.py`, `track.py`, `env.py` + static data) so the ensemble ODE stage is
  self-contained; it is imported with the working directory set to `fast_ode/`.
* The vortex surgery (`vortex_inversion/`) requires `pyamg` and `pyshtools`.
* The checkpoint in `ckpt/` is identical to the one released in the inference
  repo root (`ckpt/`).
