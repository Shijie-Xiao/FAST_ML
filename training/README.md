# Training pipeline

Complete training and data-preparation code for the FAST-ML two-stream CNN, as
used in the JAMES submission. The inference-only release in the repository root
(`fastml/`, `scripts/`) consumes the checkpoint produced here.

Layout:

```
training/
├── IBtracs_datasets.py            Stage 0: download IBTrACS best tracks (NA/EP)
├── prepare_complete_training_data.py   Stage 1: 6h track -> 1h ERA5 -> *_dataset.pkl
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
│   ├── build_pairs.py             6h velocity pairs
│   ├── train_markov.py            Per-step Gaussian Markov model
│   ├── sample_tracks.py           1000 synthetic members -> synthetic_tracks_*.nc
│   └── run_pipeline.py            All four stages in one command
├── gefs/
│   ├── download_gefs_storm.py     GEFS pgrb2a downloader (31 members x 0..240h)
│   └── gefs_storms_2024_2026.csv  Storm table used by the downloader
├── ensemble/                      Ensemble member preparation + forecast
│   ├── prepare_ensemble_storm.py  Best-track/synthetic track splicing helpers
│   ├── run_prep_div1000.py        Stage 3: per-member *_dataset.pkl (ERA5 extraction)
│   ├── predict_chi_s_div1000.py   Stage 4: CNN chi/S prediction -> chi_s.nc
│   └── run_ode_from_chis.py       Stage 5: per-member FAST ODE -> ode.nc
└── fast_ode/                      Vendored FAST ODE (Reproduce: Fast.py, env.py, ...)
```

## Prerequisites

* Python 3.10+ with `torch`, `numpy`, `pandas`, `scipy`, `xarray`, `einops`,
  `matplotlib`; plus `pyamg`, `pyshtools` (vortex surgery), `cfgrib` (GEFS
  GRIB2). Optional: `wandb` (experiment tracking).
* An ERA5 archive on disk. The scripts default to the NERSC copy at
  `/global/cfs/cdirs/m5011/Jay/ERA5` and accept an override everywhere:
  `--era5_root`, `$FASTML_ERA5_ROOT`, or `config.txt`.
* External ensemble inputs: ECMWF TIGGE XML archive (`$TIGGE_BASE_DIR`) and/or
  GEFS GRIB2 archive (`$GEFS_GRIB_ROOT`). `gefs/download_gefs_storm.py` can
  download the latter from NOAA directly.

## End-to-end workflow

```bash
cd training

# Stage 0 - IBTrACS best tracks (NA+EP, 2003-2025) -> data/{basin}/{year}/{sid}_{name}/
python IBtracs_datasets.py --config config.txt

# Stage 1 - training data: hourly ERA5 extraction -> {storm}_dataset.pkl
# (~30 min per storm-year on one core; parallelise by year)
python prepare_complete_training_data.py --data_root data --era5_root $ERA5_ROOT \
    --basins ALL --year_start 2003 --year_end 2024

# Stage 2 - train the two-stream CNN (A: S stream, B: chi stream, C: frozen
# physics check, D: end-to-end fine-tune). Single GPU:
python SciML_Fast_TwoStream.py --data_dir training_data \
    --train_years 2003-2022 --val_years 2023 --epochs_a 150 --epochs_b 150 ...
# Multi GPU: torchrun --nproc_per_node=4 SciML_Fast_TwoStream.py ...

# Evaluation
python eval_twostream_by_year.py --ckpt ckpt/twostream_final_d2.pth \
    --data_dir training_data --out_dir runs/eval --test_years 2024
```

The released checkpoint reproduces the paper metrics from
`training_data/2024` directly (see repository root README).

## Ensemble forecast preparation (a case study)

```bash
# E1 - GEFS tracks (or TIGGE): download + vortex tracking
python gefs/download_gefs_storm.py --out_root $GEFS_GRIB_ROOT \
    --basin NA --year 2024 --name BERYL --members 31 --forecast_hours 0-240

# E2 - synthetic members: Markov fit + sampling -> ensemble_tracks/<storm>/
cd track_model && python run_gefs_tracks.py   # or run_pipeline.py for TIGGE
cd ..

# E3 - per-member training data (ERA5 along each synthetic track)
export PREP_SYNTH_NC=ensemble_tracks/<storm>/synthetic_tracks_1000members.nc
export PREP_BEST_TRACK=training_data/<year>/<sid>_<name>/<sid>_<name>_dataset.pkl
export PREP_OUT_ROOT=ensemble_runs/<storm>_div1000
export PREP_ERA5_ROOT=$ERA5_ROOT
python ensemble/run_prep_div1000.py            # PREP_* env vars configure the case

# E4 - CNN chi/S for every member -> chi_s.nc
python ensemble/predict_chi_s_div1000.py --ckpt ckpt/twostream_final_d2.pth \
    --ensemble_dir ensemble_runs/<storm>_div1000 --year <year> \
    --hurricane_id <sid>_<name> --train_data_dir training_data \
    --out_nc ensemble_runs/<storm>_div1000/chi_s.nc

# E5 - per-member FAST ODE -> ode.nc (fast_vmax_kts / ml_vmax_kts)
python ensemble/run_ode_from_chis.py \
    --in_nc ensemble_runs/<storm>_div1000/chi_s.nc \
    --out_nc ensemble_runs/<storm>_div1000/ode.nc
```

The `ode.nc` output is the format consumed by the inference release
(`scripts/plot_ensemble_vs_google.py`, `scripts/plot_strike_probability.py`).

## Environment overrides

| Variable | Purpose | Default |
|---|---|---|
| `FASTML_ERA5_ROOT` | ERA5 archive root | NERSC CFS path |
| `TIGGE_BASE_DIR` | ECMWF TIGGE XML archive | NERSC CFS path |
| `GEFS_GRIB_ROOT` | GEFS GRIB2 archive | NERSC CFS path |
| `TC_RISK_PATH` | optional tropical_cyclone_risk checkout (Cd field) | sibling dir |
| `WANDB_*` | Weights & Biases credentials (never hardcode) | - |

## Notes

* `fast_ode/` is a vendored copy of the reference FAST implementation
  (`Fast.py`, `track.py`, `env.py` + static data) so the ensemble ODE stage is
  self-contained; it is imported with the working directory set to `fast_ode/`.
* The vortex surgery (`vortex_inversion/`) requires `pyamg` and `pyshtools`.
* The checkpoint in `ckpt/` is identical to the one released in the inference
  repo root (`ckpt/`).
