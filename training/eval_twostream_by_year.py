#!/usr/bin/env python3
"""
评估 TwoStreamFASTModel 在全部风暴上的表现，严格对齐 run_fast_reference.process_one_pkl。

对每个风暴：
  1. ML 模型推理（use_run_fast_physics=False）→ 得到 pred_chi_eff 和 pred_s
  2. ML 路径: xs_ml = max(pred_chi_eff * pred_s, XS_NAN_FALLBACK)
            → run_fast_with_init 物理积分 → ml_v_max
  3. FAST 路径: xs_ref = max(chi_calibrated(chi_ref) * s_ref, XS_NAN_FALLBACK)
              → run_fast_with_init 物理积分 → fast_v_max
  4. MSE/RMSE：全段 mask (v_obz > 0)，与 fast_reference.csv 完全一致
  5. 输出 storm_metrics.csv、summary_metrics.csv 和单风暴对比图

用法:
  python eval_twostream_by_year.py --ckpt twostream_final_d2.pth --out_dir eval_results
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from SciML_Fast_TwoStream import (
    TwoStreamFASTModel,
    load_1km_storms,
    compute_spatial_stats,
    create_batch,
    load_ckpt,
    _mfilt,
)
from SciML_Fast import MS_TO_KNOTS
from run_fast_reference import run_fast_with_init, _chi_calibrated_multiply, _median_filter_1d, XS_NAN_FALLBACK

# ── 消融模型支持：默认 None 时退化为原 TwoStreamFASTModel ──────────────────────
# Optional ablation variant; lives in an external ODE/ checkout if needed.
try:
    _ODE_PATH = os.environ.get(
        'ABLATION_PATH', str(Path(__file__).resolve().parent.parent.parent / 'ODE'))
    sys.path.insert(0, _ODE_PATH)
    import SciML_Fast_TwoStream_Ablation as _abl
    _ABL_AVAILABLE = True
except Exception:
    _ABL_AVAILABLE = False


def _build_model(arch, pool, dev):
    """根据消融轴构建模型；arch='dual'/pool='full'（默认）即原 TwoStreamFASTModel。"""
    if arch == 'dual' and pool == 'full':
        return TwoStreamFASTModel().to(dev)
    if not _ABL_AVAILABLE:
        raise RuntimeError(
            f'消融模型 {arch}/{pool} 需要 SciML_Fast_TwoStream_Ablation.py, 但无法导入')
    return _abl.build_model(arch, pool).to(dev)


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _short_name(hurricane):
    return str(hurricane).split('_')[-1]


def _infer_year(ds):
    """从 hurricane 字符串或 ds['year'] 中提取年份。优先用 ds['year']。"""
    yr = ds.get('year')
    if yr and isinstance(yr, int) and yr > 1900:
        return yr
    h = str(ds.get('hurricane', ''))
    # 匹配 AL/EP 后跟 2 位序号 + 4 位年份 的模式，如 AL782024 -> 2024
    m = re.search(r'(?:AL|EP)\d*(\d{4})_', h)
    if m:
        return int(m.group(1))
    # fallback：取最后一个 4 位数字
    nums = re.findall(r'(\d{4})', h)
    if nums:
        return int(nums[-1])
    return 0


def _eval_one(model, ds, stats, dev, vent_scale=1.0):
    """
    评估单个风暴，严格对齐 run_fast_reference.process_one_pkl 的逻辑。
    
    步骤：
    1. ML 模型推理（use_run_fast_physics=False）得到 pred_chi_eff 和 pred_s
    2. ML 路径：xs_ml = max(pred_chi_eff * pred_s, XS_NAN_FALLBACK)
              → run_fast_with_init(scalars, xs_ml, ..., pred_s) → ml_v_fast, ml_v_max, ml_m
    3. FAST 路径：xs_ref = max(chi_calibrated(chi_ref) * s_ref, XS_NAN_FALLBACK)
                → run_fast_with_init(scalars, xs_ref, ..., s_ref) → fast_v_fast, fast_v_max, fast_m
    4. MSE/RMSE: 全段 mask (v_obz > 0)，单位 (m/s)² / kts，与 fast_reference.csv 完全一致
    """
    model.eval()
    with torch.no_grad():
        b = create_batch([ds], 1, dev, shuffle=False, spatial_stats=stats)
        sc = _mfilt(b['scalars'].clone(), dev)

        # ── ML 推理：use_run_fast_physics=False，仅获取 chi/S ─────────────────
        out = model(
            b['spatial_3d'], b['spatial_2d'], sc, b['v_gt'][:, 0:1, :],
            precomp=None,
            env_wnds=b.get('env_wnds'),
            utran=b.get('utran'), vtran=b.get('vtran'), lats=b.get('lats'),
            use_run_fast_physics=False,
        )
        ml_s = out['s'][0, :, 0].cpu().numpy().astype(np.float64)           # [T]
        ml_chi_eff = out['chi'][0, :, 0].cpu().numpy().astype(np.float64)   # [T] 已校准 0-4

        T = b['v_gt'].shape[1]

        # ── 准备 run_fast_with_init 输入（与 process_one_pkl 完全一致）──────────
        # 注意：必须用原始 scalars（vp 中位数滤波），不要再加其他变换
        scalars_np = b['scalars'].cpu().numpy().astype(np.float64)
        # vp 中位数滤波，与 process_one_pkl L483-485 一致
        scalars_np[0, :, 3] = _median_filter_1d(scalars_np[0, :, 3], size=3)

        v_gt_np   = b['v_gt'].cpu().numpy()
        env_wnds_np = b['env_wnds'].cpu().numpy() if b.get('env_wnds') is not None else None
        utran_np  = b['utran'].cpu().numpy()  if b.get('utran')  is not None else None
        vtran_np  = b['vtran'].cpu().numpy()  if b.get('vtran')  is not None else None
        lats_np   = b['lats'].cpu().numpy()   if b.get('lats')   is not None else None
        lons_np   = b['lons'].cpu().numpy()   if b.get('lons')   is not None else None

        # ── ML 路径：xs_ml = max(pred_chi_eff * pred_s, XS_NAN_FALLBACK) ─────
        # pred_chi_eff 已经是 0-4 校准范围，不需要再调用 _chi_calibrated_multiply
        # vent_scale: 对 ventilation (chi*s) 整体乘系数（<1 -> 更弱 vent -> 更强风暴）
        xs_ml = np.maximum(
            np.nan_to_num(ml_chi_eff * ml_s * vent_scale, nan=XS_NAN_FALLBACK),
            XS_NAN_FALLBACK,
        ).reshape(1, T, 1)
        s_ml_2d = ml_s.reshape(1, T)  # run_fast_with_init 的 s_ref 参数

        ml_v_fast_ms, ml_v_max_ms, ml_m_series = run_fast_with_init(
            scalars_np, xs_ml, v_gt_np,
            env_wnds_np, utran_np, vtran_np, lats_np, s_ml_2d,
            lons=lons_np,
        )

        # ── FAST 路径：与 process_one_pkl 完全一致 ─────────────────────────────
        chi_ref_np = b['chi_ref'].cpu().numpy().astype(np.float64)
        s_ref_np   = b['s_ref'].cpu().numpy().astype(np.float64)
        chi_cal_ref = _chi_calibrated_multiply(chi_ref_np)
        xs_ref = np.maximum(
            np.nan_to_num(chi_cal_ref * s_ref_np, nan=XS_NAN_FALLBACK),
            XS_NAN_FALLBACK,
        )
        s_ref_2d = s_ref_np[0, :, 0].reshape(1, T)

        fast_v_fast_ms, fast_v_max_ms, fast_m_series = run_fast_with_init(
            scalars_np, xs_ref, v_gt_np,
            env_wnds_np, utran_np, vtran_np, lats_np, s_ref_2d,
            lons=lons_np,
        )

        # 标准化形状为 [T]
        ml_vmax = np.asarray(ml_v_max_ms).reshape(-1)[:T]
        fast_vmax = np.asarray(fast_v_max_ms).reshape(-1)[:T]
        ml_m = np.asarray(ml_m_series).reshape(-1)[:T]
        fast_m = np.asarray(fast_m_series).reshape(-1)[:T]

        # ── 观测 ─────────────────────────────────────────────────────────────
        v_obz = b['v_gt'][0, :, 0].cpu().numpy().astype(np.float64)  # m/s
        vp = b['scalars'][0, :, 3].cpu().numpy().astype(np.float64)
        t_start = int(ds.get('t_start', 0))
        seq_len = int(b['seq_len'][0].item()) if 'seq_len' in b else T

        # 位置坐标（用于 RI 分析）：lats/lons 形状一般为 [1, T]
        def _flatten_geo(arr):
            if arr is None:
                return np.full(T, np.nan)
            a = np.asarray(arr).astype(np.float64).reshape(-1)[:T]
            if a.size < T:
                pad = np.full(T - a.size, np.nan)
                a = np.concatenate([a, pad])
            return a

        lats_seq = _flatten_geo(lats_np)
        lons_seq = _flatten_geo(lons_np)

        # ── 全段 mask（v_obz > 0），与 process_one_pkl 一致 ────────────────────
        # 注意：mask 必须包括 [0, seq_len)，且 v_obz 有效，不要求 t >= t_start！
        base_mask = (np.arange(T) < seq_len) & np.isfinite(v_obz) & (v_obz > 0)
        ml_valid = base_mask & np.isfinite(ml_vmax) & (ml_vmax > 0)
        fast_valid = base_mask & np.isfinite(fast_vmax) & (fast_vmax > 0)

        def _mse_ms2_rmse_kts(pred_ms, obs_ms, mask):
            if mask.sum() == 0:
                return np.nan, np.nan
            d = pred_ms[mask] - obs_ms[mask]
            mse = float(np.mean(d ** 2))
            rmse = float(np.sqrt(mse)) * MS_TO_KNOTS
            return mse, rmse

        ml_mse, ml_rmse = _mse_ms2_rmse_kts(ml_vmax, v_obz, ml_valid)
        fast_mse, fast_rmse = _mse_ms2_rmse_kts(fast_vmax, v_obz, fast_valid)

        # ── ML curve align prestart obs：前 t_start 步替换为观测，再算 MSE ────────
        # 对应旧 eval 的 _ml_curve_align_prestart_obs；FAST 不做此对齐。
        ml_vmax_aligned = ml_vmax.copy()
        if t_start > 0:
            ml_vmax_aligned[:t_start] = v_obz[:t_start]
        ml_aligned_mse, ml_aligned_rmse = _mse_ms2_rmse_kts(ml_vmax_aligned, v_obz, ml_valid)

        gain_mse = fast_mse - ml_mse if not (np.isnan(ml_mse) or np.isnan(fast_mse)) else np.nan
        gain_rmse = fast_rmse - ml_rmse if not (np.isnan(ml_rmse) or np.isnan(fast_rmse)) else np.nan
        gain_mse_aligned = (
            fast_mse - ml_aligned_mse
            if not (np.isnan(ml_aligned_mse) or np.isnan(fast_mse)) else np.nan
        )
        gain_rmse_aligned = (
            fast_rmse - ml_aligned_rmse
            if not (np.isnan(ml_aligned_rmse) or np.isnan(fast_rmse)) else np.nan
        )
        beat_fast = 1 if (not np.isnan(gain_mse) and gain_mse > 0) else 0

        return {
            'ml_vmax_kts': ml_vmax * MS_TO_KNOTS,
            'fast_vmax_kts': fast_vmax * MS_TO_KNOTS,
            'fast_v_kts': np.asarray(fast_v_fast_ms).reshape(-1)[:T] * MS_TO_KNOTS,
            'ml_v_kts': np.asarray(ml_v_fast_ms).reshape(-1)[:T] * MS_TO_KNOTS,
            'v_obz_kts': v_obz * MS_TO_KNOTS,
            'vp_kts': vp * MS_TO_KNOTS,
            'ml_s': ml_s,
            'ml_chi': ml_chi_eff,
            'fast_s': s_ref_np[0, :, 0].astype(np.float64),
            'fast_chi': chi_cal_ref.reshape(-1)[:T].astype(np.float64),
            'ml_m': ml_m,
            'fast_m': fast_m,
            'lats': lats_seq,
            'lons': lons_seq,
            'seq_len': seq_len,
            't_start': t_start,
            'times': ds.get('times'),
            'T': T,
            'ml_mse': ml_mse,
            'ml_rmse_kts': ml_rmse,
            'ml_aligned_mse': ml_aligned_mse,
            'ml_aligned_rmse_kts': ml_aligned_rmse,
            'fast_mse': fast_mse,
            'fast_rmse_kts': fast_rmse,
            'gain_vs_fast_mse': gain_mse,
            'gain_vs_fast_rmse_kts': gain_rmse,
            'gain_vs_fast_mse_aligned': gain_mse_aligned,
            'gain_vs_fast_rmse_kts_aligned': gain_rmse_aligned,
            'beat_fast': beat_fast,
        }


def _predict_chi_s_one(model, ds, stats, dev):
    """轻量版：只跑 TwoStream 前向得到 chi/s，不做任何 ODE 积分。

    返回 FAST-ML 的 (ml_chi, ml_s) 与 FAST 的 (fast_chi=校准后 chi_ref, fast_s=s_ref)，
    以及 lat/lon/v_obz/vp，供后续用 Reproduce 的参数化 ODE 单独跑。
    与 _eval_one 的前向部分逐行对齐，保证 chi/s 完全一致。
    """
    model.eval()
    with torch.no_grad():
        b = create_batch([ds], 1, dev, shuffle=False, spatial_stats=stats)
        sc = _mfilt(b['scalars'].clone(), dev)
        out = model(
            b['spatial_3d'], b['spatial_2d'], sc, b['v_gt'][:, 0:1, :],
            precomp=None,
            env_wnds=b.get('env_wnds'),
            utran=b.get('utran'), vtran=b.get('vtran'), lats=b.get('lats'),
            use_run_fast_physics=False,
        )
        T = b['v_gt'].shape[1]
        ml_s = out['s'][0, :, 0].cpu().numpy().astype(np.float64)
        ml_chi = out['chi'][0, :, 0].cpu().numpy().astype(np.float64)

        chi_ref_np = b['chi_ref'].cpu().numpy().astype(np.float64)
        s_ref_np = b['s_ref'].cpu().numpy().astype(np.float64)
        fast_chi = _chi_calibrated_multiply(chi_ref_np).reshape(-1)[:T].astype(np.float64)
        fast_s = s_ref_np[0, :, 0].astype(np.float64)

        v_obz = b['v_gt'][0, :, 0].cpu().numpy().astype(np.float64)   # m/s
        vp = b['scalars'][0, :, 3].cpu().numpy().astype(np.float64)

        def _geo(arr):
            if arr is None:
                return np.full(T, np.nan)
            a = np.asarray(arr.cpu().numpy() if hasattr(arr, 'cpu') else arr,
                           dtype=np.float64).reshape(-1)[:T]
            if a.size < T:
                a = np.concatenate([a, np.full(T - a.size, np.nan)])
            return a

        # ── ODE 输入（裁剪后，T 与 chi/s 一致），供后续参数化 ODE 单独跑 ──
        scalars_arr = b['scalars'][0, :, :].cpu().numpy().astype(np.float64)   # [T,4] alpha,beta,gamma,vp
        ew = b.get('env_wnds')
        env_wnds_arr = (ew[0, :, :].cpu().numpy().astype(np.float64)
                        if ew is not None else np.full((T, 4), np.nan))        # [T,4]

        seq_valid = int(b['seq_len'][0].item()) if 'seq_len' in b else T
        t0 = int(ds.get('_t0', 0))

        return {
            'T': T,
            'seq_len': seq_valid, 't0': t0,
            'ml_chi': ml_chi, 'ml_s': ml_s,
            'fast_chi': fast_chi, 'fast_s': fast_s,
            'v_obz_kts': v_obz * MS_TO_KNOTS,
            'v_gt_ms': v_obz,
            'vp_kts': vp * MS_TO_KNOTS,
            'lats': _geo(b.get('lats')), 'lons': _geo(b.get('lons')),
            'scalars': scalars_arr, 'env_wnds': env_wnds_arr,
            'utran': _geo(b.get('utran')), 'vtran': _geo(b.get('vtran')),
        }


def _plot_yearly_rmse_bar(all_rows, train_years, val_years, test_years,
                          model_tag, out_path):
    """按年份绘制 FAST vs FAST_ML 的 Per-storm Mean RMSE 对比柱状图（论文风格）。

    - 绿色柱: FAST    (#5ec64a)，使用 fast_rmse_kts
    - 蓝色柱: FAST_ML (#3a86ff)，使用 ml_aligned_rmse_kts
    - 柱顶标注每年 gain = fast - ml (kts)
    - 颜色与单风暴对比图保持一致
    """
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=['ml_aligned_rmse_kts', 'fast_rmse_kts'])
    if df.empty:
        return

    grp = (df.groupby('year')
             .agg(ml_rmse=('ml_aligned_rmse_kts', 'mean'),
                  fast_rmse=('fast_rmse_kts', 'mean'),
                  n=('hurricane', 'count'))
             .sort_index())

    years = grp.index.tolist()
    x = np.arange(len(years))
    width = 0.4

    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor='white')
    ax.set_facecolor('white')

    # ── 按 split 给 val/test 年份区间加低饱和度背景色块（train 不着色）──────────
    def _split_of(yr):
        if yr in set(test_years):
            return 'test'
        if yr in set(val_years):
            return 'val'
        return None

    split_idx = {'val': [], 'test': []}
    for i, yr in enumerate(years):
        sp = _split_of(int(yr))
        if sp is not None:
            split_idx[sp].append(i)

    ax.bar(x - width / 2, grp['fast_rmse'], width,
           label='FAST', color='#5ec64a',
           edgecolor='black', linewidth=0.6, zorder=2)
    ax.bar(x + width / 2, grp['ml_rmse'], width,
           label='FAST_ML', color='#3a86ff',
           edgecolor='black', linewidth=0.6, zorder=2)

    for i, yr in enumerate(years):
        gain = grp.loc[yr, 'fast_rmse'] - grp.loc[yr, 'ml_rmse']
        ymax = max(grp.loc[yr, 'fast_rmse'], grp.loc[yr, 'ml_rmse'])
        ax.text(i, ymax + 0.5, f'{gain:+.1f}',
                ha='center', va='bottom',
                color='#222222', fontsize=16, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in years],
                       rotation=45, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('Year', fontsize=22)
    ax.set_ylabel('Per-storm Mean RMSE (knots)', fontsize=22)

    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

    ax.legend(loc='upper left', fontsize=22, framealpha=0.95,
              edgecolor='black')
    ax.grid(True, axis='y', alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(years) - 0.4)
    ymax_all = max(grp['fast_rmse'].max(), grp['ml_rmse'].max())
    ax.set_ylim(0, ymax_all * 1.18)

    # ── val(22-23) / test(24): 用虚线分隔 + 顶部 Val/Test 标注（不进图例）────────
    y_top = ymax_all * 1.18
    boundaries = set()
    for sp, idxs in split_idx.items():
        if not idxs:
            continue
        boundaries.add(min(idxs) - 0.5)            # 区段左边界
        ax.text((min(idxs) + max(idxs)) / 2, y_top * 0.99,
                'Val' if sp == 'val' else 'Test',
                ha='center', va='top', fontsize=21, fontweight='bold',
                color='black', zorder=5)
    for xb in sorted(boundaries):
        ax.axvline(xb, color='#555555', linestyle='--', linewidth=1.8,
                   alpha=0.8, zorder=3)

    out_path = Path(out_path).with_suffix('.svg')
    fig.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_storm_rmse_one_year(df_year, year, split_name, out_path):
    """单个年份的 per-storm RMSE 图：FAST vs FAST_ML 双柱, 柱顶标注 gain=fast-ml,
    无背景色。"""
    df = df_year.dropna(subset=['ml_aligned_rmse_kts', 'fast_rmse_kts']).copy()
    if df.empty:
        return None
    df['sname'] = df['hurricane'].map(_short_name)
    df = df.sort_values('sname').reset_index(drop=True)

    n = len(df)
    x = np.arange(n)
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(9, n * 0.62), 6.8), facecolor='white')
    ax.set_facecolor('white')

    ax.bar(x - width / 2, df['fast_rmse_kts'], width, label='FAST',
           color='#5ec64a', edgecolor='black', linewidth=0.5, zorder=2)
    ax.bar(x + width / 2, df['ml_aligned_rmse_kts'], width, label='FAST_ML',
           color='#3a86ff', edgecolor='black', linewidth=0.5, zorder=2)

    ymax_all = max(df['fast_rmse_kts'].max(), df['ml_aligned_rmse_kts'].max())
    y_top = ymax_all * 1.20
    for i in range(n):
        gain = df['fast_rmse_kts'].iloc[i] - df['ml_aligned_rmse_kts'].iloc[i]
        ytop = max(df['fast_rmse_kts'].iloc[i], df['ml_aligned_rmse_kts'].iloc[i])
        ax.text(i, ytop + y_top * 0.012, f'{gain:+.1f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                color='#1a7f37' if gain >= 0 else '#c0392b')

    ax.set_xticks(x)
    ax.set_xticklabels(df['sname'], rotation=90, fontsize=14)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel('Per-storm RMSE (knots)', fontsize=22)
    ax.set_title(f'{year}  ({split_name})', fontsize=24, fontweight='bold')
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, y_top)
    ax.grid(True, axis='y', alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.legend(loc='upper left', fontsize=20, framealpha=0.95, edgecolor='black')

    out_path = Path(out_path).with_suffix('.svg')
    fig.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def _plot_storm_rmse(all_rows, val_years, test_years, model_tag, out_dir):
    """为 val/test 的每个年份(默认 2022,2023=Val / 2024=Test) 各画一张 per-storm RMSE 图。"""
    df = pd.DataFrame(all_rows)
    saved = []
    for yr in sorted(set(val_years) | set(test_years)):
        dy = df[df['year'] == yr]
        if dy.empty:
            continue
        split_name = 'Test' if yr in set(test_years) else 'Val'
        op = Path(out_dir) / f'storm_rmse_{yr}_{model_tag}.svg'
        r = _plot_storm_rmse_one_year(dy, yr, split_name, op)
        if r is not None:
            saved.append(r)
    return saved


def _last_strong_idx(series, i0, min_intensity_kts, peak_drop_frac):
    """series 从 i0 起最后一次 > max(min_intensity, peak*peak_drop_frac) 的索引。"""
    v = np.asarray(series, dtype=float)
    valid = np.isfinite(v)
    masked = np.where(valid, v, -np.inf)
    peak = float(masked[i0:].max()) if i0 < v.size else -np.inf
    if not np.isfinite(peak):
        return None
    thr = max(float(min_intensity_kts), peak * float(peak_drop_frac))
    above = valid & (v > thr)
    if not above.any():
        return None
    return int(np.where(above)[0][-1])


def _find_plot_window(v_obz_kts, min_intensity_kts=30.0,
                      peak_drop_frac=0.4, tail_buffer_hours=12,
                      model_kts=None):
    """自动选取单风暴绘图时间窗 [i0, i1)，截到“快速下降之后”。

    规则：
      - i0: obs 首次 > min_intensity_kts 的索引
      - i1: 取 obs 与模型(ml/fast 包络) 各自“最后一次高于其峰值*peak_drop_frac”
            索引中**较早**者 + 1 + tail_buffer_hours。
            → 当 obs 有长平台尾巴、但模型已快速衰减时, 按模型衰减截断,
              不再保存完整生命周期。
    """
    v = np.asarray(v_obz_kts, dtype=float)
    T = int(v.size)
    valid = np.isfinite(v)
    above_min = valid & (v > float(min_intensity_kts))
    if not above_min.any():
        return 0, T
    i0 = int(np.where(above_min)[0][0])

    obs_last = _last_strong_idx(v, i0, min_intensity_kts, peak_drop_frac)
    if obs_last is None:
        return i0, T

    last_strong = obs_last
    if model_kts is not None:
        m_last = _last_strong_idx(model_kts, i0, min_intensity_kts, peak_drop_frac)
        if m_last is not None:
            last_strong = min(obs_last, m_last)   # 谁先衰减听谁的

    i1 = min(T, last_strong + 1 + int(tail_buffer_hours))
    i1 = max(i1, i0 + 2)
    return i0, i1


# split 配色（低饱和度；train 最淡, val/test 较显眼但仍柔和）
SPLIT_COLORS = {'train': '#b8bcc2', 'val': '#7e6bd6', 'test': '#e07a3a'}
SPLIT_BG_ALPHA = {'train': 0.10, 'val': 0.16, 'test': 0.18}
SPLIT_LABEL = {'train': 'TRAIN', 'val': 'VAL', 'test': 'TEST'}


def _draw_fastref_compare(times, v_obz_kts, ml_vmax_kts, fast_vmax_kts,
                          storm_name, year, out_path,
                          min_intensity_kts=30.0,
                          peak_drop_frac=0.4,
                          tail_buffer_hours=24,
                          ml_vent=None, fast_vent=None,
                          split=None, storm_id=None):
    """三曲线对比图（IBTrACS / FAST_ML / FAST），自动截取时间窗。

    若提供 ml_vent / fast_vent（vent=chi·S），底部追加 vent 面板，
    风格复用 ensemble/plot_vs_google_vmax.py（FAST=#16a34a, FAST-ML=#2563eb）。
    """
    model_env = np.fmax(np.asarray(ml_vmax_kts, dtype=float),
                        np.asarray(fast_vmax_kts, dtype=float))
    i0, i1 = _find_plot_window(
        v_obz_kts,
        min_intensity_kts=min_intensity_kts,
        peak_drop_frac=peak_drop_frac,
        tail_buffer_hours=tail_buffer_hours,
        model_kts=model_env,
    )
    sl = slice(i0, i1)

    times_p = times[sl]
    v_obz_p = np.asarray(v_obz_kts)[sl]
    ml_p = np.asarray(ml_vmax_kts)[sl]
    fast_p = np.asarray(fast_vmax_kts)[sl]

    has_vent = ml_vent is not None and fast_vent is not None
    if has_vent:
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(14, 7), facecolor='white', sharex=True,
            gridspec_kw={'height_ratios': [2.2, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
        axv = None
    ax.set_facecolor('white')

    ax.plot(times_p, v_obz_p, label='IBTrACS', color='black',   lw=2.6, alpha=0.95)
    ax.plot(times_p, ml_p,    label='FAST_ML', color='#3a86ff', lw=2.6, alpha=0.95)
    ax.plot(times_p, fast_p,  label='FAST',    color='#5ec64a', lw=2.6, alpha=0.95)

    ax.set_ylabel('Intensity (knots)', fontsize=22)
    ax.set_ylim(0, 200)
    if len(times_p) > 0:
        ax.set_xlim(pd.Timestamp(times_p.min()), pd.Timestamp(times_p.max()))
    ax.tick_params(axis='both', labelsize=18)

    title = storm_id if storm_id else (f'{year} {storm_name}' if year else storm_name)
    ax.set_title(title, fontsize=26, fontweight='bold')
    ax.legend(loc='best', fontsize=20, framealpha=0.9)
    ax.grid(True, alpha=0.3, color='grey')

    # split 角标（只标 val/test, train 不强调）
    if split in ('val', 'test'):
        sc = SPLIT_COLORS[split]
        ax.text(0.012, 0.94, SPLIT_LABEL[split], transform=ax.transAxes,
                fontsize=20, fontweight='bold', color='white', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=sc,
                          edgecolor='none', alpha=0.95))

    # ── 底部 vent (chi·S) 面板（复用 ensemble 风格）─────────────────────────────
    if has_vent:
        mlv = np.asarray(ml_vent)[sl]
        fav = np.asarray(fast_vent)[sl]
        axv.set_facecolor('white')
        axv.plot(times_p, fav, color='#16a34a', lw=3.0, alpha=0.95)
        axv.plot(times_p, mlv, color='#2563eb', lw=3.0, alpha=0.95)
        axv.set_ylabel('vent  (chi*S)', fontsize=22)
        axv.set_xlabel('Date', fontsize=22)
        axv.set_ylim(0, None)
        axv.grid(True, alpha=0.3, color='grey')
        axv.tick_params(axis='both', labelsize=18)
        xax = axv
    else:
        ax.set_xlabel('Date', fontsize=22)
        xax = ax

    xax.xaxis.set_major_locator(mdates.AutoDateLocator())
    xax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    fig.autofmt_xdate()

    out_path = Path(out_path).with_suffix('.svg')
    fig.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_fastref_style(ds, result, out_path, min_intensity_kts=30.0,
                        peak_drop_frac=0.4, tail_buffer_hours=24):
    """从 (ds, result) 画单风暴对比图（评估流程内联调用入口）。"""
    T = result['T']
    times = ds.get('times')
    if times is not None:
        times = pd.to_datetime(np.asarray(times).ravel()[:T])
    else:
        times = pd.date_range(start='2000-01-01', periods=T, freq='h')

    storm_name = _short_name(ds.get('hurricane', 'unknown'))
    year = _infer_year(ds)

    # vent = chi·S（ML 用预测 chi/s；FAST 用校准 chi_ref/s_ref）
    ml_vent = np.asarray(result['ml_chi'], dtype=float) * np.asarray(result['ml_s'], dtype=float)
    fast_vent = np.asarray(result['fast_chi'], dtype=float) * np.asarray(result['fast_s'], dtype=float)

    _draw_fastref_compare(
        times,
        result['v_obz_kts'],
        result['ml_vmax_kts'],
        result['fast_vmax_kts'],
        storm_name=storm_name,
        year=year,
        out_path=out_path,
        min_intensity_kts=min_intensity_kts,
        peak_drop_frac=peak_drop_frac,
        tail_buffer_hours=tail_buffer_hours,
        ml_vent=ml_vent,
        fast_vent=fast_vent,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate TwoStreamFASTModel vs FAST reference on all storms')
    parser.add_argument('--ckpt', default=None, help='Model checkpoint (.pth)')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parent / 'training_data'))
    parser.add_argument('--train_years', default='2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022')
    parser.add_argument('--val_years', default='2023,2024')
    parser.add_argument('--test_years', default='')
    parser.add_argument('--model_tag', default='B_D2')
    parser.add_argument('--arch', default='dual', choices=['dual', 'mono'],
                        help='消融模型架构 (dual=双流, mono=单体共享 CNN)')
    parser.add_argument('--pool', default='full', choices=['full', 'avg', 'avg_ext'],
                        help='编码器池化模式 (full=Avg+Max双池化, avg=仅Avg, avg_ext=Avg+max/min极值)')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--storm_name', default=None,
                        help='Only evaluate storms matching this name')
    parser.add_argument('--min_vmax_kts', type=float, default=45.0)
    parser.add_argument('--min_duration_h', type=int, default=72)
    parser.add_argument('--no_plot', action='store_true')
    parser.add_argument('--plot_only', action='store_true',
                        help='Skip evaluation; replot yearly bar + per-storm '
                             'comparison plots from existing CSVs under --out_dir')
    parser.add_argument('--plot_min_kts', type=float, default=30.0,
                        help='Lower bound of IBTrACS used as front-end trim')
    parser.add_argument('--plot_peak_drop_frac', type=float, default=0.4,
                        help='Tail-trim once IBTrACS drops below peak*frac '
                             '(or below plot_min_kts, whichever larger)')
    parser.add_argument('--plot_tail_buffer_h', type=int, default=12,
                        help='Hours of padding kept after the decay endpoint')
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _parse_years(s):
        ys = set()
        for tok in str(s).split(','):
            tok = tok.strip()
            if not tok:
                continue
            if '-' in tok and len(tok.split('-')) == 2:
                a, b = tok.split('-')
                try:
                    ys.update(range(int(a), int(b) + 1))
                except ValueError:
                    ys.add(int(tok))
            else:
                try:
                    ys.add(int(tok))
                except ValueError:
                    pass
        return sorted(ys)

    train_years = _parse_years(args.train_years)
    val_years = _parse_years(args.val_years)
    test_years = _parse_years(args.test_years)

    # ── plot_only: 仅从已有 CSV 重绘柱状图 + 所有单风暴对比图 ────────────────
    if args.plot_only:
        csv_path = out_dir / 'storm_metrics.csv'
        if not csv_path.exists():
            raise FileNotFoundError(
                f'{csv_path} not found. Run full eval first to produce it.')
        df_metrics = pd.read_csv(csv_path)
        all_rows = df_metrics.to_dict('records')

        bar_path = out_dir / f'yearly_rmse_bar_{args.model_tag}.svg'
        _plot_yearly_rmse_bar(
            all_rows, train_years, val_years, test_years,
            args.model_tag, bar_path,
        )
        print(f'Saved {bar_path}')

        saved_storm = _plot_storm_rmse(all_rows, val_years, test_years,
                                       args.model_tag, out_dir)
        for p in saved_storm:
            print(f'Saved {p}')

        if not args.no_plot:
            print(f'Replotting per-storm comparison plots in {out_dir} ...')
            replotted, skipped = 0, 0
            for row in all_rows:
                try:
                    year_i = int(row['year'])
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                hurricane = str(row['hurricane'])
                split = str(row.get('split', 'all'))
                ts_csv = (out_dir / split / args.model_tag / str(year_i)
                          / f'{hurricane}_timeseries.csv')
                if not ts_csv.exists():
                    skipped += 1
                    continue
                try:
                    ts = pd.read_csv(ts_csv)
                    ts['time'] = pd.to_datetime(ts['time'])
                    if 'is_valid' in ts.columns:
                        ts = ts[ts['is_valid'] == 1].reset_index(drop=True)
                    if ts.empty:
                        skipped += 1
                        continue
                    out_png = ts_csv.with_name(f'{hurricane}.svg')
                    ml_vent = fast_vent = None
                    if {'ml_chi_eff', 'ml_s', 'fast_chi', 'fast_s'}.issubset(ts.columns):
                        ml_vent = ts['ml_chi_eff'].to_numpy() * ts['ml_s'].to_numpy()
                        fast_vent = ts['fast_chi'].to_numpy() * ts['fast_s'].to_numpy()
                    if year_i in set(test_years):
                        split_i = 'test'
                    elif year_i in set(val_years):
                        split_i = 'val'
                    elif year_i in set(train_years):
                        split_i = 'train'
                    else:
                        split_i = None
                    _draw_fastref_compare(
                        ts['time'],
                        ts['v_obz_kts'].to_numpy(),
                        ts['ml_vmax_kts'].to_numpy(),
                        ts['fast_vmax_kts'].to_numpy(),
                        storm_name=_short_name(hurricane),
                        year=year_i,
                        out_path=out_png,
                        min_intensity_kts=args.plot_min_kts,
                        peak_drop_frac=args.plot_peak_drop_frac,
                        tail_buffer_hours=args.plot_tail_buffer_h,
                        ml_vent=ml_vent,
                        fast_vent=fast_vent,
                        split=split_i,
                        storm_id=hurricane.split('_')[0],
                    )
                    replotted += 1
                except Exception as e:
                    print(f'  [warn] {hurricane}: {e}')
                    skipped += 1
            print(f'Replotted {replotted} storms, skipped {skipped}.')
        return

    if not args.ckpt:
        raise ValueError('--ckpt is required unless --plot_only is set.')

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    print(f'Loading model from {args.ckpt}  (arch={args.arch}, pool={args.pool})')
    model = _build_model(args.arch, args.pool, dev)
    load_ckpt(model, args.ckpt, dev)
    model.eval()
    print('Model loaded.')

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    print('Loading storms...')
    tr, va, te = load_1km_storms(
        args.data_dir, 480, args.min_vmax_kts, args.min_vmax_kts, 72,
        train_years, val_years, test_years,
        None, None,
    )

    # 选择 split
    if args.split == 'train':
        storm_list = tr
        split_label = 'train'
    elif args.split == 'val':
        storm_list = va
        split_label = 'val'
    elif args.split == 'test':
        storm_list = te
        split_label = 'test'
    else:
        storm_list = tr + va + te
        split_label = 'all'

    if not storm_list:
        print(f'No storms found for split={args.split}')
        return

    if args.storm_name:
        storm_list = [d for d in storm_list if args.storm_name in str(d.get('hurricane', ''))]
        if not storm_list:
            print(f'No storms matching "{args.storm_name}"')
            return

    stats = compute_spatial_stats(tr if tr else storm_list)
    print(f'Evaluating {len(storm_list)} storms (split={split_label})...')

    # ── 逐风暴评估 ───────────────────────────────────────────────────────────
    all_rows = []
    for idx, ds in enumerate(storm_list):
        hurricane = ds.get('hurricane', f'storm_{idx}')
        year = _infer_year(ds)
        short = _short_name(hurricane)
        print(f'  [{idx+1}/{len(storm_list)}] {short} ({year})')

        try:
            result = _eval_one(model, ds, stats, dev)
        except Exception as e:
            print(f'    ERROR: {e}')
            import traceback
            traceback.print_exc()
            continue

        # ── 保存 per-storm 时间序列 CSV（用于后续 RI 等分析）──────────────────
        # 包含每个时间点：时间、经纬度、观测 / ML / FAST 的风速、势强度等
        per_storm_dir = out_dir / split_label / args.model_tag / str(year)
        per_storm_dir.mkdir(parents=True, exist_ok=True)
        per_storm_csv = per_storm_dir / f'{hurricane}_timeseries.csv'

        T = result['T']
        times_raw = result.get('times')
        if times_raw is not None:
            try:
                times_arr = pd.to_datetime(np.asarray(times_raw).ravel()[:T])
            except Exception:
                times_arr = pd.date_range(start='2000-01-01', periods=T, freq='h')
        else:
            times_arr = pd.date_range(start='2000-01-01', periods=T, freq='h')

        seq_len_i = int(result.get('seq_len', T))
        t_start_i = int(result.get('t_start', 0))
        steps = np.arange(T)
        is_valid_seq = (steps < seq_len_i).astype(int)
        is_prestart = (steps < t_start_i).astype(int)

        ts_df = pd.DataFrame({
            'step':           steps,
            'time':           times_arr,
            'lat':            result['lats'],
            'lon':            result['lons'],
            'vp_kts':         result['vp_kts'],
            'v_obz_kts':      result['v_obz_kts'],
            'ml_vmax_kts':    result['ml_vmax_kts'],
            'fast_vmax_kts':  result['fast_vmax_kts'],
            'ml_v_kts':       result['ml_v_kts'],
            'fast_v_kts':     result['fast_v_kts'],
            'ml_s':           result['ml_s'],
            'ml_chi_eff':     result['ml_chi'],
            'fast_s':         result['fast_s'],
            'fast_chi':       result['fast_chi'],
            'ml_m':           result['ml_m'],
            'fast_m':         result['fast_m'],
            'is_valid':       is_valid_seq,
            'is_prestart':    is_prestart,
        })
        ts_df.to_csv(per_storm_csv, index=False)

        # 保存图
        if not args.no_plot:
            plot_path = per_storm_dir / f'{hurricane}.png'
            _plot_fastref_style(
                ds, result, plot_path,
                min_intensity_kts=args.plot_min_kts,
                peak_drop_frac=args.plot_peak_drop_frac,
                tail_buffer_hours=args.plot_tail_buffer_h,
            )

        all_rows.append({
            'split': split_label,
            'model_tag': args.model_tag,
            'year': year,
            'hurricane': hurricane,
            'ml_mse': result['ml_mse'],
            'ml_rmse_kts': result['ml_rmse_kts'],
            'ml_aligned_mse': result['ml_aligned_mse'],
            'ml_aligned_rmse_kts': result['ml_aligned_rmse_kts'],
            'fast_mse': result['fast_mse'],
            'fast_rmse_kts': result['fast_rmse_kts'],
            'gain_vs_fast_mse': result['gain_vs_fast_mse'],
            'gain_vs_fast_rmse_kts': result['gain_vs_fast_rmse_kts'],
            'gain_vs_fast_mse_aligned': result['gain_vs_fast_mse_aligned'],
            'gain_vs_fast_rmse_kts_aligned': result['gain_vs_fast_rmse_kts_aligned'],
            'beat_fast': result['beat_fast'],
        })

    if not all_rows:
        print('No results to save.')
        return

    # ── 保存 CSV ─────────────────────────────────────────────────────────────
    storm_csv = out_dir / 'storm_metrics.csv'
    with open(storm_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f'Saved {storm_csv}')

    # 汇总
    def _safe_mean(arr, key):
        vals = [r[key] for r in arr if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else np.nan

    # 汇总：RMSE = sqrt(平均 MSE) * MS_TO_KNOTS，与旧 eval 一致
    ml_mse_mean = _safe_mean(all_rows, 'ml_mse')
    ml_aligned_mse_mean = _safe_mean(all_rows, 'ml_aligned_mse')
    fast_mse_mean = _safe_mean(all_rows, 'fast_mse')
    gain_mse_mean = _safe_mean(all_rows, 'gain_vs_fast_mse')
    gain_mse_aligned_mean = _safe_mean(all_rows, 'gain_vs_fast_mse_aligned')

    def _mse_to_rmse_kts(mse):
        return float(np.sqrt(mse) * MS_TO_KNOTS) if np.isfinite(mse) and mse >= 0 else np.nan

    summary = {
        'split': split_label,
        'model_tag': args.model_tag,
        'ml_mse': ml_mse_mean,
        'ml_rmse_kts': _mse_to_rmse_kts(ml_mse_mean),
        'ml_aligned_mse': ml_aligned_mse_mean,
        'ml_aligned_rmse_kts': _mse_to_rmse_kts(ml_aligned_mse_mean),
        'fast_mse': fast_mse_mean,
        'fast_rmse_kts': _mse_to_rmse_kts(fast_mse_mean),
        'gain_vs_fast_mse': gain_mse_mean,
        'gain_vs_fast_rmse_kts': (
            _mse_to_rmse_kts(fast_mse_mean) - _mse_to_rmse_kts(ml_mse_mean)
            if np.isfinite(fast_mse_mean) and np.isfinite(ml_mse_mean) else np.nan
        ),
        'gain_vs_fast_mse_aligned': gain_mse_aligned_mean,
        'gain_vs_fast_rmse_kts_aligned': (
            _mse_to_rmse_kts(fast_mse_mean) - _mse_to_rmse_kts(ml_aligned_mse_mean)
            if np.isfinite(fast_mse_mean) and np.isfinite(ml_aligned_mse_mean) else np.nan
        ),
        'beat_fast': 1 if (np.isfinite(gain_mse_mean) and gain_mse_mean > 0) else 0,
    }
    summary_csv = out_dir / 'summary_metrics.csv'
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(f'Saved {summary_csv}')

    if not args.no_plot:
        yearly_bar_path = out_dir / f'yearly_rmse_bar_{args.model_tag}.png'
        try:
            _plot_yearly_rmse_bar(
                all_rows, train_years, val_years, test_years,
                args.model_tag, yearly_bar_path,
            )
            print(f'Saved {yearly_bar_path}')
        except Exception as e:
            print(f'[warn] failed to plot yearly bar: {e}')

    print(f'\nSummary: n={len(all_rows)}, '
          f'ML RMSE={summary["ml_aligned_rmse_kts"]:.2f} kts, '
          f'FAST RMSE={summary["fast_rmse_kts"]:.2f} kts, '
          f'gain={summary["gain_vs_fast_rmse_kts"]:+.2f} kts')


if __name__ == '__main__':
    main()
