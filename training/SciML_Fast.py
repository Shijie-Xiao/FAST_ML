"""Honest FAST: real-data train/val only. Train 2003-2021, val 2022, test 2023; seq_len=240, valid_mask for pad."""
import os
import pickle
import glob
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from einops import rearrange

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import einops
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "einops"])
    import einops

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- 1. Physics ---
class PhysicalBound(nn.Module):
    def __init__(self, low=0.0, high=1.0, transition_width=0.05):
        super().__init__()
        self.low, self.high, self.eps = low, high, transition_width
    def forward(self, x):
        return self.low + F.softplus(x - self.low, beta=1.0/self.eps) - F.softplus(x - self.high, beta=1.0/self.eps)

class FAST_Physics_Engine(nn.Module):
    def __init__(self, dt=1.0, sub_steps=4):
        super().__init__()
        self.dt, self.sub_steps = dt, sub_steps
        self.step_size = dt / sub_steps
        self.coeff = 0.5 * (0.0012 / 1400.0) * 3600.0
        self.m_bound = PhysicalBound(0.0, 1.0, 0.02)
        self.v_bound = PhysicalBound(0.0, 200.0, 1.0)

    def _derivs(self, V, m, xs, scalars):
        alpha, beta, gamma, vp = scalars[:, 0:1], scalars[:, 1:2], scalars[:, 2:3], scalars[:, 3:4]
        is_ocean = (vp > 1.0).float()
        vp = torch.where(torch.isnan(vp) | (vp <= 0), torch.tensor(30.0, device=vp.device), vp)
        alpha = torch.where(torch.isnan(alpha), torch.tensor(1.0, device=alpha.device), alpha)
        beta = torch.where(torch.isnan(beta), torch.tensor(0.5, device=beta.device), beta)
        gamma = torch.where(torch.isnan(gamma), torch.tensor(0.4, device=gamma.device), gamma)
        xs = torch.where(torch.isnan(xs), torch.tensor(0.0, device=xs.device), xs)
        m3, v2 = torch.pow(m, 3), torch.pow(V, 2)
        dV = self.coeff * (alpha * beta * vp**2 * m3 - (1.0 - gamma * m3) * v2)
        dm = self.coeff * (is_ocean * (1.0 - m) * V - xs * m)
        return dV, dm

    def forward(self, xs, prev_state, scalars):
        V, m = prev_state[:, 0:1], prev_state[:, 1:2]
        for _ in range(self.sub_steps):
            dV1, dm1 = self._derivs(V, m, xs, scalars)
            V_mid = self.v_bound(V + dV1 * self.step_size)
            m_mid = self.m_bound(m + dm1 * self.step_size)
            dV2, dm2 = self._derivs(V_mid, m_mid, xs, scalars)
            V = self.v_bound(V + 0.5 * (dV1 + dV2) * self.step_size)
            m = self.m_bound(m + 0.5 * (dm1 + dm2) * self.step_size)
        return torch.cat([V, m], dim=1)

# --- 2. Model ---
# 归一化: 从训练数据计算 mean/std，不做固定值

def compute_spatial_stats(datasets):
    """
    逐变量逐层 (per-variable per-level) 计算 mean/std，忽略 NaN/Inf。
    stats_3d: list[5 vars] of list[7 levels] of (mu, std)
    stats_2d: list[2 vars] of (mu, std)

    自动按空间尺寸分组，取 dataset 数量最多的那组（避免 20x20 与 72x72 混合 concatenate 报错）。
    """
    if not datasets:
        return None

    # 按 spatial_3d 的空间尺寸分组
    groups = {}
    for d in datasets:
        s3 = d.get('spatial_3d')
        s2 = d.get('spatial_2d')
        if s3 is None or s2 is None:
            continue
        hw = (s3.shape[-2], s3.shape[-1])  # (H, W)
        if hw not in groups:
            groups[hw] = {'s3': [], 's2': []}
        groups[hw]['s3'].append(s3)
        groups[hw]['s2'].append(s2)

    if not groups:
        return None

    # 选数量最多的尺寸组（全部是 72×72 时只有一组）
    best_hw = max(groups, key=lambda k: len(groups[k]['s3']))
    s3_list = groups[best_hw]['s3']
    s2_list = groups[best_hw]['s2']

    s3_all = np.concatenate(s3_list, axis=0)  # [B, T, 5, 7, H, W]
    s2_all = np.concatenate(s2_list, axis=0)  # [B, T, 2, H, W]

    n_var3, n_lev = s3_all.shape[2], s3_all.shape[3]
    n_var2 = s2_all.shape[2]

    # 3D：每变量每层独立统计（保留垂直梯度敏感性）
    stats_3d = []
    for vi in range(n_var3):
        level_stats = []
        for li in range(n_lev):
            x = s3_all[:, :, vi, li].ravel()
            x = x[np.isfinite(x)]
            mu  = float(np.mean(x)) if len(x) > 0 else 0.0
            std = float(np.std(x))  if len(x) > 1 else 1.0
            level_stats.append((mu, max(std, 1e-8)))
        stats_3d.append(level_stats)

    # 2D：每变量独立统计
    stats_2d = []
    for vi in range(n_var2):
        x = s2_all[:, :, vi].ravel()
        x = x[np.isfinite(x)]
        mu  = float(np.mean(x)) if len(x) > 0 else 0.0
        std = float(np.std(x))  if len(x) > 1 else 1.0
        stats_2d.append((mu, max(std, 1e-8)))

    return {'3d': stats_3d, '2d': stats_2d, 'spatial_hw': best_hw}


def normalize_spatial(spatial_3d, spatial_2d, stats=None):
    """
    逐变量逐层归一化。
    spatial_3d: [B, T, 5, 7, H, W]
    spatial_2d: [B, T, 2, H, W]
    stats 格式：{'3d': [[...7 (mu,std)...] × 5 vars], '2d': [(mu,std) × 2 vars]}
    NaN/无穷 强制替换为 0（对 CNN 安全）。
    """
    s3 = np.array(spatial_3d, dtype=np.float32, copy=True)
    s2 = np.array(spatial_2d, dtype=np.float32, copy=True)

    if stats is None:
        _s3 = s3[np.newaxis] if s3.ndim == 5 else s3
        _s2 = s2[np.newaxis] if s2.ndim == 4 else s2
        stats = compute_spatial_stats([{'spatial_3d': _s3, 'spatial_2d': _s2}])

    for vi, level_stats in enumerate(stats['3d']):
        for li, (mu, std) in enumerate(level_stats):
            s3[:, :, vi, li] = np.nan_to_num((s3[:, :, vi, li] - mu) / std, nan=0.0, posinf=0.0, neginf=0.0)

    for vi, (mu, std) in enumerate(stats['2d']):
        s2[:, :, vi] = np.nan_to_num((s2[:, :, vi] - mu) / std, nan=0.0, posinf=0.0, neginf=0.0)

    return s3, s2


# 与 run_fast_reference 一致：χ_effective 用于 xs_ref
CHI_MULTIPLIER_REF = 5
CHI_D_ATLANTIC_REF = 4.0


def _chi_calibrated_multiply(chi, chi_multiplier=CHI_MULTIPLIER_REF, chi_max=CHI_D_ATLANTIC_REF):
    """χ_effective = χ * multiplier，clip 到物理上限。与 run_fast_reference 一致。"""
    chi = np.asarray(chi, dtype=np.float64)
    chi = np.nan_to_num(chi, nan=1e-10)
    chi = np.maximum(chi, 1e-10)
    return np.clip(chi * chi_multiplier, 0.0, chi_max)


def median_filter_chi(chi, window=5):
    """对 chi 沿时间轴做滑动中位数滤波，窗口=window。chi: [B,T,1] 或 [1,T,1]，忽略 NaN。
    与 run_fast_reference 一致：极端冷水/陆上 chi 失效时填充 CHI_D_ATLANTIC_REF=4.0。"""
    chi = np.asarray(chi, dtype=np.float32)
    chi = np.nan_to_num(chi, nan=CHI_D_ATLANTIC_REF)
    orig_shape = chi.shape
    if chi.ndim == 2:
        chi = chi[np.newaxis, :, :]
    B, T, C = chi.shape
    out = np.zeros_like(chi)
    half = window // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(B):
            for c in range(C):
                for t in range(T):
                    lo, hi = max(0, t - half), min(T, t + half + 1)
                    out[b, t, c] = np.nanmedian(chi[b, lo:hi, c])
                    if np.isnan(out[b, t, c]):
                        out[b, t, c] = np.nanmean(chi[b, :, c]) if np.any(np.isfinite(chi[b, :, c])) else CHI_D_ATLANTIC_REF
    out = np.nan_to_num(out, nan=CHI_D_ATLANTIC_REF)
    return out.reshape(orig_shape)


class Spatial3D2DEncoder(nn.Module):
    """方案 B: 3D 分支(气压层) + 2D 分支(地表)，参考 Pangu 结构。"""
    def __init__(self, dim=64):
        super().__init__()
        self.branch_3d = nn.Sequential(
            nn.Conv3d(5, 32, (2, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, 64, (2, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.BatchNorm3d(64), nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.branch_2d = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fuse = nn.Linear(64 + 64, dim)

    def forward(self, x_3d, x_2d):
        # x_3d: [B*T, 5, 7, 20, 20], x_2d: [B*T, 2, 20, 20]
        f3 = self.branch_3d(x_3d).flatten(1)
        f2 = self.branch_2d(x_2d).flatten(1)
        return self.fuse(torch.cat([f3, f2], 1))


class Honest_FAST_Model(nn.Module):
    """
    3D+2D CNN 编码器；预测 chi 和 S，物理前向与 run_fast_reference 完全一致。
    当 pred_chi=chi_ref 且 pred_s=s_ref 时，输出与 run_fast_reference 完全一致。
    """
    def __init__(self, scalar_dim=4, hidden_dim=128):
        super().__init__()
        self.encoder = Spatial3D2DEncoder(dim=hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim + scalar_dim + 1, hidden_dim)  # env_feats, scalars, xs_feedback
        self.head_s = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())
        self.head_chi = nn.Sequential(nn.Linear(hidden_dim + scalar_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        nn.init.constant_(self.head_chi[-2].bias, -0.5)

    def forward(self, x_spatial_3d, x_spatial_2d, x_scalars, v_init, xs_teacher=None, teacher_forcing_ratio=0.0, m0_ref=None,
                precomp=None, env_wnds=None, utran=None, vtran=None, lats=None, use_run_fast_physics=True):
        b, seq = x_spatial_3d.shape[0], x_spatial_3d.shape[1]
        flat_3d = rearrange(x_spatial_3d, 'b t c l h w -> (b t) c l h w')
        flat_2d = rearrange(x_spatial_2d, 'b t c h w -> (b t) c h w')
        env_feats = rearrange(self.encoder(flat_3d, flat_2d), '(b t) d -> b t d', b=b)
        xs_feedback = torch.zeros(b, 1, device=x_spatial_3d.device)
        hx = torch.zeros(b, 128, device=x_spatial_3d.device)
        cx = torch.zeros(b, 128, device=x_spatial_3d.device)
        res = {'s': [], 'chi': [], 'xs': []}
        from run_fast_physics_torch import chi_calibrated_torch
        for t in range(seq):
            rnn_in = torch.cat([env_feats[:, t], x_scalars[:, t], xs_feedback], dim=1)
            hx, cx = self.lstm(rnn_in, (hx, cx))
            pred_s = torch.clamp(self.head_s(hx), 0.1, 50.0)  # 数值稳定：S 物理范围约 5-25
            pred_chi = torch.clamp(0.15 + 1.05 * self.head_chi(torch.cat([hx, x_scalars[:, t]], dim=1)), 0.15, 1.2)
            pred_xs = chi_calibrated_torch(pred_chi) * pred_s
            use_teacher = (np.random.random() < teacher_forcing_ratio) and (xs_teacher is not None)
            xs_feedback = xs_teacher[:, t, :] if use_teacher else pred_xs.detach()
            res['s'].append(pred_s); res['chi'].append(pred_chi); res['xs'].append(pred_xs)
        pred_s = torch.stack(res['s'], 1)
        pred_chi = torch.stack(res['chi'], 1)
        pred_xs = torch.stack(res['xs'], 1)
        if use_run_fast_physics and precomp is not None and env_wnds is not None and utran is not None and vtran is not None and lats is not None:
            from run_fast_physics_torch import run_fast_physics_torch
            v_max, v_fast, m_series = run_fast_physics_torch(pred_chi, pred_s, x_scalars, precomp, env_wnds, utran, vtran, lats)
            return {'v': v_fast, 'm': m_series, 's': pred_s, 'chi': pred_chi, 'xs': pred_xs, 'v_max': v_max}
        else:
            return {'v': torch.zeros_like(pred_s), 'm': torch.zeros_like(pred_s), 's': pred_s, 'chi': pred_chi, 'xs': pred_xs, 'v_max': None}

# --- 3. Data ---
def axi_to_max_wind_torch(tc_v, pred_s, env_wnds, utran, vtran, lats):
    """
    严格按 tc_wind.axi_to_max_wind：V_max = f(V_fast, S, env_wnds, u_T, lat)
    G = min(1, 0.8 + 0.35*(1+tanh((|lat|-35)/10)))
    使用模型预测的 S 替代原始公式中的 shear 相关项，梯度经 S 回传。
    Inputs: tc_v [B,T,1], pred_s [B,T,1], env_wnds [B,T,4], utran [B,T,1], vtran [B,T,1], lats [B,T]
    """
    G = torch.clamp(0.8 + 0.35 * (1.0 + torch.tanh((torch.abs(lats) - 35.0) / 10.0)), max=1.0)
    G = G.unsqueeze(-1) if G.dim() == 2 else G  # [B,T,1]
    u_shr = env_wnds[..., 0] - env_wnds[..., 2]  # [B,T]
    v_shr = env_wnds[..., 1] - env_wnds[..., 3]
    shear_mag = torch.sqrt(u_shr.pow(2) + v_shr.pow(2) + 1e-12)
    has_env = ~(torch.isnan(u_shr) | torch.isnan(v_shr) | (shear_mag < 1e-6))
    u_dir = torch.where(has_env, u_shr / shear_mag, torch.zeros_like(u_shr))
    v_dir = torch.where(has_env, v_shr / shear_mag, torch.zeros_like(v_shr))
    shear_coeff = 0.1 * pred_s.squeeze(-1) * tc_v.squeeze(-1) / 15.0  # [B,T]
    U_inc = G.squeeze(-1) * utran.squeeze(-1) + shear_coeff * u_dir
    V_inc = G.squeeze(-1) * vtran.squeeze(-1) + shear_coeff * v_dir
    mag_inc = torch.sqrt(U_inc.pow(2) + V_inc.pow(2) + 1e-12)
    mag_fac = torch.clamp((tc_v.squeeze(-1) * 0.5) / mag_inc, max=1.0)
    theta_opt = torch.atan2(-U_inc, V_inc)
    ug = tc_v.squeeze(-1) * (-torch.sin(theta_opt)) + U_inc * mag_fac
    vg = tc_v.squeeze(-1) * torch.cos(theta_opt) + V_inc * mag_fac
    tc_vmax = torch.sqrt(ug.pow(2) + vg.pow(2) + 1e-12)
    return tc_vmax.unsqueeze(-1)  # [B,T,1]

TRAIN_YEAR_START, TRAIN_YEAR_END = 2003, 2021
VAL_YEAR, TEST_YEAR = 2022, 2023
MIN_VMAX_KTS = 50
MS_TO_KNOTS = 1.94384
TARGET_SEQ_LEN = 240

# 时间起点：从 vmax > start_from_vmax_kts 的时刻开始（过滤热带低压等弱阶段）
START_FROM_VMAX_KTS = 40

# 人工挑选飓风：storm_include_list 仅包含列表中的；storm_exclude_list 排除列表中的。格式 "year/storm_dir"
# 例: storm_include.txt 每行 "2022/AL112022_north_atlantic_UNNAMED"

# 划分配置：按年份划分 train/val/test
TRAIN_YEARS = list(range(TRAIN_YEAR_START, TRAIN_YEAR_END + 1))  # 2003-2021
VAL_YEARS = [VAL_YEAR]   # 2022
TEST_YEARS = [TEST_YEAR]  # 2023

def _axi_to_max_wind_single_np(v_axisym, s, env, ut, vt, lat):
    """单时刻：由轴对称角平均风速 V 算最大地面风速 v_max。与 axi_to_max_wind 公式一致。"""
    v_axisym = float(np.asarray(v_axisym).flat[0])
    s = float(np.nan_to_num(np.asarray(s).flat[0], nan=0.0))
    env = np.atleast_2d(env)
    if env.shape[0] == 1:
        env = env.reshape(4)
    ut = float(np.asarray(ut).flat[0])
    vt = float(np.asarray(vt).flat[0])
    lat = float(np.asarray(lat).flat[0])
    G = min(1.0, 0.8 + 0.35 * (1.0 + np.tanh((abs(lat) - 35.0) / 10.0)))
    u_shr = env[0] - env[2]
    v_shr = env[1] - env[3]
    shear_mag = np.sqrt(u_shr**2 + v_shr**2 + 1e-12)
    has_env = not (np.isnan(u_shr) or np.isnan(v_shr) or shear_mag < 1e-6)
    u_dir = (u_shr / shear_mag) if has_env else 0.0
    v_dir = (v_shr / shear_mag) if has_env else 0.0
    shear_coeff = 0.1 * s * v_axisym / 15.0
    U_inc = G * ut + shear_coeff * u_dir
    V_inc = G * vt + shear_coeff * v_dir
    mag_inc = np.sqrt(U_inc**2 + V_inc**2 + 1e-12)
    mag_fac = min(1.0, (v_axisym * 0.5) / mag_inc) if mag_inc > 1e-12 else 0.0
    theta_opt = np.arctan2(-U_inc, V_inc)
    ug = v_axisym * (-np.sin(theta_opt)) + U_inc * mag_fac
    vg = v_axisym * np.cos(theta_opt) + V_inc * mag_fac
    return float(np.sqrt(ug**2 + vg**2 + 1e-12))


def _invert_vmax_to_V_axisym_np(v_max_obs, s, env, ut, vt, lat, v_lo=0.0, v_hi=200.0, tol=0.01, max_iter=50):
    """由观测最大风速 v_obz 反算轴对称角平均风速 V_axisym（二分）。"""
    if np.isnan(v_max_obs) or v_max_obs <= 0:
        return np.nan
    v_max_obs = float(v_max_obs)
    for _ in range(max_iter):
        v_mid = (v_lo + v_hi) * 0.5
        v_max_mid = _axi_to_max_wind_single_np(v_mid, s, env, ut, vt, lat)
        if abs(v_max_mid - v_max_obs) < tol:
            return v_mid
        if v_max_mid < v_max_obs:
            v_lo = v_mid
    else:
            v_hi = v_mid
    return (v_lo + v_hi) * 0.5


def calculate_m0_from_fast(v, dv_dt, alpha, beta, gamma, vp, device=None):
    """Infer m0 from FAST equation. v, dv_dt, alpha, beta, gamma, vp: [B] or [B,1]."""
    coeff = 0.5 * (0.0012 / 1400.0) * 3600.0
    if device is None:
        device = v.device if isinstance(v, torch.Tensor) else torch.device('cpu')
    for t in [v, dv_dt, alpha, beta, gamma, vp]:
        if not isinstance(t, torch.Tensor):
            raise TypeError("All inputs must be tensors")
    v = v.to(device).float().squeeze()
    dv_dt = dv_dt.to(device).float().squeeze()
    alpha, beta, gamma, vp = alpha.to(device).float().squeeze(), beta.to(device).float().squeeze(), gamma.to(device).float().squeeze(), vp.to(device).float().squeeze()
    num = dv_dt / coeff + v**2
    den = alpha * beta * vp**2 + gamma * v**2 + 1e-8
    m0 = torch.clamp(num / den, min=0.0).pow(1.0/3.0)
    return torch.clamp(m0, min=0.01, max=1.0)

def _trim_from_vmax_kts(data, data_dict, start_vmax_kts):
    """从 vmax >= start_vmax_kts 的时刻开始截断，丢弃之前的弱阶段。返回 (s3d, s2d, data_trimmed, data_dict_trimmed) 或 None。"""
    v_gt = data['v_gt'][0, :, 0]
    vmax_kts = np.asarray(v_gt, dtype=np.float64) * MS_TO_KNOTS
    valid = np.isfinite(vmax_kts) & ~np.isnan(vmax_kts)
    t0 = 0
    for t in range(len(vmax_kts)):
        if valid[t] and vmax_kts[t] >= start_vmax_kts:
            t0 = t
            break
    if t0 >= len(vmax_kts):
        return None
    T_trim = len(vmax_kts) - t0
    if T_trim < 3:
        return None
    s3d = data['spatial_3d'][:, t0:, :, :, :, :]
    s2d = data['spatial_2d'][:, t0:, :, :, :]
    data_trimmed = {k: (data[k][:, t0:, ...] if hasattr(data[k], 'shape') and data[k].ndim >= 2 else data[k])
                    for k in ['scalars', 'chi_ref', 's_ref', 'xs_ref', 'v_init', 'v_gt'] if k in data}
    data_trimmed['hurricane'] = data.get('hurricane', '')
    if 'times' in data:
        data_trimmed['times'] = np.asarray(data['times']).ravel()[t0:]
    data_dict_trimmed = {
        'env_wnds': data_dict['env_wnds'][:, t0:, :],
        'utran': data_dict['utran'][:, t0:, :],
        'vtran': data_dict['vtran'][:, t0:, :],
        'lats': data_dict['lats'][:, t0:] if data_dict['lats'].ndim > 1 else data_dict['lats'][t0:],
        'lons': np.atleast_2d(data_dict['lons'])[:, t0:] if 'lons' in data_dict else np.zeros((1, T_trim), dtype=np.float32),
    }
    return s3d, s2d, data_trimmed, data_dict_trimmed


def _load_storm_list(path_or_list):
    """从文件路径或列表加载飓风标识。每行格式 year/storm_dir，如 2022/AL112022_north_atlantic_UNNAMED。"""
    if path_or_list is None:
        return None
    if isinstance(path_or_list, (list, tuple)):
        return set(str(x).strip() for x in path_or_list if x)
    path = Path(path_or_list)
    if not path.exists():
        alt = Path(__file__).resolve().parent / path.name
        path = alt if alt.exists() else path
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return set(line.strip().split('#')[0].strip() for line in f if line.strip() and not line.strip().startswith('#'))


def _pad_or_truncate_to_fixed_len(data_dict, s3d, s2d, data, target_len, median_filter_chi_fn):
    """统一填充或截断到 target_len，返回 (spatial_3d, spatial_2d, scalars, chi_ref, s_ref, xs_ref, v_init, v_gt, env_wnds, utran, vtran, lats, lons, seq_len_valid, times)。"""
    T_full = s3d.shape[1]
    if T_full >= target_len:
        seq_len_valid = target_len
        spatial_3d = s3d[:, :target_len, :, :, :, :]
        spatial_2d = s2d[:, :target_len, :, :, :]
        scalars = data['scalars'][:, :target_len, :]
        chi_raw = np.nan_to_num(data['chi_ref'][:, :target_len, :], nan=CHI_D_ATLANTIC_REF)
        chi_ref = median_filter_chi_fn(chi_raw, window=5)
        s_ref = data['s_ref'][:, :target_len, :]
        chi_eff = _chi_calibrated_multiply(chi_ref)
        xs_ref = s_ref * chi_eff
        v_init = data['v_init'][:, :target_len, :]
        v_gt = data['v_gt'][:, :target_len, :]
        env_wnds = data_dict['env_wnds'][:, :target_len, :]
        utran = data_dict['utran'][:, :target_len, :]
        vtran = data_dict['vtran'][:, :target_len, :]
        lats = data_dict['lats'][:, :target_len]
        lons = data_dict['lons'][:, :target_len] if 'lons' in data_dict else np.zeros((1, target_len), dtype=np.float32)
        times_out = np.asarray(data['times']).ravel()[:target_len] if 'times' in data else None
    else:
        seq_len_valid = T_full
        pad_len = target_len - T_full
        spatial_3d = np.concatenate([s3d, np.repeat(s3d[:, -1:], pad_len, axis=1)], axis=1)
        spatial_2d = np.concatenate([s2d, np.repeat(s2d[:, -1:], pad_len, axis=1)], axis=1)
        scalars = np.concatenate([data['scalars'], np.repeat(data['scalars'][:, -1:], pad_len, axis=1)], axis=1)
        chi_raw = np.concatenate([data['chi_ref'], np.repeat(data['chi_ref'][:, -1:], pad_len, axis=1)], axis=1)
        chi_raw = np.nan_to_num(chi_raw, nan=CHI_D_ATLANTIC_REF)
        chi_ref = median_filter_chi_fn(chi_raw, window=5)
        s_ref = np.concatenate([data['s_ref'], np.repeat(data['s_ref'][:, -1:], pad_len, axis=1)], axis=1)
        chi_eff = _chi_calibrated_multiply(chi_ref)
        xs_ref = s_ref * chi_eff
        v_init = np.concatenate([data['v_init'], np.repeat(data['v_init'][:, -1:], pad_len, axis=1)], axis=1)
        v_gt = np.concatenate([data['v_gt'], np.repeat(data['v_gt'][:, -1:], pad_len, axis=1)], axis=1)
        env_wnds = np.concatenate([data_dict['env_wnds'], np.repeat(data_dict['env_wnds'][:, -1:], pad_len, axis=1)], axis=1)
        utran = np.concatenate([data_dict['utran'], np.repeat(data_dict['utran'][:, -1:], pad_len, axis=1)], axis=1)
        vtran = np.concatenate([data_dict['vtran'], np.repeat(data_dict['vtran'][:, -1:], pad_len, axis=1)], axis=1)
        lats = np.concatenate([data_dict['lats'], np.repeat(data_dict['lats'][:, -1:], pad_len, axis=1)], axis=1)
        lons = np.concatenate([data_dict['lons'], np.repeat(data_dict['lons'][:, -1:], pad_len, axis=1)], axis=1) if 'lons' in data_dict else np.zeros((1, target_len), dtype=np.float32)
        if 'times' in data and pd is not None:
            try:
                times_arr = np.asarray(data['times']).ravel()[:T_full]
                last_dt = pd.to_datetime(times_arr[-1])
                pad_dt = pd.date_range(start=last_dt + pd.Timedelta(hours=1), periods=pad_len, freq='h')
                times_out = np.concatenate([times_arr, pad_dt.to_numpy()])
            except Exception:
                times_out = None
        else:
            times_out = None
    return spatial_3d, spatial_2d, scalars, chi_ref, s_ref, xs_ref, v_init, v_gt, env_wnds, utran, vtran, lats, lons, seq_len_valid, times_out


def build_storms_by_year(data_dir, seq_len=TARGET_SEQ_LEN, min_vmax_kts=MIN_VMAX_KTS, start_from_vmax_kts=START_FROM_VMAX_KTS,
                         storm_include_list=None, storm_exclude_list=None, require_1km=False):
    """
    按固定时间长度加载数据，构建 storms_by_year = {year: [(storm_dir_name, dataset), ...]}。
    时间起点从 vmax >= start_from_vmax_kts 开始；storm_include_list 仅包含列表中的，storm_exclude_list 排除。
    require_1km=True: 只加载有 _spatial_1000km.pkl (72×72) 的风暴；无该文件的直接跳过。
    格式 "year/storm_dir"，如 2022/AL112022_north_atlantic_UNNAMED。
    """
    data_dir = Path(data_dir)
    storms_by_year = {}
    min_vmax_ms = min_vmax_kts / MS_TO_KNOTS
    target_len = int(seq_len)
    include_set = _load_storm_list(storm_include_list)
    exclude_set = _load_storm_list(storm_exclude_list)
    all_years = sorted([int(d.name) for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    for year in all_years:
        year_dir = data_dir / str(year)
        if not year_dir.is_dir():
            continue
        storms_by_year[year] = []
        for storm_dir in sorted(year_dir.iterdir()):
            if not storm_dir.is_dir():
                continue
            pkl_files = list(storm_dir.glob('*_dataset.pkl'))
            if not pkl_files:
                continue
            try:
                with open(pkl_files[0], 'rb') as f:
                    data = pickle.load(f)
            except Exception:
                continue
            storm_id = f"{year}/{storm_dir.name}"
            if exclude_set and storm_id in exclude_set:
                continue
            if include_set and storm_id not in include_set:
                continue
            # 优先使用 1000km 空间场（若存在）
            _spatial_1km_path = pkl_files[0].parent / pkl_files[0].name.replace('_dataset.pkl', '_spatial_1000km.pkl')
            if _spatial_1km_path.exists():
                try:
                    with open(_spatial_1km_path, 'rb') as _f:
                        _sp1km = pickle.load(_f)
                    s3d = _sp1km.get('spatial_3d_1km')
                    s2d = _sp1km.get('spatial_2d_1km')
                    if s3d is None or s2d is None:
                        raise ValueError("spatial_3d_1km/spatial_2d_1km missing")
                except Exception as _e:
                    warnings.warn(f"Failed to load spatial_1000km for {storm_dir.name}: {_e}; fallback to 20x20")
                    s3d = data.get('spatial_3d')
                    s2d = data.get('spatial_2d')
            else:
                s3d = data.get('spatial_3d')
                s2d = data.get('spatial_2d')
            if s3d is None or s2d is None:
                continue
            T_full = s3d.shape[1]
            v_gt_full = data['v_gt'][0, :, 0]
            vmax_ms = np.nanmax(v_gt_full) if np.any(~np.isnan(v_gt_full)) else 0.0
            if vmax_ms < min_vmax_ms:
                continue
            env_wnds = data.get('env_wnds')
            utran = data.get('utran')
            vtran = data.get('vtran')
            lats = data.get('lats')
            lons = data.get('lons')
            if env_wnds is None:
                env_wnds = np.full((1, T_full, 4), np.nan, dtype=np.float32)
            if utran is None:
                utran = np.zeros((1, T_full, 1), dtype=np.float32)
            if vtran is None:
                vtran = np.zeros((1, T_full, 1), dtype=np.float32)
            if lats is None:
                lats = np.zeros((T_full,), dtype=np.float32)
            if env_wnds.ndim == 2:
                env_wnds = env_wnds[np.newaxis, :, :]
            if utran.ndim == 1:
                utran = utran[np.newaxis, :, np.newaxis]
            elif utran.ndim == 2:
                utran = utran[np.newaxis, :, :] if utran.shape[-1] == 1 else utran[:, :, np.newaxis]
            if vtran.ndim == 1:
                vtran = vtran[np.newaxis, :, np.newaxis]
            elif vtran.ndim == 2:
                vtran = vtran[np.newaxis, :, :] if vtran.shape[-1] == 1 else vtran[:, :, np.newaxis]
            if lats.ndim == 1:
                lats = lats[np.newaxis, :]
            if lons is None:
                lons = np.zeros((T_full,), dtype=np.float32)
            if lons.ndim == 1:
                lons = lons[np.newaxis, :]
            data_dict = {'env_wnds': env_wnds, 'utran': utran, 'vtran': vtran, 'lats': lats, 'lons': lons}
            trim_result = _trim_from_vmax_kts(data, data_dict, start_from_vmax_kts)
            if trim_result is None:
                continue
            s3d, s2d, data, data_dict = trim_result
            spatial_3d, spatial_2d, scalars, chi_ref, s_ref, xs_ref, v_init, v_gt, env_wnds, utran, vtran, lats, lons, seq_len_valid, times_out = _pad_or_truncate_to_fixed_len(
                data_dict, s3d, s2d, data, target_len, median_filter_chi)
            v_gt_np = np.asarray(v_gt[0, :seq_len_valid, 0], dtype=np.float32)
            scalars_np = np.asarray(scalars[0, :seq_len_valid, :], dtype=np.float32)
            T_val = len(v_gt_np)
            # m0 由观测最大风速 v_obz 反算轴对称角平均风速 V_axisym，再用 FAST 公式得 m0（区分角平均与最大地面风速）
            s_r = s_ref[0, :T_val, 0]
            ew = env_wnds[0, :T_val, :]
            ut = utran[0, :T_val, 0]
            vt = vtran[0, :T_val, 0]
            la = lats[0, :T_val] if lats.ndim > 1 else np.asarray(lats).flat[:T_val]
            V_axisym_np = np.full(T_val, np.nan, dtype=np.float32)
            for i in range(T_val):
                V_axisym_np[i] = _invert_vmax_to_V_axisym_np(
                    v_gt_np[i], s_r[i], ew[i], ut[i], vt[i], la[i])
            dv_dt_axisym = np.zeros(T_val, dtype=np.float32)
            if T_val > 1:
                dv_dt_axisym[0] = (V_axisym_np[1] - V_axisym_np[0]) if not np.isnan(V_axisym_np[0]) and not np.isnan(V_axisym_np[1]) else 0.0
                if T_val > 2:
                    for i in range(1, T_val - 1):
                        if not np.isnan(V_axisym_np[i-1]) and not np.isnan(V_axisym_np[i+1]):
                            dv_dt_axisym[i] = (V_axisym_np[i+1] - V_axisym_np[i-1]) * 0.5
                    if not np.isnan(V_axisym_np[-2]) and not np.isnan(V_axisym_np[-1]):
                        dv_dt_axisym[-1] = V_axisym_np[-1] - V_axisym_np[-2]
                else:
                    if not np.isnan(V_axisym_np[-1]) and not np.isnan(V_axisym_np[-2]):
                        dv_dt_axisym[-1] = V_axisym_np[-1] - V_axisym_np[-2]
            v_t = torch.from_numpy(V_axisym_np).float().unsqueeze(1)
            dv_dt = torch.from_numpy(dv_dt_axisym).float().unsqueeze(1)
            m0_vals = []
            for i in range(min(5, T_val)):
                if np.isnan(V_axisym_np[i]) or V_axisym_np[i] <= 0:
                    continue
                m0_i = calculate_m0_from_fast(v_t[i:i+1], dv_dt[i:i+1], torch.from_numpy(scalars_np[i:i+1, 0]).float(),
                    torch.from_numpy(scalars_np[i:i+1, 1]).float(), torch.from_numpy(scalars_np[i:i+1, 2]).float(),
                    torch.from_numpy(scalars_np[i:i+1, 3]).float(), device=torch.device('cpu'))
                if not torch.isnan(m0_i) and 0 < m0_i.item() <= 1.0:
                    m0_vals.append(m0_i.item())
            m0_ref = np.nanmean(m0_vals) if m0_vals else 0.5
            dataset = {'spatial_3d': spatial_3d, 'spatial_2d': spatial_2d, 'scalars': scalars, 'chi_ref': chi_ref, 's_ref': s_ref, 'xs_ref': xs_ref,
                       'v_init': v_init, 'v_gt': v_gt, 'hurricane': data['hurricane'], 'seq_len': seq_len_valid,
                       'm0_ref': np.array([[m0_ref]], dtype=np.float32),
                       'env_wnds': env_wnds, 'utran': utran, 'vtran': vtran, 'lats': lats, 'lons': lons, 'year': year, 'storm_dir': storm_dir.name}
            if times_out is not None:
                dataset['times'] = times_out
            storms_by_year[year].append((storm_dir.name, dataset))
    return storms_by_year


def split_storms_by_years(storms_by_year, train_years=None, val_years=None, test_years=None):
    """
    按年份划分 train/val/test。storms_by_year = {year: [(storm_dir, dataset), ...]}
    返回 train_list, val_list, test_list（每个元素为 dataset 字典）。
    """
    train_years = train_years or TRAIN_YEARS
    val_years = val_years or VAL_YEARS
    test_years = test_years or TEST_YEARS
    train_list, val_list, test_list = [], [], []
    for year, storms in storms_by_year.items():
        for _name, ds in storms:
            if year in train_years:
                train_list.append(ds)
            elif year in val_years:
                val_list.append(ds)
            elif year in test_years:
                test_list.append(ds)
    return train_list, val_list, test_list


def load_train_val_test_data(data_dir='training_data', seq_len=TARGET_SEQ_LEN, min_vmax_kts=MIN_VMAX_KTS, precalc_dir='precalc_data', truncate_after_landfall=False,
                              train_years=None, val_years=None, test_years=None, start_from_vmax_kts=START_FROM_VMAX_KTS,
                              storm_include_list=None, storm_exclude_list=None):
    """
    按固定时间长度加载数据，时间不足的飓风用掩码填补（valid_mask）。
    时间起点从 vmax >= start_from_vmax_kts；可人工指定 storm_include_list / storm_exclude_list。
    """
    storms_by_year = build_storms_by_year(data_dir, seq_len=seq_len, min_vmax_kts=min_vmax_kts,
                                          start_from_vmax_kts=start_from_vmax_kts,
                                          storm_include_list=storm_include_list, storm_exclude_list=storm_exclude_list)
    train_list, val_list, test_list = split_storms_by_years(storms_by_year, train_years, val_years, test_years)
    target_len = int(seq_len)
    print(f"Storms by year: {[(y, len(s)) for y, s in sorted(storms_by_year.items())]}")
    print(f"Data: train={len(train_list)}, val={len(val_list)}, test={len(test_list)} (seq_len={target_len}, mask for padded)")
    return train_list, val_list, test_list

def create_batch(datasets, batch_size, device, shuffle=True, spatial_stats=None):
    """Build batch from list; optional shuffle=False for eval order. spatial_stats 来自 compute_spatial_stats。"""
    if len(datasets) == 0:
        raise ValueError("Empty dataset list")
    n = min(batch_size, len(datasets))
    selected = np.random.choice(len(datasets), size=n, replace=False) if shuffle else np.arange(n)
    s3d = np.concatenate([datasets[i]['spatial_3d'] for i in selected], 0)
    s2d = np.concatenate([datasets[i]['spatial_2d'] for i in selected], 0)
    s3d_n, s2d_n = normalize_spatial(s3d, s2d, stats=spatial_stats)
    batch = {
        'spatial_3d': torch.from_numpy(s3d_n).float().to(device),
        'spatial_2d': torch.from_numpy(s2d_n).float().to(device),
        'scalars': torch.cat([torch.from_numpy(datasets[i]['scalars']).float() for i in selected], 0).to(device),
        'chi_ref': torch.cat([torch.from_numpy(datasets[i]['chi_ref']).float() for i in selected], 0).to(device),
        's_ref': torch.cat([torch.from_numpy(datasets[i]['s_ref']).float() for i in selected], 0).to(device),
        'xs_ref': torch.cat([torch.from_numpy(datasets[i]['xs_ref']).float() for i in selected], 0).to(device),
        'v_init': torch.cat([torch.from_numpy(datasets[i]['v_init']).float() for i in selected], 0).to(device),
        'v_gt': torch.cat([torch.from_numpy(datasets[i]['v_gt']).float() for i in selected], 0).to(device),
    }
    B, T = batch['v_gt'].shape[0], batch['v_gt'].shape[1]
    seq_lens = [datasets[i].get('seq_len', datasets[i]['v_gt'].shape[1]) for i in selected]
    batch['seq_len'] = torch.tensor(seq_lens, dtype=torch.long, device=device)
    batch['valid_mask'] = (torch.arange(T, device=device).unsqueeze(0) < batch['seq_len'].unsqueeze(1))
    if 'm0_ref' in datasets[0]:
        batch['m0_ref'] = torch.cat([torch.from_numpy(datasets[i]['m0_ref']).float() for i in selected], 0).to(device)
    if 'env_wnds' in datasets[0]:
        batch['env_wnds'] = torch.cat([torch.from_numpy(datasets[i]['env_wnds']).float() for i in selected], 0).to(device)
    if 'utran' in datasets[0]:
        batch['utran'] = torch.cat([torch.from_numpy(datasets[i]['utran']).float() for i in selected], 0).to(device)
    if 'vtran' in datasets[0]:
        batch['vtran'] = torch.cat([torch.from_numpy(datasets[i]['vtran']).float() for i in selected], 0).to(device)
    if 'lats' in datasets[0]:
        batch['lats'] = torch.cat([torch.from_numpy(datasets[i]['lats']).float() for i in selected], 0).to(device)
    if 'lons' in datasets[0]:
        batch['lons'] = torch.cat([torch.from_numpy(datasets[i]['lons']).float() for i in selected], 0).to(device)
    batch['times'] = [datasets[i].get('times') for i in selected]
    return batch

# --- 4. Training ---
class DynamicWeightedMSE(nn.Module):
    def __init__(self, penalty_scale=10.0, threshold=0.1):
        super().__init__()
        self.scale, self.thresh = penalty_scale, threshold
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        return torch.mean((diff**2) * (1.0 + self.scale * F.relu(diff - self.thresh)))

def _safe_filename(name):
    return (name.replace('/', '_').replace(' ', '_').strip() or 'unknown')

def plot_comparison_one_sample(ref_v, ref_s, ref_chi, pred_v, pred_s, pred_chi, pred_xs, pred_m, v_fast, m_fast, valid_len, hurricane_name, save_path, m0_ref=None, times=None):
    """Single-sample comparison: Obs, FAST, ML; S, Chi, XS; errors; m. 单位统一 kts（与 run_fast_reference 一致）。
    times: 可选，用于 x 轴日期（与 run_fast_reference 一致）；无则用 Time step (h)。"""
    n = int(valid_len)
    ref_v, ref_s, ref_chi = ref_v[:n], ref_s[:n], ref_chi[:n]
    pred_v, pred_s, pred_chi, pred_xs, pred_m = pred_v[:n], pred_s[:n], pred_chi[:n], pred_xs[:n], pred_m[:n]
    v_fast, m_fast = v_fast[:n], m_fast[:n]
    # 风速统一转为 kts（内部 m/s），确保 v_max 单位与 run_fast_reference 一致
    ref_v_kts = np.nan_to_num(ref_v, nan=0.0) * MS_TO_KNOTS
    pred_v_kts = np.nan_to_num(pred_v, nan=0.0) * MS_TO_KNOTS
    v_fast_kts = np.nan_to_num(v_fast, nan=0.0) * MS_TO_KNOTS
    # 当 pred_s/chi/xs 含 NaN 时（如 checkpoint 权重损坏），用 ref 替代以便绘图
    pred_was_nan = not (np.any(np.isfinite(pred_s)) and np.any(np.isfinite(pred_chi)))
    pred_s = np.where(np.isfinite(pred_s), pred_s, ref_s)
    pred_chi = np.where(np.isfinite(pred_chi), pred_chi, ref_chi)
    pred_xs = np.where(np.isfinite(pred_xs), pred_xs, ref_s * _chi_calibrated_multiply(ref_chi))
    chi_eff = _chi_calibrated_multiply(ref_chi)
    xs_ref = ref_s * chi_eff
    # x 轴：有 times 则用日期（与 run_fast_reference 一致），否则用时间步
    if times is not None and pd is not None:
        try:
            times_arr = np.asarray(times).ravel()
            if len(times_arr) >= n:
                x_vals = pd.to_datetime(times_arr[:n])
                x_label = 'Date'
            else:
                x_vals = np.arange(n)
                x_label = 'Time step (h)'
        except Exception:
            x_vals = np.arange(n)
            x_label = 'Time step (h)'
    else:
        x_vals = np.arange(n)
        x_label = 'Time step (h)'
    fig, ax = plt.subplots(2, 3, figsize=(14, 9), facecolor='white')
    for a in ax.flat:
        a.set_facecolor('white')
    # Wind 子图：与 run_fast_reference 一致，ylabel='Intensity (knots)'，ylim 0-200
    ax[0,0].plot(x_vals, ref_v_kts, 'k-', lw=2, label='v_obz (obs)')
    ax[0,0].plot(x_vals, v_fast_kts, 'g--', lw=1.5, label='v_max (FAST)')
    ax[0,0].plot(x_vals, pred_v_kts, 'b-', alpha=0.9, lw=1.5, label='v_max (ML)')
    ax[0,0].set_title('Wind (v_max kts)')
    ax[0,0].set_ylabel('Intensity (knots)')
    ax[0,0].set_ylim(0, 200)
    ax[0,0].legend()
    ax[0,0].grid(True, alpha=0.3, color='grey')
    if x_label == 'Date':
        ax[0,0].xaxis.set_major_locator(mdates.AutoDateLocator())
        ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    ax[0,1].plot(x_vals, ref_s, 'g--', label='Ref S'); ax[0,1].plot(x_vals, pred_s, 'r-', label='Pred S'); ax[0,1].set_title('S'); ax[0,1].legend(); ax[0,1].grid(True)
    ax[0,2].plot(x_vals, ref_chi, 'g--', label='Ref Chi'); ax[0,2].plot(x_vals, pred_chi, 'r-', label='Pred Chi'); ax[0,2].set_title('Chi'); ax[0,2].legend(); ax[0,2].grid(True)
    ax[1,0].plot(x_vals, xs_ref, 'g--', label='Ref XS'); ax[1,0].plot(x_vals, pred_xs, 'r-', label='Pred XS'); ax[1,0].set_title('XS'); ax[1,0].legend(); ax[1,0].grid(True)
    ax[1,1].plot(x_vals, pred_v_kts - ref_v_kts, 'b-', label='ML err'); ax[1,1].plot(x_vals, v_fast_kts - ref_v_kts, 'g-', label='FAST err'); ax[1,1].axhline(0, color='k', ls='--', alpha=0.3); ax[1,1].set_title('Wind error (kts)'); ax[1,1].set_ylabel('Error (kts)'); ax[1,1].legend(); ax[1,1].grid(True)
    ax[1,2].plot(x_vals, pred_m, 'r-', label='Pred m'); ax[1,2].plot(x_vals, m_fast, 'g--', label='FAST m'); (ax[1,2].axhline(m0_ref, color='k', ls=':', alpha=0.5, label=f'm0={m0_ref:.2f}') if m0_ref is not None else None); ax[1,2].set_title('m'); ax[1,2].legend(); ax[1,2].grid(True)
    for a in ax.flat:
        a.set_xlabel(x_label)
        if x_label == 'Date':
            a.xaxis.set_major_locator(mdates.AutoDateLocator())
            a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    if x_label == 'Date':
        fig.autofmt_xdate()
    supt = f'{hurricane_name} (len={n})' + (' [Pred S/Chi unavailable, using Ref]' if pred_was_nan else '')
    plt.suptitle(supt); plt.tight_layout(); plt.savefig(save_path, dpi=150, facecolor='white'); plt.close()

def _run_fast_reference_batch(s_ref, chi_ref, xs_ref, scalars, v_gt, env_wnds, utran, vtran, lats, lons=None):
    """
    按 run_fast_reference.run_fast_with_init 对每个样本计算 v_max 和 m，与 run_fast_reference.py 完全一致。
    Returns: v_max [B,T], m_series [B,T] (numpy, m/s)
    注：SciML 使用 trim 后数据（从 vmax>=25kts 起），run_fast_reference 使用完整 pkl，故同一风暴的
    时间窗与初始化不同，FAST 曲线会有差异。若需完全一致，需对同一数据窗运行两者。
    """
    try:
        import sys
        from pathlib import Path
        _ode = Path(__file__).resolve().parent
        if str(_ode) not in sys.path:
            sys.path.insert(0, str(_ode))
        from run_fast_reference import run_fast_with_init, _chi_calibrated_multiply, _median_filter_1d
    except ImportError as e:
        return None, None  # fallback to _run_physics_ref
    B, T = s_ref.shape[0], s_ref.shape[1]
    v_max_list, m_list = [], []
    for i in range(B):
        scalars_i = np.array(scalars[i].cpu().numpy(), dtype=np.float64)
        if scalars_i.ndim == 2:
            scalars_i = scalars_i[np.newaxis, :, :]
        scalars_i[0, :, 3] = _median_filter_1d(scalars_i[0, :, 3], size=3)
        chi_eff = _chi_calibrated_multiply(np.array(chi_ref[i].cpu().numpy()))
        s_i = np.array(s_ref[i].cpu().numpy())
        xs_i = np.maximum(np.nan_to_num(chi_eff * s_i, nan=1e-5), 1e-5)
        xs_i = np.atleast_3d(xs_i).reshape(1, T, 1)
        v_gt_i = v_gt[i:i+1].cpu().numpy()
        env_i = env_wnds[i:i+1].cpu().numpy() if env_wnds is not None else None
        ut_i = utran[i:i+1].cpu().numpy() if utran is not None else None
        vt_i = vtran[i:i+1].cpu().numpy() if vtran is not None else None
        la_i = lats[i].cpu().numpy() if lats is not None else None
        lo_i = lons[i].cpu().numpy() if lons is not None else None
        scalars_1 = scalars_i.reshape(1, T, 4) if scalars_i.shape != (1, T, 4) else scalars_i
        try:
            _, v_max_i, m_i = run_fast_with_init(scalars_1, xs_i, v_gt_i, env_i, ut_i, vt_i, la_i, s_ref[i:i+1].cpu().numpy(), lons=lo_i)
        except Exception:
            return None, None
        v_max_list.append(v_max_i)
        m_list.append(m_i)
    return np.stack(v_max_list, 0), np.stack(m_list, 0)


def _run_physics_ref(s_ref, chi_ref, scalars, v_init, m_init, physics_engine, device):
    """Rollout FAST physics with ref S&Chi（xs=chi_calibrated*s）；fill NaN in scalars/xs. Returns v_ref [B,T,1], m_ref [B,T,1]."""
    chi_eff = _chi_calibrated_multiply(chi_ref.cpu().numpy())
    xs_ref = torch.from_numpy((s_ref.cpu().numpy() * chi_eff).astype(np.float32)).to(device)
    B, T = s_ref.shape[0], s_ref.shape[1]
    last_alpha = torch.full((B,), 1.0, device=device); last_beta = torch.full((B,), 0.5, device=device)
    last_gamma = torch.full((B,), 0.4, device=device); last_vp = torch.full((B,), 30.0, device=device); last_xs = torch.zeros((B, 1), device=device)
    v_curr, m_curr = v_init, m_init
    v_list, m_list = [], []
    for t in range(T):
        xs_t = xs_ref[:, t, :].clone()
        xs_t = torch.where(torch.isnan(xs_t) | torch.isinf(xs_t), last_xs, xs_t)
        last_xs = xs_t.clone()
        scalars_t = scalars[:, t, :].clone()
        for idx, last in enumerate([last_alpha, last_beta, last_gamma, last_vp]):
            col = scalars_t[:, idx:idx+1]
            bad = torch.isnan(col) | torch.isinf(col)
            if idx == 3:
                col = torch.where(bad | (col <= 0), torch.tensor(30.0, device=device), col)
            else:
                col = torch.where(bad, last.unsqueeze(1), col)
            scalars_t[:, idx:idx+1] = col
            ok = ~bad.squeeze()
            if ok.any():
                if idx == 0: last_alpha[ok] = col[ok, 0]
                elif idx == 1: last_beta[ok] = col[ok, 0]
                elif idx == 2: last_gamma[ok] = col[ok, 0]
                else: last_vp[ok] = col[ok, 0]
        state = torch.cat([v_curr, m_curr], dim=1)
        next_s = physics_engine(xs_t, state, scalars_t)
        v_curr, m_curr = next_s[:, 0:1], next_s[:, 1:2]
        v_list.append(v_curr); m_list.append(m_curr)
    return torch.stack(v_list, 1), torch.stack(m_list, 1)

def train_honest_system(data_dir='training_data', seq_len=TARGET_SEQ_LEN, min_vmax_kts=MIN_VMAX_KTS, precalc_dir='precalc_data', truncate_after_landfall=False,
                       train_years=None, val_years=None, test_years=None, start_from_vmax_kts=START_FROM_VMAX_KTS,
                       storm_include_list=None, storm_exclude_list=None, epochs=150):
    """Train on real data. 时间起点 vmax>=start_from_vmax_kts；可人工挑选 storm_include/exclude。epochs 用于快速验证。"""
    print("Loading train/val/test (start vmax>=%d kts, fixed seq_len)..." % start_from_vmax_kts)
    train_datasets, val_datasets, test_datasets = load_train_val_test_data(
        data_dir=data_dir, seq_len=seq_len, min_vmax_kts=min_vmax_kts, precalc_dir=precalc_dir, truncate_after_landfall=truncate_after_landfall,
        train_years=train_years, val_years=val_years, test_years=test_years, start_from_vmax_kts=start_from_vmax_kts,
        storm_include_list=storm_include_list, storm_exclude_list=storm_exclude_list)
    if len(train_datasets) == 0:
        print("No training data."); return
    spatial_stats = compute_spatial_stats(train_datasets)
    print("Spatial stats (mean,std) 3d:", [(f"{m:.2f}", f"{s:.2f}") for m, s in spatial_stats['3d']], "2d:", [(f"{m:.2f}", f"{s:.2f}") for m, s in spatial_stats['2d']])
    model = Honest_FAST_Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 降低 lr 提高稳定性，避免梯度爆炸
    loss_s_fn = DynamicWeightedMSE(2.0, 5.0)
    loss_chi_fn = DynamicWeightedMSE(50.0, 0.05)
    warmup = min(50, epochs // 3)
    stage1_epochs = min(80, max(10, epochs // 2))  # 阶段1: ref chi/s 引导; 阶段2: 仅 obz loss
    loss_hist, val_loss_hist = [], []
    best_val_loss, best_epoch = float('inf'), -1
    print("Loss terms: V=MSE(pred_v_max,v_obs) S=pred_s vs s_ref Chi=pred_chi vs chi_ref m_rate=m change penalty smooth=xs temporal smooth")

    for epoch in range(epochs):
        if epoch < warmup:
            tf_ratio = 1.0 - (epoch / warmup) * 0.5
            w_ref, w_phys = 5.0, 0.5
            phase = "WarmUp"
        else:
            progress = (epoch - warmup) / max(1, epochs - warmup)
            tf_ratio = 0.5 * (1.0 - progress)
            w_ref, w_phys = 5.0, 50.0
            phase = "Physics"
        use_ref_loss = epoch < stage1_epochs  # 阶段1用ref chi/s，阶段2只用obz
        model.train()
        optimizer.zero_grad()
        batch = create_batch(train_datasets, batch_size=min(8, len(train_datasets)), device=device, spatial_stats=spatial_stats)
        scalars = batch['scalars'].clone()
        for b in range(scalars.shape[0]):
            vp = scalars[b, :, 3].cpu().numpy()
            from run_fast_reference import _median_filter_1d
            scalars[b, :, 3] = torch.from_numpy(_median_filter_1d(vp, size=3)).float().to(device)
        v_gt = batch['v_gt']; v_init = v_gt[:, 0:1, :]
        s_ref, chi_ref, xs_ref = batch['s_ref'], batch['chi_ref'], batch['xs_ref']
        precomp = None
        if all(k in batch for k in ['env_wnds', 'utran', 'vtran', 'lats']):
            from run_fast_physics_torch import precompute_run_fast_init
            precomp = precompute_run_fast_init(
                scalars.cpu().numpy(), batch['v_gt'].cpu().numpy(),
                batch['env_wnds'].cpu().numpy(), batch['utran'].cpu().numpy(),
                batch['vtran'].cpu().numpy(), batch['lats'].cpu().numpy(),
                batch['s_ref'].cpu().numpy(), batch['lons'].cpu().numpy() if 'lons' in batch else None
            )
        preds = model(batch['spatial_3d'], batch['spatial_2d'], scalars, v_init, xs_teacher=xs_ref, teacher_forcing_ratio=tf_ratio,
                     precomp=precomp, env_wnds=batch.get('env_wnds'), utran=batch.get('utran'), vtran=batch.get('vtran'), lats=batch.get('lats'))
        valid_v = batch['valid_mask'].unsqueeze(-1)
        if preds.get('v_max') is not None:
            loss_v = ((preds['v_max'] - v_gt).pow(2) * valid_v.float()).sum() / valid_v.float().sum().clamp(min=1) if valid_v.any() else torch.tensor(0.0, device=device)
        elif all(k in batch for k in ['env_wnds', 'utran', 'vtran', 'lats']):
            pred_vmax = axi_to_max_wind_torch(preds['v'], preds['s'], batch['env_wnds'], batch['utran'], batch['vtran'], batch['lats'])
            loss_v = ((pred_vmax - v_gt).pow(2) * valid_v.float()).sum() / valid_v.float().sum().clamp(min=1) if valid_v.any() else torch.tensor(0.0, device=device)
        else:
            loss_v = ((preds['v'] - v_gt).pow(2) * valid_v.float()).sum() / valid_v.float().sum().clamp(min=1) if valid_v.any() else torch.tensor(0.0, device=device)
        s_mask = valid_v & ~torch.isnan(s_ref) & (s_ref > 0)
        loss_s = loss_s_fn(preds['s'][s_mask], s_ref[s_mask]) if s_mask.sum() > 0 else torch.tensor(0.0, device=device)
        chi_mask = valid_v & ~torch.isnan(chi_ref) & (chi_ref > 0)
        loss_chi = loss_chi_fn(preds['chi'][chi_mask], chi_ref[chi_mask]) * 100.0 if chi_mask.sum() > 0 else torch.tensor(0.0, device=device)
        dm = preds['m'][:, 1:] - preds['m'][:, :-1]
        valid_dm = (torch.arange(dm.shape[1], device=device).unsqueeze(0) + 1 < batch['seq_len'].unsqueeze(1)).unsqueeze(-1)
        loss_m_rate = (F.relu(torch.abs(dm) - 0.05) * valid_dm.float()).sum() / valid_dm.float().sum().clamp(min=1) * 100.0 if valid_dm.any() else torch.tensor(0.0, device=device)
        xs_diff = (preds['xs'][:, 1:] - preds['xs'][:, :-1]).pow(2)
        loss_smooth = (xs_diff * valid_dm.float()).sum() / valid_dm.float().sum().clamp(min=1) if valid_dm.any() else torch.tensor(0.0, device=device)
        # NaN 安全：任一 loss 为 NaN/inf 时用 0 替代，避免 total 污染
        def _safe(t, name=''):
            ok = bool(torch.isfinite(t).all().item()) if t.numel() > 0 else True
            if not ok and name == 'V':
                pv = preds.get('v_max')
                if pv is None and all(k in batch for k in ['env_wnds', 'utran', 'vtran', 'lats']):
                    pv = axi_to_max_wind_torch(preds['v'], preds['s'], batch['env_wnds'], batch['utran'], batch['vtran'], batch['lats'])
                elif pv is None:
                    pv = preds['v']
                nan_pct = (torch.isnan(pv).sum().item() / max(1, pv.numel()) * 100) if pv is not None else 0
                print(f"  [Ep {epoch+1}] V loss NaN/inf (pred_v_max NaN%: {nan_pct:.1f}) -> set 0")
            return t if ok else torch.tensor(0.0, device=t.device)
        loss_v = _safe(loss_v, 'V')
        loss_s, loss_chi = _safe(loss_s), _safe(loss_chi)
        loss_m_rate, loss_smooth = _safe(loss_m_rate), _safe(loss_smooth)
        if use_ref_loss:
            total = w_phys * loss_v + w_ref * (loss_s + loss_chi) + loss_m_rate + 0.1 * loss_smooth
        else:
            total = w_phys * loss_v + loss_m_rate + 0.1 * loss_smooth
        total = _safe(total)
        # 数值稳定：total 仍异常时跳过
        if not (torch.isfinite(total).all() and total.item() < 1e6):
            print(f"Ep {epoch+1:03d} SKIP (total={total.item()})")
            continue
        total.backward()
        # 梯度含 NaN 时跳过更新
        grad_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not grad_ok:
            optimizer.zero_grad()
            print(f"Ep {epoch+1:03d} SKIP (grad NaN/inf)")
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更强裁剪防梯度爆炸
        optimizer.step()
        loss_hist.append(total.item())
        if val_datasets and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vb = create_batch(val_datasets, len(val_datasets), device, shuffle=False, spatial_stats=spatial_stats)
                scalars_v = vb['scalars'].clone()
                for b in range(scalars_v.shape[0]):
                    from run_fast_reference import _median_filter_1d
                    vp = scalars_v[b, :, 3].cpu().numpy()
                    scalars_v[b, :, 3] = torch.from_numpy(_median_filter_1d(vp, size=3)).float().to(device)
                from run_fast_physics_torch import precompute_run_fast_init
                precomp_v = precompute_run_fast_init(scalars_v.cpu().numpy(), vb['v_gt'].cpu().numpy(), vb['env_wnds'].cpu().numpy(), vb['utran'].cpu().numpy(), vb['vtran'].cpu().numpy(), vb['lats'].cpu().numpy(), vb['s_ref'].cpu().numpy(), vb['lons'].cpu().numpy() if 'lons' in vb else None) if all(k in vb for k in ['env_wnds', 'utran', 'vtran', 'lats']) else None
                vp = model(vb['spatial_3d'], vb['spatial_2d'], scalars_v, vb['v_gt'][:, 0:1, :], xs_teacher=None, teacher_forcing_ratio=0.0, precomp=precomp_v, env_wnds=vb.get('env_wnds'), utran=vb.get('utran'), vtran=vb.get('vtran'), lats=vb.get('lats'))
                valid_vb = vb['valid_mask'].unsqueeze(-1)
                pred_vmax_v = vp['v_max'] if vp.get('v_max') is not None else (axi_to_max_wind_torch(vp['v'], vp['s'], vb['env_wnds'], vb['utran'], vb['vtran'], vb['lats']) if all(k in vb for k in ['env_wnds', 'utran', 'vtran', 'lats']) else vp['v'])
                val_loss_v = ((pred_vmax_v - vb['v_gt']).pow(2) * valid_vb.float()).sum() / valid_vb.float().sum().clamp(min=1) if valid_vb.any() else torch.tensor(0.0, device=device)
                val_loss_hist.append(val_loss_v.item())
                if val_loss_v.item() < best_val_loss and torch.isfinite(val_loss_v):
                    best_val_loss = val_loss_v.item()
                    best_epoch = epoch + 1
                    if not any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                        torch.save(model.state_dict(), 'honest_fast_model_best.pth')
            model.train()
        # 每 epoch 打印 loss 分项。loss_v: pred_v_max vs 观测; loss_s: pred_s vs s_ref; loss_chi: pred_chi vs chi_ref; loss_m_rate: m 变化率惩罚; loss_smooth: xs 时间平滑
        vinfo = f" Val:{val_loss_hist[-1]:.4f}" if val_loss_hist and (epoch + 1) % 10 == 0 else ""
        print(f"Ep {epoch+1:03d} [{phase}] TF:{tf_ratio:.2f} Loss:{total.item():.2f} | V:{loss_v.item():.4f}(v_max vs obs) S:{loss_s.item():.4f}(s vs ref) Chi:{loss_chi.item():.4f}(chi vs ref) m_rate:{loss_m_rate.item():.2f} smooth:{loss_smooth.item():.4f}{vinfo}")

    # 仅在权重无 NaN/inf 时保存，避免损坏的 checkpoint 导致 Pred S/Chi/XS 不显示
    has_nan = any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters())
    if not has_nan:
        torch.save(model.state_dict(), 'honest_fast_model_2017.pth')
        print("Model saved.")
    else:
        print("WARNING: Model has NaN/inf weights, NOT saved. Pred S/Chi/XS will not display. Check training (lr, gradient clip).")
    # 若有 best checkpoint，加载用于 eval/plots（best 验证损失最低）
    best_path = Path(__file__).resolve().parent / 'honest_fast_model_best.pth'
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best checkpoint (ep {best_epoch}, val_loss={best_val_loss:.4f}) for eval/plots.")

    # Test eval + plots
    if test_datasets:
        model.eval()
        with torch.no_grad():
            test_batch = create_batch(test_datasets, len(test_datasets), device, shuffle=False, spatial_stats=spatial_stats)
            test_scalars = test_batch['scalars'].clone()
            for b in range(test_scalars.shape[0]):
                from run_fast_reference import _median_filter_1d
                vp = test_scalars[b, :, 3].cpu().numpy()
                test_scalars[b, :, 3] = torch.from_numpy(_median_filter_1d(vp, size=3)).float().to(device)
            from run_fast_physics_torch import precompute_run_fast_init
            precomp_t = precompute_run_fast_init(test_scalars.cpu().numpy(), test_batch['v_gt'].cpu().numpy(), test_batch['env_wnds'].cpu().numpy(), test_batch['utran'].cpu().numpy(), test_batch['vtran'].cpu().numpy(), test_batch['lats'].cpu().numpy(), test_batch['s_ref'].cpu().numpy(), test_batch['lons'].cpu().numpy() if 'lons' in test_batch else None) if all(k in test_batch for k in ['env_wnds', 'utran', 'vtran', 'lats']) else None
            test_preds = model(test_batch['spatial_3d'], test_batch['spatial_2d'], test_scalars, test_batch['v_gt'][:, 0:1, :], xs_teacher=None, teacher_forcing_ratio=0.0, precomp=precomp_t, env_wnds=test_batch.get('env_wnds'), utran=test_batch.get('utran'), vtran=test_batch.get('vtran'), lats=test_batch.get('lats'))
            valid_t = test_batch['valid_mask'].unsqueeze(-1)
            pred_v_for_loss = test_preds['v_max'] if test_preds.get('v_max') is not None else (axi_to_max_wind_torch(test_preds['v'], test_preds['s'], test_batch['env_wnds'], test_batch['utran'], test_batch['vtran'], test_batch['lats']) if all(k in test_batch for k in ['env_wnds', 'utran', 'vtran', 'lats']) else test_preds['v'])
            test_loss_v = ((pred_v_for_loss - test_batch['v_gt']).pow(2) * valid_t.float()).sum() / valid_t.float().sum().clamp(min=1) if valid_t.any() else torch.tensor(0.0, device=device)
            print(f"Test Loss V: {test_loss_v.item():.4f}")
            scalars_ref = test_batch['scalars'].clone()
            lons_b = test_batch.get('lons')
            v_max_ref, m_ref_np = _run_fast_reference_batch(
                test_batch['s_ref'], test_batch['chi_ref'], test_batch['xs_ref'], scalars_ref,
                test_batch['v_gt'], test_batch['env_wnds'], test_batch['utran'], test_batch['vtran'],
                test_batch['lats'], lons_b)
            if v_max_ref is not None:
                v_ref_physics = torch.from_numpy(v_max_ref.astype(np.float32)).unsqueeze(-1).to(device)
                m_ref_physics = torch.from_numpy(m_ref_np.astype(np.float32)).unsqueeze(-1).to(device)
            else:
                physics_engine = FAST_Physics_Engine(dt=1.0).to(device)
                scalars_test = test_batch['scalars'].clone(); scalars_test[:, :, 3] *= 1.2
                v_ref_physics, m_ref_physics = _run_physics_ref(test_batch['s_ref'], test_batch['chi_ref'], scalars_test.to(device), test_batch['v_gt'][:, 0:1, 0], test_preds['m'][:, 0:1, 0], physics_engine, device)
                if all(k in test_batch for k in ['env_wnds', 'utran', 'vtran', 'lats']):
                    v_ref_physics = axi_to_max_wind_torch(v_ref_physics, test_batch['s_ref'], test_batch['env_wnds'], test_batch['utran'], test_batch['vtran'], test_batch['lats'])
            vm = test_batch['valid_mask'].unsqueeze(-1) & ~torch.isnan(v_ref_physics) & ~torch.isnan(test_batch['v_gt'])
            if vm.sum() > 0:
                print(f"Test Loss (FAST ref): {nn.MSELoss()(v_ref_physics[vm], test_batch['v_gt'][vm]).item():.4f}")
            test_pred_vmax = test_preds['v_max'] if test_preds.get('v_max') is not None else (axi_to_max_wind_torch(test_preds['v'], test_preds['s'], test_batch['env_wnds'], test_batch['utran'], test_batch['vtran'], test_batch['lats']) if all(k in test_batch for k in ['env_wnds', 'utran', 'vtran', 'lats']) else test_preds['v'])
            os.makedirs('eval_test', exist_ok=True)
            for i in range(test_batch['v_gt'].shape[0]):
                plot_comparison_one_sample(
                    test_batch['v_gt'][i,:,0].cpu().numpy(), test_batch['s_ref'][i,:,0].cpu().numpy(), test_batch['chi_ref'][i,:,0].cpu().numpy(),
                    test_pred_vmax[i,:,0].cpu().numpy(), test_preds['s'][i,:,0].cpu().numpy(), test_preds['chi'][i,:,0].cpu().numpy(),
                    test_preds['xs'][i,:,0].cpu().numpy(), test_preds['m'][i,:,0].cpu().numpy(),
                    v_ref_physics[i,:,0].cpu().numpy(), m_ref_physics[i,:,0].cpu().numpy(),
                    test_batch['seq_len'][i].item(), test_datasets[i].get('hurricane', f'test_{i}'),
                    os.path.join('eval_test', _safe_filename(test_datasets[i].get('hurricane', f'test_{i}')) + '.png'),
                    test_batch['m0_ref'][i,0].item() if 'm0_ref' in test_batch else None,
                    test_batch['times'][i] if 'times' in test_batch else None)
            print(f"Saved {test_batch['v_gt'].shape[0]} plots to eval_test/")

    # Val plots
    if val_datasets:
        model.eval()
        with torch.no_grad():
            val_batch = create_batch(val_datasets, len(val_datasets), device, shuffle=False, spatial_stats=spatial_stats)
            val_scalars = val_batch['scalars'].clone()
            for b in range(val_scalars.shape[0]):
                from run_fast_reference import _median_filter_1d
                vp = val_scalars[b, :, 3].cpu().numpy()
                val_scalars[b, :, 3] = torch.from_numpy(_median_filter_1d(vp, size=3)).float().to(device)
            from run_fast_physics_torch import precompute_run_fast_init
            precomp_val = precompute_run_fast_init(val_scalars.cpu().numpy(), val_batch['v_gt'].cpu().numpy(), val_batch['env_wnds'].cpu().numpy(), val_batch['utran'].cpu().numpy(), val_batch['vtran'].cpu().numpy(), val_batch['lats'].cpu().numpy(), val_batch['s_ref'].cpu().numpy(), val_batch['lons'].cpu().numpy() if 'lons' in val_batch else None) if all(k in val_batch for k in ['env_wnds', 'utran', 'vtran', 'lats']) else None
            val_preds = model(val_batch['spatial_3d'], val_batch['spatial_2d'], val_scalars, val_batch['v_gt'][:, 0:1, :], xs_teacher=None, teacher_forcing_ratio=0.0, precomp=precomp_val, env_wnds=val_batch.get('env_wnds'), utran=val_batch.get('utran'), vtran=val_batch.get('vtran'), lats=val_batch.get('lats'))
            v_max_val, m_ref_val_np = _run_fast_reference_batch(
                val_batch['s_ref'], val_batch['chi_ref'], val_batch['xs_ref'], val_batch['scalars'].clone(),
                val_batch['v_gt'], val_batch['env_wnds'], val_batch['utran'], val_batch['vtran'],
                val_batch['lats'], val_batch.get('lons'))
            if v_max_val is not None:
                v_ref_val = torch.from_numpy(v_max_val.astype(np.float32)).unsqueeze(-1).to(device)
                m_ref_val = torch.from_numpy(m_ref_val_np.astype(np.float32)).unsqueeze(-1).to(device)
            else:
                v_ref_val, m_ref_val = _run_physics_ref(val_batch['s_ref'], val_batch['chi_ref'], val_batch['scalars'].clone().to(device), val_batch['v_gt'][:, 0:1, 0], val_preds['m'][:, 0:1, 0], FAST_Physics_Engine(dt=1.0).to(device), device)
                if all(k in val_batch for k in ['env_wnds', 'utran', 'vtran', 'lats']):
                    v_ref_val = axi_to_max_wind_torch(v_ref_val, val_batch['s_ref'], val_batch['env_wnds'], val_batch['utran'], val_batch['vtran'], val_batch['lats'])
            val_pred_vmax = val_preds['v_max'] if val_preds.get('v_max') is not None else (axi_to_max_wind_torch(val_preds['v'], val_preds['s'], val_batch['env_wnds'], val_batch['utran'], val_batch['vtran'], val_batch['lats']) if all(k in val_batch for k in ['env_wnds', 'utran', 'vtran', 'lats']) else val_preds['v'])
            os.makedirs('eval_val', exist_ok=True)
            for i in range(val_batch['v_gt'].shape[0]):
                plot_comparison_one_sample(
                    val_batch['v_gt'][i,:,0].cpu().numpy(), val_batch['s_ref'][i,:,0].cpu().numpy(), val_batch['chi_ref'][i,:,0].cpu().numpy(),
                    val_pred_vmax[i,:,0].cpu().numpy(), val_preds['s'][i,:,0].cpu().numpy(), val_preds['chi'][i,:,0].cpu().numpy(),
                    val_preds['xs'][i,:,0].cpu().numpy(), val_preds['m'][i,:,0].cpu().numpy(),
                    v_ref_val[i,:,0].cpu().numpy(), m_ref_val[i,:,0].cpu().numpy(),
                    val_batch['seq_len'][i].item(), val_datasets[i].get('hurricane', f'val_{i}'),
                    os.path.join('eval_val', _safe_filename(val_datasets[i].get('hurricane', f'val_{i}')) + '.png'),
                    val_batch['m0_ref'][i,0].item() if 'm0_ref' in val_batch else None,
                    val_batch['times'][i] if 'times' in val_batch else None)
        print(f"Saved {len(val_datasets)} plots to eval_val/")

if __name__ == "__main__":
    import argparse
    import sys
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='training_data')
    p.add_argument('--seq_len', type=int, default=TARGET_SEQ_LEN)
    p.add_argument('--min_vmax_kts', type=float, default=MIN_VMAX_KTS)
    p.add_argument('--precalc_dir', default='precalc_data')
    p.add_argument('--train_years', type=str, default='', help='Comma-separated years, e.g. 2003,2004,...,2021')
    p.add_argument('--val_years', type=str, default='', help='Comma-separated years, e.g. 2022')
    p.add_argument('--test_years', type=str, default='', help='Comma-separated years, e.g. 2023')
    p.add_argument('--start_from_vmax_kts', type=float, default=START_FROM_VMAX_KTS, help='Time start from vmax>= this (kts)')
    p.add_argument('--storm_include', type=str, default='', help='File: storm IDs to include (year/storm_dir per line)')
    p.add_argument('--storm_exclude', type=str, default='storm_exclude.txt', help='File: storm IDs to exclude (vmax<40kts or FAST MAE>15kts)')
    p.add_argument('--epochs', type=int, default=150, help='Training epochs (use fewer for quick verification)')
    p.add_argument('--list_storms', action='store_true', help='List all storm IDs and exit (for building storm_include/exclude)')
    args = p.parse_args()
    def _parse_years(s):
        if not s.strip():
            return None
        return [int(y.strip()) for y in s.split(',') if y.strip()]
    train_years = _parse_years(args.train_years)
    val_years = _parse_years(args.val_years)
    test_years = _parse_years(args.test_years)
    storm_include = args.storm_include.strip() or None
    storm_exclude = args.storm_exclude.strip() or None
    if args.list_storms:
        storms = build_storms_by_year(args.data_dir, seq_len=args.seq_len, min_vmax_kts=args.min_vmax_kts,
                                      start_from_vmax_kts=args.start_from_vmax_kts)
        for y in sorted(storms.keys()):
            for name, ds in storms[y]:
                print(f"{y}/{name}")
        sys.exit(0)
    train_honest_system(data_dir=args.data_dir, seq_len=args.seq_len, min_vmax_kts=args.min_vmax_kts, precalc_dir=args.precalc_dir,
                        train_years=train_years, val_years=val_years, test_years=test_years,
                        start_from_vmax_kts=args.start_from_vmax_kts,
                        storm_include_list=storm_include, storm_exclude_list=storm_exclude, epochs=args.epochs)
