#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 prepare_complete_training_data 生成的新 pkl 运行 FAST 物理模型（参考 chi、s、scalars），
得到 v_fast，再经 axi_to_max_wind 得 v_max；与观测 v_gt 直接对比。无任何 vp 缩放系数。

参考 tropical_cyclone_risk: coupled_fast.py, geo.py。仅 NA 区域。
- 引入 Ck/Cd 读取：从 geo.read_drag 获取空间变化的 Cd，陆上 Cd 更大以表示更快衰减
- 参数范围与参考代码一致，不人为设定 low/high
- 陆上处理：vp=0（prepare 已设），Cd 由 geo 插值得到

数据格式（与 prepare_complete_training_data 输出一致）：
  scalars [1,T,4]= (alpha, beta, gamma, vp), chi_ref/s_ref [1,T,1], v_init/v_gt [1,T,1],
  env_wnds [1,T,4], utran/vtran [1,T,1], lats [T], lons [T]。单位均为 m/s，输出与 CSV/图统一为 kts。

用法:
  python run_fast_reference.py [--data_dir training_data] [--year_start 2021] [--year_end 2021] [--limit N]
  默认测试 2021 年；全量可加 --year_start 2003 --year_end 2023
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 尝试导入 tropical_cyclone_risk 的 geo、namelist（用于 Cd、h_bl）
# Optional external checkout of github.com/linjonathan/tropical_cyclone_risk;
# when absent, a constant Cd fallback is used (see _load_cd_and_hbl).
_TC_RISK_PATH = os.environ.get(
    'TC_RISK_PATH',
    str(Path(__file__).resolve().parent.parent.parent / 'tropical_cyclone_risk'))
if _TC_RISK_PATH not in sys.path:
    sys.path.insert(0, _TC_RISK_PATH)

_F_Cd = None
_H_BL = 1400.0
_Cd_CONST = 1.2e-3
_BASIN = None


def _load_cd_and_hbl():
    """从 geo.read_drag 读取 Cd 插值器，从 namelist 读取 h_bl。仅 NA 区域。"""
    global _F_Cd, _H_BL, _Cd_CONST, _BASIN
    if _F_Cd is not None and _BASIN == 'NA':
        return _F_Cd, _H_BL
    try:
        from util.basins import TC_Basin
        import namelist
        basin = TC_Basin('NA')
        _BASIN = 'NA'
        from intensity import geo
        _F_Cd = geo.read_drag(basin)
        h_bl_dict = getattr(namelist, 'atm_bl_depth', None)
        _H_BL = float(h_bl_dict['NA']) if h_bl_dict and 'NA' in h_bl_dict else 1400.0
        _Cd_CONST = getattr(namelist, 'Cd', 1.2e-3)
        return _F_Cd, _H_BL
    except Exception:
        _F_Cd = None
        _H_BL = 1400.0
        _Cd_CONST = 1.2e-3
        return None, _H_BL


def _get_cd_at(lon, lat, f_Cd):
    """在 (lon, lat) 处插值 Cd。与 coupled_fast._get_current_Cd 一致。"""
    if f_Cd is None:
        return _Cd_CONST
    lon = float(lon) if np.isfinite(lon) else 0.0
    lat = float(lat) if np.isfinite(lat) else 0.0
    lon = lon + 360 if lon < 0 else lon
    try:
        return float(f_Cd.ev(lon, lat).flatten()[0])
    except Exception:
        return _Cd_CONST


STEP_SIZE = 1.0 / 4.0  # dt=1, sub_steps=4
# 单位转换：内部 m/s，保存与可视化统一为 kts
MS_TO_KNOTS = 1.94384  # m/s -> knots
# xs_ref (chi*s) 的 NaN 替代值：用 1e-5 而非 0，确保陆上 m 可衰减（对齐 tropical_cyclone_risk chi_month floor）
XS_NAN_FALLBACK = 1e-5
# 路线 B：乘法放大模拟 90th 百分位（2020 论文思想），替代 Lin 2023 对数公式
# 干空气分布近似对数正态，90th 约为平均值的 2~3 倍
CHI_MULTIPLIER = 5
CHI_D_ATLANTIC = 4.0  # 物理上限


def _median_filter_1d(arr, size=3):
    """1D 中位数滑窗（3h），抑制 vp 在 1h 内的骤升骤降；边界用边缘值填充"""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr
    padded = np.pad(arr, 1, mode='edge')
    return np.array([np.median(padded[i:i + size]) for i in range(n)], dtype=np.float64)


def _chi_calibrated_multiply(chi_val, chi_multiplier=CHI_MULTIPLIER, chi_max=CHI_D_ATLANTIC):
    """乘法放大模拟寻找最干空气：χ_effective = χ_val * multiplier，clip 到物理上限"""
    chi_val = np.asarray(chi_val, dtype=np.float64)
    chi_val = np.nan_to_num(chi_val, nan=1e-10)
    chi_val = np.maximum(chi_val, 1e-10)
    chi_effective = chi_val * chi_multiplier
    return np.clip(chi_effective, 0.0, chi_max)


def _safe_scalar(x, default):
    """仅处理 NaN，不人为设定参数范围（与 coupled_fast 一致）。"""
    if np.isnan(x):
        return float(default)
    return float(x)


def _coeff_from_cd(cd, h_bl):
    """COEFF = 0.5 * Cd / h_bl * 3600（m/s per hour），与 coupled_fast dvdt/dmdt 一致。"""
    return 0.5 * float(cd) / float(h_bl) * 3600.0


def _physics_rhs_v(V, m, alpha, beta, gamma, vp, coeff):
    """FAST 方程 dV/dt 的物理右端项。与 coupled_fast._dvdt 一致。"""
    vp = _safe_scalar(vp, 0.0)
    alpha = _safe_scalar(alpha, 1.0)
    beta = _safe_scalar(beta, 0.57)
    gamma = _safe_scalar(gamma, 0.43)
    m3 = m ** 3
    v2 = V ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        rhs = coeff * (alpha * beta * vp**2 * m3 - (1.0 - gamma * m3) * v2)
    return rhs if np.isfinite(rhs) else 0.0


def _physics_rhs_m(V, m, xs, coeff):
    """FAST 方程 dm/dt 的物理右端项。与 coupled_fast._dmdt 一致。"""
    xs = _safe_scalar(xs, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        rhs = coeff * ((1.0 - m) * V - xs * m)
    return rhs if np.isfinite(rhs) else 0.0


def fast_step_numpy(xs, V, m, alpha, beta, gamma, vp, coeff, dV_extra=0.0):
    """单步 FAST ODE（Heun），标量输入/输出。dV_extra 为附加到 dV/dt 的强迫项（如 F*decay）。"""
    vp = _safe_scalar(vp, 0.0)
    alpha = _safe_scalar(alpha, 1.0)
    beta = _safe_scalar(beta, 0.57)
    gamma = _safe_scalar(gamma, 0.43)
    xs = _safe_scalar(xs, 0.0)
    m3 = m ** 3
    v2 = V ** 2
    dV = coeff * (alpha * beta * vp**2 * m3 - (1.0 - gamma * m3) * v2) + dV_extra
    dm = coeff * ((1.0 - m) * V - xs * m)
    V_mid = max(0.0, min(200.0, V + dV * STEP_SIZE))
    m_mid = max(0.0, min(1.0, m + dm * STEP_SIZE))
    m3_mid = m_mid ** 3
    v2_mid = V_mid ** 2
    dV2 = coeff * (alpha * beta * vp**2 * m3_mid - (1.0 - gamma * m3_mid) * v2_mid) + dV_extra
    dm2 = coeff * ((1.0 - m_mid) * V_mid - xs * m_mid)
    V_next = max(0.0, min(200.0, V + 0.5 * (dV + dV2) * STEP_SIZE))
    m_next = max(0.0, min(1.0, m + 0.5 * (dm + dm2) * STEP_SIZE))
    return V_next, m_next


def calculate_m0_from_fast_numpy(v, dv_dt, alpha, beta, gamma, vp, coeff):
    """从 FAST 方程反算 m0（纯 NumPy）。m^3 = (dV/dt/coeff + V^2) / (alpha*beta*vp^2 + gamma*V^2)。"""
    vp = _safe_scalar(vp, 0.0)
    alpha = _safe_scalar(alpha, 1.0)
    beta = _safe_scalar(beta, 0.57)
    gamma = _safe_scalar(gamma, 0.43)
    numerator = dv_dt / (coeff + 1e-12) + v**2
    denominator = alpha * beta * vp**2 + gamma * v**2
    m3 = numerator / (denominator + 1e-8)
    m3 = np.clip(m3, 0.0, None)
    m0 = np.power(m3, 1.0/3.0)
    return np.clip(m0, 0.01, 1.0)


def find_all_dataset_pkl(data_dir, year_start=2003, year_end=2023):
    """列出 data_dir 下 year_start~year_end 各年、各飓风目录中的 *_dataset.pkl。"""
    pkl_list = []
    for y in range(year_start, year_end + 1):
        year_dir = Path(data_dir) / str(y)
        if not year_dir.is_dir():
            continue
        for storm_dir in sorted(year_dir.iterdir()):
            if not storm_dir.is_dir():
                continue
            for f in storm_dir.glob('*_dataset.pkl'):
                pkl_list.append(f)
    return sorted(pkl_list)


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


def axi_to_max_wind_numpy(tc_v, s_ref, env_wnds, utran, vtran, lats):
    """
    V_max = f(V_fast, S, env_wnds, u_T, lat). 与 SciML_Fast.axi_to_max_wind_torch 一致。
    G = min(1, 0.8 + 0.35*(1+tanh((|lat|-35)/10)))，用 s_ref 调制 shear 贡献。
    Inputs: tc_v [T], s_ref [T], env_wnds [T,4], utran [T], vtran [T], lats [T]
    """
    G = np.minimum(1.0, 0.8 + 0.35 * (1.0 + np.tanh((np.abs(lats) - 35.0) / 10.0)))
    u_shr = env_wnds[:, 0] - env_wnds[:, 2]
    v_shr = env_wnds[:, 1] - env_wnds[:, 3]
    shear_mag = np.sqrt(u_shr**2 + v_shr**2 + 1e-12)
    has_env = ~(np.isnan(u_shr) | np.isnan(v_shr) | (shear_mag < 1e-6))
    u_dir = np.where(has_env, u_shr / shear_mag, 0.0)
    v_dir = np.where(has_env, v_shr / shear_mag, 0.0)
    s_safe = np.nan_to_num(s_ref, nan=0.0)
    shear_coeff = 0.1 * s_safe * tc_v / 15.0
    U_inc = G * utran + shear_coeff * u_dir
    V_inc = G * vtran + shear_coeff * v_dir
    mag_inc = np.sqrt(U_inc**2 + V_inc**2 + 1e-12)
    mag_fac = np.minimum(1.0, (tc_v * 0.5) / mag_inc)
    theta_opt = np.arctan2(-U_inc, V_inc)
    ug = tc_v * (-np.sin(theta_opt)) + U_inc * mag_fac
    vg = tc_v * np.cos(theta_opt) + V_inc * mag_fac
    return np.sqrt(ug**2 + vg**2 + 1e-12)


# 45 kts 起报阈值（m/s）
VMAX_START_KTS = 45.0
VMAX_START_MS = VMAX_START_KTS / MS_TO_KNOTS
INIT_HOURS = 48
T0_DECAY_HOURS = 24.0


def run_fast_forward(scalars, xs_ref, v_init, v_gt=None, lons=None, lats=None):
    """
    用参考 xs (chi*s) 和 scalars（vp 列，不乘任何系数）做 FAST 前向积分，得到 v_fast 序列。
    纯 NumPy 实现，与 coupled_fast 一致。lons/lats 可选，用于 Cd 插值。

    scalars: [1, T, 4]  (alpha, beta, gamma, vp)，vp 为 m/s，直接使用
    xs_ref:  [1, T, 1]  (chi * s，参考值)
    v_init:  [1, T, 1]  取 v_init[0,0,0] 作为初始 V
    v_gt:    [1, T, 1]  观测风速，用于反算初始 m0（可选）
    lons, lats: [T] 可选，用于 Cd 插值。

    Returns:
        v_fast: [T] numpy (m/s); vp_used: [T] (vp 原值，m/s)
    """
    T = scalars.shape[1]
    scalars_np = np.array(scalars[0, :, :], dtype=np.float64)
    vp_used = scalars_np[:, 3].copy()
    alpha = scalars_np[:, 0]
    beta = scalars_np[:, 1]
    gamma = scalars_np[:, 2]
    xs_np = np.maximum(np.nan_to_num(np.array(xs_ref[0, :, 0], dtype=np.float64), nan=XS_NAN_FALLBACK), XS_NAN_FALLBACK)

    # Cd/coeff：有 lons/lats 则从 geo 读取，否则用常数
    if lons is not None and lats is not None:
        la = np.asarray(lats).reshape(-1)[:T]
        lo = np.asarray(lons).reshape(-1)[:T]
        f_Cd, h_bl = _load_cd_and_hbl()
        coeff_arr = np.array([_coeff_from_cd(_get_cd_at(lo[t], la[t], f_Cd), h_bl) for t in range(T)], dtype=np.float64)
    else:
        coeff_arr = np.full(T, _coeff_from_cd(_Cd_CONST, _H_BL), dtype=np.float64)

    v0 = float(v_init[0, 0, 0])
    if np.isnan(v0) or v0 <= 0:
        v0 = 5.0
    V = np.float64(v0)

    # 用 v_gt 前两步的 dv_dt 反算 m0
    if v_gt is not None and T >= 2:
        v_gt_np = np.array(v_gt[0, :, 0], dtype=np.float64)
        if not (np.isnan(v_gt_np[0]) or np.isnan(v_gt_np[1])):
            dv_dt = v_gt_np[1] - v_gt_np[0]
            m0 = calculate_m0_from_fast_numpy(
                v0, dv_dt, alpha[0], beta[0], gamma[0], vp_used[0], coeff_arr[0]
            )
            m0 = float(np.clip(m0, 0.01, 1.0))
        else:
            m0 = 0.5
    else:
        m0 = 0.5
    m = np.float64(m0)

    v_fast_list = [v0]
    for t in range(T - 1):
        for _ in range(4):  # sub_steps=4 per hour
            V, m = fast_step_numpy(
                xs_np[t], V, m,
                alpha[t], beta[t], gamma[t], vp_used[t], coeff_arr[t]
            )
        v_fast_list.append(float(V))
    v_fast = np.array(v_fast_list, dtype=np.float32)
    return v_fast, np.array(vp_used, dtype=np.float32)


def run_fast_with_init(scalars, xs_ref, v_gt, env_wnds, utran, vtran, lats, s_ref, lons=None):
    """
    按论文与文档实现：计算并存储强迫项 F，预报期用 F_init_end×衰减 加入 dV/dt。

    1. 起报点 t_start：v_obz 首次达到 45 kts
    2. 初始化期 [t_init_start, t_start)：起报点前 48h；若不足则延后起报
    3. 初始化期：Vtarget=V_axisym(obs)；F(t)=观测加速度-Physics_RHS(Vtarget,m)；m 用 Vtarget 自然 spin-up
    4. 记录 F_init_end = 初始化结束时刻的 F
    5. 预报期：dV/dt = Physics_RHS + F_init_end × exp(-2*(t_lead/24)^2)

    v_gt, env_wnds, utran, vtran, lats, lons, s_ref 需为 [T] 或兼容形状。
    lons: 用于从 geo.read_drag 插值 Cd（陆上 Cd 更大，与 coupled_fast 一致）；无则用常数 Cd。
    Returns: v_fast [T], v_max [T] (m/s)
    """
    T = scalars.shape[1]
    scalars_np = np.array(scalars[0, :, :], dtype=np.float64)
    vp_used = scalars_np[:, 3]
    alpha, beta, gamma = scalars_np[:, 0], scalars_np[:, 1], scalars_np[:, 2]
    xs_np = np.maximum(np.nan_to_num(np.array(xs_ref[0, :, 0], dtype=np.float64), nan=XS_NAN_FALLBACK), XS_NAN_FALLBACK)
    v_obz = np.array(v_gt[0, :, 0], dtype=np.float64)

    # 读取 Cd、h_bl（与 coupled_fast/geo 一致，仅 NA）
    la = np.zeros(T)
    if lats is not None:
        a = np.asarray(lats).reshape(-1)
        la[: min(len(a), T)] = a[:T]
    lo = np.zeros(T)
    if lons is not None:
        a = np.asarray(lons).reshape(-1)
        lo[: min(len(a), T)] = a[:T]
    f_Cd, h_bl = _load_cd_and_hbl()
    coeff_arr = np.zeros(T, dtype=np.float64)
    for t in range(T):
        cd_t = _get_cd_at(lo[t], la[t], f_Cd)
        coeff_arr[t] = _coeff_from_cd(cd_t, h_bl)

    ew = np.full((T, 4), np.nan)
    if env_wnds is not None:
        arr = np.asarray(env_wnds)
        if arr.ndim == 3:
            ew = np.array(arr[0, :T, :], dtype=np.float64)
        else:
            ew[: min(arr.shape[0], T), :] = arr[:T, :]
    ut = np.zeros(T)
    if utran is not None:
        a = np.asarray(utran).reshape(-1)
        ut[: min(len(a), T)] = a[:T]
    vt = np.zeros(T)
    if vtran is not None:
        a = np.asarray(vtran).reshape(-1)
        vt[: min(len(a), T)] = a[:T]
    s_r = np.nan_to_num(np.asarray(s_ref).reshape(T, -1)[:, 0], nan=0.0) if s_ref is not None else np.zeros(T)

    # 起报点 = 首次达到 45 kts；拟合期 = 起报点前 48h；若前段不足 48h 则延后起报
    t_40 = None
    for i in range(T):
        if not np.isnan(v_obz[i]) and v_obz[i] >= VMAX_START_MS:
            t_40 = i
            break
    if t_40 is None:
        t_40 = 0
    t_start = t_40
    t_init_start = t_start - INIT_HOURS
    if t_init_start < 0:
        t_start = INIT_HOURS
        t_init_start = 0
    if t_start > T:
        t_start = T
        t_init_start = max(0, T - INIT_HOURS)

    # Vtarget = V_axisym(obs)（已扣除移动+切变分量的轴对称等效）
    Vtarget = np.full(T, np.nan)
    for i in range(T):
        if np.isnan(v_obz[i]) or v_obz[i] <= 0:
            continue
        env_i = ew[i] if ew.ndim == 2 else ew[i, :]
        Vtarget[i] = _invert_vmax_to_V_axisym_np(
            v_obz[i], s_r[i], env_i, ut[i], vt[i], la[i]
        )

    v_fast = np.full(T, np.nan)
    m_series = np.full(T, np.nan)
    if t_init_start >= T:
        v_max = np.full(T, np.nan)
        return v_fast, v_max, m_series

    v0 = float(Vtarget[t_init_start]) if not np.isnan(Vtarget[t_init_start]) else 5.0
    if v0 <= 0:
        v0 = 5.0
    V = np.float64(v0)
    dv_dt = 0.0
    if t_init_start + 1 < T and not np.isnan(Vtarget[t_init_start]) and not np.isnan(Vtarget[t_init_start + 1]):
        dv_dt = Vtarget[t_init_start + 1] - Vtarget[t_init_start]
    coeff_init = coeff_arr[t_init_start]
    m0 = calculate_m0_from_fast_numpy(v0, dv_dt, alpha[t_init_start], beta[t_init_start], gamma[t_init_start], vp_used[t_init_start], coeff_init)
    m = np.float64(np.clip(m0, 0.01, 1.0))

    F_init_end = 0.0
    F_history = []

    for t in range(t_init_start, T):
        coeff_t = coeff_arr[t]
        if t < t_start:
            # 初始化期：F(t)=观测加速度-Physics_RHS(Vtarget,m)；V 跟踪 Vtarget；m 用 Vtarget 自然 spin-up
            Vtar_t = float(Vtarget[t]) if not np.isnan(Vtarget[t]) else V
            Vtar_next = float(Vtarget[t + 1]) if t + 1 < T and not np.isnan(Vtarget[t + 1]) else Vtar_t
            observed_accel = Vtar_next - Vtar_t
            physics_rhs = _physics_rhs_v(Vtar_t, m, alpha[t], beta[t], gamma[t], vp_used[t], coeff_t)
            F_t = observed_accel - physics_rhs
            F_history.append(F_t)
            if t == min(t_start, T) - 1:
                window = min(12, len(F_history))
                F_init_end = float(np.mean(F_history[-window:]))
            V = np.float64(Vtar_next)
            for _ in range(4):
                dm = _physics_rhs_m(Vtar_next, m, xs_np[t], coeff_t)
                m = np.float64(np.clip(m + dm * STEP_SIZE, 0.01, 1.0))
            m_series[t] = float(m)
        else:
            # 预报期：dV/dt = Physics_RHS + F_init_end × exp(-2*(t_lead/24)^2)
            lead_h = t - t_start
            decay = np.exp(-2.0 * (lead_h / T0_DECAY_HOURS) ** 2)
            dV_extra = F_init_end * decay
            for _ in range(4):
                V, m = fast_step_numpy(xs_np[t], V, m, alpha[t], beta[t], gamma[t], vp_used[t], coeff_t, dV_extra=dV_extra)
        v_fast[t] = float(V)
        m_series[t] = float(m)

    v_max = axi_to_max_wind_numpy(v_fast, s_r, ew, ut, vt, la)
    return v_fast, v_max, m_series


def process_one_pkl(pkl_path, save_csv=True, save_plot=True):
    """
    处理一个 *_dataset.pkl（新 prepare 格式）：运行 FAST（45kts 起报 + 起报前 48h 初始化 + 预报期强迫衰减），
    axi_to_max_wind 得 v_max，与观测直接对比。不使用任何 vp 缩放系数。
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    scalars = data['scalars']       # [1, T, 4] (alpha, beta, gamma, vp) m/s
    chi_ref = data['chi_ref']       # [1, T, 1]
    s_ref = data['s_ref']           # [1, T, 1]
    v_init = data['v_init']         # [1, T, 1]
    v_gt = data['v_gt']             # [1, T, 1] 观测 v_obz m/s

    T = scalars.shape[1]
    # vp 中位数滑窗 3h，抑制 1h 内骤升骤降
    scalars = np.array(scalars, dtype=np.float64, copy=True)
    vp_raw = scalars[0, :, 3]
    scalars[0, :, 3] = _median_filter_1d(vp_raw, size=3)

    chi_cal = _chi_calibrated_multiply(chi_ref)
    xs_ref = np.maximum(np.nan_to_num(chi_cal * s_ref, nan=XS_NAN_FALLBACK), XS_NAN_FALLBACK)

    env_wnds = data.get('env_wnds')
    utran = data.get('utran')
    vtran = data.get('vtran')
    lats = data.get('lats')
    lons = data.get('lons')
    times = data.get('times')
    if times is not None:
        times = pd.to_datetime(np.asarray(times).ravel()[:T])
    else:
        times = pd.date_range(start='2000-01-01', periods=T, freq='h')

    # 带 48h 初始化与 45kts 起报的 FAST（引入 Cd 读取，陆上处理与 coupled_fast 一致）
    v_fast_ms, v_max_ms, m_series = run_fast_with_init(
        scalars, xs_ref, v_gt, env_wnds, utran, vtran, lats, s_ref, lons=lons
    )
    v_obz_ms = np.array(v_gt[0, :, 0], dtype=np.float32)
    vp_ms = np.array(scalars[0, :, 3], dtype=np.float32)

    # 统一转换为 kts
    vp_kts = vp_ms * MS_TO_KNOTS
    v_obz_kts = v_obz_ms * MS_TO_KNOTS
    v_fast_kts = v_fast_ms * MS_TO_KNOTS
    v_max_kts = v_max_ms * MS_TO_KNOTS

    storm_name = data.get('hurricane', pkl_path.stem.replace('_dataset', ''))
    out_dir = pkl_path.parent

    # 保存 CSV（含 v_max_kts）
    if save_csv:
        df = pd.DataFrame({
            'step': np.arange(T),
            'time': times,
            'vp_kts': vp_kts,
            'v_obz_kts': v_obz_kts,
            'v_fast_kts': v_fast_kts,
            'v_max_kts': v_max_kts,
            'm': m_series,
        })
        csv_path = out_dir / 'fast_reference.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

    # 绘图：单图，黑色线，横坐标标注具体日期、2天一格，纵坐标固定 0-200 kts
    if save_plot:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(times, v_max_kts, label='v_max (FAST)', color='blue', linewidth=1.5, alpha=0.9)
        ax.plot(times, v_obz_kts, label='v_obz (obs)', color='black', linewidth=1.5, alpha=0.9)
        ax.plot(times, vp_kts, label='vp (potential intensity)', color='red', linewidth=1.5, alpha=0.9)
        ax.set_xlabel('Date')
        ax.set_ylabel('Intensity (knots)')
        ax.set_ylim(0, 200)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        fig.autofmt_xdate()
        ax.set_title(storm_name)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, color='grey')
        fig.tight_layout()
        plot_path = out_dir / 'fast_reference.png'
        fig.savefig(plot_path, dpi=150, facecolor='white')
        plt.close(fig)
        print(f"  Saved {plot_path}")

    # 验证指标：v_max vs v_obz
    valid = ~np.isnan(v_obz_ms) & ~np.isnan(v_max_ms) & (v_obz_ms > 0)
    if valid.sum() > 0:
        rmse = np.sqrt(np.mean((v_max_ms[valid] - v_obz_ms[valid])**2)) * MS_TO_KNOTS
        bias = np.mean(v_max_ms[valid] - v_obz_ms[valid]) * MS_TO_KNOTS
        mae_kts = np.mean(np.abs(v_max_ms[valid] - v_obz_ms[valid])) * MS_TO_KNOTS
        vm, vo = v_max_ms[valid], v_obz_ms[valid]
        if valid.sum() > 1 and np.std(vm) > 1e-12 and np.std(vo) > 1e-12:
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.corrcoef(vm, vo)[0, 1]
            corr = float(corr) if not np.isnan(corr) else 0.0
        else:
            corr = 0.0
    else:
        rmse, bias, corr = np.nan, np.nan, 0.0
        mae_kts = np.nan
    max_v_obz_kts = float(np.nanmax(v_obz_kts)) if np.any(np.isfinite(v_obz_kts)) else 0.0
    return storm_name, vp_kts, v_obz_kts, v_fast_kts, v_max_kts, {
        'rmse_kts': rmse, 'bias_kts': bias, 'mae_kts': mae_kts, 'corr': corr, 'max_v_obz_kts': max_v_obz_kts
    }


def main():
    parser = argparse.ArgumentParser(description='Run FAST with reference chi/s for each storm in training_data')
    parser.add_argument('--data_dir', type=str, default='training_data', help='Root of training data (e.g. training_data)')
    parser.add_argument('--year_start', type=int, default=2003, help='First year (default 2021 for testing)')
    parser.add_argument('--year_end', type=int, default=2024, help='Last year inclusive (default 2021 for testing)')
    parser.add_argument('--limit', type=int, default=None, help='Max number of storms to process (default all)')
    parser.add_argument('--no_csv', action='store_true', help='Do not save CSV')
    parser.add_argument('--no_plot', action='store_true', help='Do not save plot')
    parser.add_argument('--export_exclude', type=str, default='', help='Export storm IDs to exclude: vmax<40kts or FAST MAE>15kts. Output txt for SciML_Fast --storm_exclude')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: data_dir not found: {data_dir}")
        return

    pkl_list = find_all_dataset_pkl(args.data_dir, args.year_start, args.year_end)
    if args.limit is not None:
        pkl_list = pkl_list[: args.limit]
    print(f"Found {len(pkl_list)} dataset pkl files (years {args.year_start}-{args.year_end})")

    all_rmse, all_bias, all_corr = [], [], []
    exclude_candidates = []  # (storm_id, reason, max_v_obz, mae)
    for pkl_path in pkl_list:
        try:
            out = process_one_pkl(
                pkl_path,
                save_csv=not args.no_csv,
                save_plot=not args.no_plot,
            )
            metrics = out[-1]
            if not np.isnan(metrics.get('rmse_kts', np.nan)):
                all_rmse.append(metrics['rmse_kts'])
                all_bias.append(metrics['bias_kts'])
                all_corr.append(metrics['corr'])
            year = pkl_path.parent.parent.name
            storm_dir = pkl_path.parent.name
            storm_id = f"{year}/{storm_dir}"
            max_v = metrics.get('max_v_obz_kts', 0.0)
            mae = metrics.get('mae_kts', np.nan)
            reasons = []
            if max_v < 40:
                reasons.append(f"vmax<40kts({max_v:.1f})")
            if not np.isnan(mae) and mae > 20:
                reasons.append(f"MAE>{mae:.1f}kts")
            if reasons:
                exclude_candidates.append((storm_id, '; '.join(reasons), max_v, mae if not np.isnan(mae) else -1))
        except Exception as e:
            print(f"  Error processing {pkl_path}: {e}")
            import traceback
            traceback.print_exc()
    if all_rmse:
        print(f"\nValidation (v_max vs obs): n={len(all_rmse)}, RMSE={np.mean(all_rmse):.2f} kts, Bias={np.mean(all_bias):.2f} kts, Corr={np.mean(all_corr):.3f}")
    if args.export_exclude and exclude_candidates:
        out_path = Path(args.export_exclude)
        with open(out_path, 'w') as f:
            f.write("# Exclude: vmax<40kts or FAST MAE>15kts. For SciML_Fast --storm_exclude\n")
            for sid, reason, _mv, _mae in exclude_candidates:
                f.write(f"{sid}  # {reason}\n")
        print(f"\nExported {len(exclude_candidates)} storms to {out_path} (vmax<40kts or MAE>15kts)")
    elif args.export_exclude:
        print(f"\nNo storms to exclude (all passed vmax>=40kts and MAE<=15kts)")
    print("Done.")


if __name__ == '__main__':
    main()