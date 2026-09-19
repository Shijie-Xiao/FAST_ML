#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch 实现的 FAST 物理。

训练模式 (training_mode=True): 纯 PyTorch 前向积分，无 .item()/NumPy，梯度可回传到 pred_chi/pred_s。
推理模式 (training_mode=False): 完整 run_fast_with_init（48h 初始化 + F 衰减），用于评估与 run_fast_reference 一致。
"""
import numpy as np
import torch

# 与 run_fast_reference 完全一致
CHI_MULTIPLIER = 5
CHI_D_ATLANTIC = 4.0
XS_NAN_FALLBACK = 1e-5
STEP_SIZE = 1.0 / 4.0
VMAX_START_KTS = 45.0
VMAX_START_MS = VMAX_START_KTS / 1.94384
INIT_HOURS = 48
T0_DECAY_HOURS = 24.0
COEFF_CONST = 0.5 * (1.2e-3 / 1400.0) * 3600.0

# 训练时 loss 截断：前 N 步不参与 loss，让 LSTM 消化
LOSS_TRIM_STEPS = 12


def chi_calibrated_torch(chi, chi_multiplier=CHI_MULTIPLIER, chi_max=CHI_D_ATLANTIC):
    """χ_effective = χ * multiplier，clip 到物理上限。"""
    chi = torch.nan_to_num(chi, nan=1e-10)
    chi = torch.clamp(chi, min=1e-10)
    chi_eff = chi * chi_multiplier
    return torch.clamp(chi_eff, 0.0, chi_max)


def _get_coeff_arr_numpy(lons, lats, T):
    """从 run_fast_reference 获取每时刻的 coeff。仅用于 precompute（推理模式）。"""
    try:
        import sys
        from pathlib import Path
        _ode = Path(__file__).resolve().parent
        if str(_ode) not in sys.path:
            sys.path.insert(0, str(_ode))
        from run_fast_reference import _load_cd_and_hbl, _get_cd_at, _coeff_from_cd
        f_Cd, h_bl = _load_cd_and_hbl()
        lo = np.asarray(lons).reshape(-1)[:T]
        la = np.asarray(lats).reshape(-1)[:T]
        return np.array([_coeff_from_cd(_get_cd_at(lo[t], la[t], f_Cd), h_bl) for t in range(T)], dtype=np.float64)
    except Exception:
        return np.full(T, COEFF_CONST, dtype=np.float64)


def calculate_m0_from_fast_torch(v, dv_dt, alpha, beta, gamma, vp, coeff):
    """
    纯 PyTorch 从 FAST 方程反算 m0，梯度可追踪。
    m^3 = (dV/dt/coeff + V^2) / (alpha*beta*vp^2 + gamma*V^2)
    """
    # m0 反算仅在起报点（海上）调用；vp<=0 或 NaN 保底 0，使分母退化为 gamma*V^2，m0 仍可估算
    vp = torch.where(torch.isnan(vp) | (vp < 0), torch.zeros_like(vp), vp)
    alpha = torch.nan_to_num(alpha, nan=1.0)
    beta = torch.nan_to_num(beta, nan=0.57)
    gamma = torch.nan_to_num(gamma, nan=0.43)
    num = dv_dt / (coeff + 1e-12) + v ** 2
    den = alpha * beta * vp ** 2 + gamma * v ** 2 + 1e-8
    m3 = torch.clamp(num / den, min=0.0)
    m0 = torch.pow(m3 + 1e-12, 1.0 / 3.0)
    return torch.clamp(m0, min=0.01, max=1.0)


def _fast_step_torch(xs, V, m, alpha, beta, gamma, vp, coeff, dV_extra=None, is_ocean=None):
    """
    单步 FAST ODE (Heun)。含 is_ocean 掩码：陆上切断水汽输入 (1-m)*V。
    is_ocean: [B] 或 [B,1]，vp>1 为海上。None 时默认全为海上。
    """
    if dV_extra is None:
        dV_extra = torch.zeros_like(V)
    # is_ocean 在调用侧已经计算好并传入；vp 陆地保底 0（非30），使 vp^2=0 令台风在陆地上无法增强
    vp_safe = torch.where(torch.isnan(vp) | (vp < 0), torch.zeros_like(vp), vp)
    alpha = torch.nan_to_num(alpha, nan=1.0)
    beta = torch.nan_to_num(beta, nan=0.57)
    gamma = torch.nan_to_num(gamma, nan=0.43)
    xs = torch.nan_to_num(xs, nan=0.0)
    xs = torch.clamp(xs, min=XS_NAN_FALLBACK)
    if is_ocean is None:
        is_ocean = torch.ones_like(V)
    m3 = m ** 3
    v2 = V ** 2
    dV = coeff * (alpha * beta * vp_safe ** 2 * m3 - (1.0 - gamma * m3) * v2) + dV_extra
    dm = coeff * (is_ocean * (1.0 - m) * V - xs * m)
    V_mid = torch.clamp(V + dV * STEP_SIZE, 0.0, 200.0)
    m_mid = torch.clamp(m + dm * STEP_SIZE, 0.0, 1.0)
    m3_mid = m_mid ** 3
    v2_mid = V_mid ** 2
    dV2 = coeff * (alpha * beta * vp_safe ** 2 * m3_mid - (1.0 - gamma * m3_mid) * v2_mid) + dV_extra
    dm2 = coeff * (is_ocean * (1.0 - m_mid) * V_mid - xs * m_mid)
    V_next = torch.clamp(V + 0.5 * (dV + dV2) * STEP_SIZE, 0.0, 200.0)
    m_next = torch.clamp(m + 0.5 * (dm + dm2) * STEP_SIZE, 0.0, 1.0)
    return V_next, m_next


def run_fast_physics_torch_training(pred_chi, pred_s, x_scalars, v_init, env_wnds, utran, vtran, lats,
                                     v_gt=None, f_init_end=None, m_init=None, t_start=None,
                                     v_axisym_at_tstart=None, pred_chi_is_eff=False):
    """
    训练模式：纯 PyTorch 前向积分，梯度完整回传到 pred_chi/pred_s。

    当 v_axisym_at_tstart 不为 None 时（完整对齐模式）：
      - 逐样本从 t_start[b] 开始自由积分，完全对齐 run_fast_reference 逻辑：
        * V_init[b] = v_axisym_at_tstart[b]（precompute 48h nudging 后的轴对称风速）
        * m_init[b] = m_at_tstart（由外部 m_init 传入，48h spin-up 结束时的 m）
        * dV_extra(t) = F_init_end × exp(-2×((t-t_start)/24h)²)，仅预报期施加
        * v_fast[b, t<t_start]=0（被 _vm(use_tstart=True) 排除出 loss）
        * v_fast[b, t>=t_start] 对应物理时刻 t，与 v_gt[b,t] 对齐
    当 v_axisym_at_tstart 为 None 时（兼容模式）：
      - 从 t=0 开始批量积分
    """
    B, T = pred_chi.shape[0], pred_chi.shape[1]
    dev = pred_chi.device

    chi_eff = pred_chi.squeeze(-1) if pred_chi_is_eff else chi_calibrated_torch(pred_chi.squeeze(-1))
    pred_xs = torch.clamp(chi_eff * pred_s.squeeze(-1), min=XS_NAN_FALLBACK)

    alpha = torch.nan_to_num(x_scalars[:, :, 0], nan=1.0)
    beta = torch.nan_to_num(x_scalars[:, :, 1], nan=0.57)
    gamma = torch.nan_to_num(x_scalars[:, :, 2], nan=0.43)
    vp = x_scalars[:, :, 3]
    # 隐患3修复：vp>0.1 为海洋。陆地被预处理硬设为 0；高纬冷水区 vp 可能 <1，仍为海洋
    is_ocean = ((~torch.isnan(vp)) & (vp > 0.1)).float()
    # 陆地保底 0（vp^2=0，令 dV<0，台风在陆上无法增强）；NaN 或负值均归零
    vp = torch.where(torch.isnan(vp) | (vp < 0), torch.zeros_like(vp), vp)

    coeff = torch.full((B, T), COEFF_CONST, device=dev, dtype=pred_chi.dtype)
    ts_arr = t_start.long().to(dev).flatten()[:B] if t_start is not None else torch.zeros(B, dtype=torch.long, device=dev)
    fie = f_init_end.float().to(dev).flatten()[:B] if f_init_end is not None else None

    # ── 完整对齐模式：逐样本从 t_start 开始积分 ──────────────────────────────
    # 完全对应 run_fast_reference 的预报期逻辑：
    #   V_init = V_axisym 在 t_start（48h nudging 结束），m_init = 48h spin-up 末尾 m
    #   v_fast[b, t<t_start] = 0（被 _vm(use_tstart=True) 排出 loss）
    #   v_fast[b, t>=t_start] 对应物理时刻 t，与 v_gt[b,t] 时间对齐
    if v_axisym_at_tstart is not None:
        V_init = v_axisym_at_tstart.squeeze().flatten()[:B].clone().to(dev)
        V_init = torch.where(torch.isnan(V_init) | (V_init <= 0), torch.full_like(V_init, 5.0), V_init)
        if m_init is not None:
            M_init = m_init.squeeze().flatten()[:B].clone().to(dev)
        else:
            M_init = torch.full((B,), 0.5, device=dev, dtype=pred_chi.dtype)
        M_init = torch.clamp(M_init, 0.01, 1.0)

        # 逐样本积分（t_start 因样本不同）
        vf_samples, ms_samples = [], []
        for b in range(B):
            ts_b = int(ts_arr[b].item())
            V_b = V_init[b]
            m_b = M_init[b]
            fie_b = fie[b] if fie is not None else torch.zeros(1, device=dev, dtype=pred_chi.dtype).squeeze()

            zeros = torch.zeros(ts_b, device=dev, dtype=pred_chi.dtype)
            vf_b  = list(zeros) + [V_b]   # t=0..ts_b-1: 0（被 mask），t=ts_b: V_axisym
            ms_b  = list(zeros) + [m_b]

            for t in range(ts_b, T - 1):
                xs_t       = pred_xs[b, t]
                coeff_t    = coeff[b, t]
                alpha_t    = alpha[b, t]
                beta_t     = beta[b, t]
                gamma_t    = gamma[b, t]
                vp_t       = vp[b, t]
                is_ocean_t = is_ocean[b, t]
                lead_h     = float(t - ts_b)
                decay      = torch.exp(torch.tensor(-2.0 * (lead_h / T0_DECAY_HOURS) ** 2,
                                                    device=dev, dtype=pred_chi.dtype))
                dV_extra   = fie_b * decay
                for _ in range(4):
                    V_b, m_b = _fast_step_torch(
                        xs_t.unsqueeze(0), V_b.unsqueeze(0), m_b.unsqueeze(0),
                        alpha_t.unsqueeze(0), beta_t.unsqueeze(0), gamma_t.unsqueeze(0),
                        vp_t.unsqueeze(0), coeff_t.unsqueeze(0),
                        dV_extra=dV_extra.unsqueeze(0),
                        is_ocean=is_ocean_t.unsqueeze(0))
                    V_b, m_b = V_b.squeeze(0), m_b.squeeze(0)
                vf_b.append(V_b)
                ms_b.append(m_b)

            vf_samples.append(torch.stack(vf_b))
            ms_samples.append(torch.stack(ms_b))

        v_fast   = torch.stack(vf_samples, dim=0)   # [B, T]
        m_series = torch.stack(ms_samples, dim=0)   # [B, T]
        v_max    = _axi_to_max_wind_torch(v_fast.unsqueeze(-1), pred_s, env_wnds, utran, vtran, lats)
        return v_max, v_fast.unsqueeze(-1), m_series.unsqueeze(-1)

    # ── 兼容模式：从 t=0 批量积分（原逻辑）─────────────────────────────────────
    V = v_init.squeeze().flatten()[:B].clone()
    V = torch.where(torch.isnan(V) | (V <= 0), torch.full_like(V, 5.0), V)
    if m_init is not None:
        m = m_init.squeeze().flatten()[:B].clone().to(dev)
        m = torch.clamp(m, 0.01, 1.0)
    elif v_gt is not None and T >= 2:
        v0 = v_gt[:, 0, 0]; dv_dt = torch.nan_to_num(v_gt[:, 1, 0] - v_gt[:, 0, 0], nan=0.0)
        m = torch.clamp(calculate_m0_from_fast_torch(
            v0, dv_dt, alpha[:, 0], beta[:, 0], gamma[:, 0], vp[:, 0], coeff[:, 0]), 0.01, 1.0)
    else:
        m = torch.full((B,), 0.5, device=dev, dtype=pred_chi.dtype)

    def _vec(x): return x.squeeze().flatten()[:B]
    v_fast_list = [_vec(V.clone())]; m_list = [_vec(m.clone())]

    for t in range(T - 1):
        xs_t = _vec(pred_xs[:, t]); coeff_t = _vec(coeff[:, t])
        alpha_t = _vec(alpha[:, t]); beta_t = _vec(beta[:, t]); gamma_t = _vec(gamma[:, t])
        vp_t = _vec(vp[:, t]); is_ocean_t = _vec(is_ocean[:, t])
        if fie is not None:
            is_forecast = (t >= ts_arr).float()
            lead_h = (t - ts_arr).float().clamp(min=0.0)
            dV_extra = fie * torch.exp(-2.0 * (lead_h / T0_DECAY_HOURS) ** 2) * is_forecast
        else:
            dV_extra = torch.zeros(B, device=dev, dtype=pred_chi.dtype)
        for _ in range(4):
            V, m = _fast_step_torch(xs_t, _vec(V), _vec(m), alpha_t, beta_t, gamma_t,
                                    vp_t, coeff_t, dV_extra=dV_extra, is_ocean=is_ocean_t)
        v_fast_list.append(_vec(V.clone())); m_list.append(_vec(m.clone()))

    v_fast = torch.stack(v_fast_list, dim=1); m_series = torch.stack(m_list, dim=1)
    v_max  = _axi_to_max_wind_torch(v_fast.unsqueeze(-1), pred_s, env_wnds, utran, vtran, lats)
    return v_max, v_fast.unsqueeze(-1), m_series.unsqueeze(-1)


def precompute_run_fast_init(scalars_np, v_gt_np, env_wnds_np, utran_np, vtran_np, lats_np, s_ref_np, lons_np=None, chi_ref_np=None):
    """
    预计算 run_fast_with_init 所需结构。与 run_fast_reference.run_fast_with_init 完全一致：
    - vp 中位数滑窗 3h
    - xs_ref = _chi_calibrated_multiply(chi_ref) * s_ref
    - Cd/coeff 从 geo 插值（有 lons/lats 时）
    """
    try:
        from run_fast_reference import (
            _invert_vmax_to_V_axisym_np, calculate_m0_from_fast_numpy,
            _physics_rhs_v, _physics_rhs_m, _chi_calibrated_multiply, _median_filter_1d,
            XS_NAN_FALLBACK, VMAX_START_MS, INIT_HOURS
        )
    except ImportError:
        return None

    B, T = scalars_np.shape[0], scalars_np.shape[1]
    v_obz = np.array(v_gt_np[:, :, 0], dtype=np.float64)
    scalars = np.array(scalars_np[:, :, :], dtype=np.float64)
    # vp 中位数滑窗 3h，与 run_fast_reference process_one_pkl 一致
    for i in range(B):
        scalars[i, :, 3] = _median_filter_1d(scalars[i, :, 3], size=3)
    vp_used = scalars[:, :, 3]
    alpha, beta, gamma = scalars[:, :, 0], scalars[:, :, 1], scalars[:, :, 2]
    s_r = np.nan_to_num(np.array(s_ref_np).reshape(B, T, -1)[:, :, 0], nan=0.0) if s_ref_np is not None else np.zeros((B, T))
    # xs_ref = chi_calibrated(chi_ref) * s_ref，与 run_fast_reference 完全一致
    if chi_ref_np is not None:
        chi_cal = _chi_calibrated_multiply(np.array(chi_ref_np).reshape(B, T, -1)[:, :, 0])
        xs_np = np.maximum(np.nan_to_num(chi_cal * s_r, nan=XS_NAN_FALLBACK), XS_NAN_FALLBACK)
    else:
        xs_np = np.full((B, T), XS_NAN_FALLBACK, dtype=np.float64)

    ew = np.full((B, T, 4), np.nan)
    if env_wnds_np is not None:
        arr = np.asarray(env_wnds_np)
        if arr.ndim == 3:
            ew = arr[:, :T, :].astype(np.float64)
    ut = np.zeros((B, T))
    if utran_np is not None:
        a = np.asarray(utran_np).reshape(B, -1)
        ut[:, :min(a.shape[1], T)] = a[:, :T]
    vt = np.zeros((B, T))
    if vtran_np is not None:
        a = np.asarray(vtran_np).reshape(B, -1)
        vt[:, :min(a.shape[1], T)] = a[:, :T]
    la = np.zeros((B, T))
    if lats_np is not None:
        a = np.asarray(lats_np).reshape(B, -1)
        la[:, :min(a.shape[1], T)] = a[:, :T]
    lo = np.zeros((B, T))
    if lons_np is not None:
        a = np.asarray(lons_np).reshape(B, -1)
        lo[:, :min(a.shape[1], T)] = a[:, :T]

    coeff_arr = np.zeros((B, T), dtype=np.float64)
    for i in range(B):
        coeff_arr[i] = _get_coeff_arr_numpy(lo[i], la[i], T)

    Vtarget = np.full((B, T), np.nan, dtype=np.float64)
    for i in range(B):
        for t in range(T):
            if np.isnan(v_obz[i, t]) or v_obz[i, t] <= 0:
                continue
            env_i = ew[i, t]
            Vtarget[i, t] = _invert_vmax_to_V_axisym_np(
                v_obz[i, t], s_r[i, t], env_i, ut[i, t], vt[i, t], la[i, t]
            )

    t_start = np.zeros(B, dtype=np.int64)
    t_init_start = np.zeros(B, dtype=np.int64)
    for i in range(B):
        t_40 = 0
        for t in range(T):
            if not np.isnan(v_obz[i, t]) and v_obz[i, t] >= VMAX_START_MS:
                t_40 = t
                break
        ts = t_40
        ti = max(0, ts - INIT_HOURS)
        if ti < 0:
            ts = INIT_HOURS
            ti = 0
        if ts > T:
            ts = T
            ti = max(0, T - INIT_HOURS)
        t_start[i] = ts
        t_init_start[i] = ti

    F_init_end = np.zeros(B, dtype=np.float64)
    m0_init = np.full(B, 0.5, dtype=np.float64)     # m 在 t_init_start 时刻（48h spin-up 开始前）
    m_at_tstart = np.full(B, 0.5, dtype=np.float64) # m 在 t_start 时刻（48h spin-up 结束后，预报起点）
    for i in range(B):
        ti = t_init_start[i]
        ts = t_start[i]
        if ti >= T:
            continue
        v0 = float(Vtarget[i, ti]) if not np.isnan(Vtarget[i, ti]) else 5.0
        if v0 <= 0:
            v0 = 5.0
        dv_dt = 0.0
        if ti + 1 < T and not np.isnan(Vtarget[i, ti]) and not np.isnan(Vtarget[i, ti + 1]):
            dv_dt = Vtarget[i, ti + 1] - Vtarget[i, ti]
        m0 = calculate_m0_from_fast_numpy(
            v0, dv_dt, alpha[i, ti], beta[i, ti], gamma[i, ti], vp_used[i, ti], coeff_arr[i, ti]
        )
        m = np.clip(m0, 0.01, 1.0)
        m0_init[i] = float(m)   # 记录 48h init 开始时的 m（历史保留）
        V = np.float64(v0)
        F_history = []
        for t in range(ti, min(ts, T)):
            Vtar_t = float(Vtarget[i, t]) if not np.isnan(Vtarget[i, t]) else V
            Vtar_next = float(Vtarget[i, t + 1]) if t + 1 < T and not np.isnan(Vtarget[i, t + 1]) else Vtar_t
            observed_accel = Vtar_next - Vtar_t
            physics_rhs = _physics_rhs_v(Vtar_t, m, alpha[i, t], beta[i, t], gamma[i, t], vp_used[i, t], coeff_arr[i, t])
            F_t = observed_accel - physics_rhs
            F_history.append(F_t)
            if t == min(ts, T) - 1:
                window = min(12, len(F_history))
                F_init_end[i] = float(np.mean(F_history[-window:]))
            V = np.float64(Vtar_next)
            for _ in range(4):
                dm = _physics_rhs_m(V, m, xs_np[i, t], coeff_arr[i, t])
                m = np.clip(m + dm * STEP_SIZE, 0.01, 1.0)
        # 记录 48h spin-up 结束时的 m（预报起点的 m，更物理准确）
        m_at_tstart[i] = float(m)

    # v_axisym_at_tstart: V_axisym 在预报起点的值（用于训练 ODE 从 t_start 开始的正确初始 V）
    v_axisym_at_tstart = np.zeros(B, dtype=np.float64)
    for i in range(B):
        ts = t_start[i]
        v_ts = float(Vtarget[i, ts]) if ts < T and not np.isnan(Vtarget[i, ts]) else 5.0
        v_axisym_at_tstart[i] = max(v_ts, 1.0)

    return {
        'Vtarget': Vtarget, 't_start': t_start, 't_init_start': t_init_start,
        'coeff_arr': coeff_arr, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'vp_used': vp_used,
        'F_init_end': F_init_end, 'm0_init': m0_init, 'm_at_tstart': m_at_tstart,
        'v_axisym_at_tstart': v_axisym_at_tstart,
        'ew': ew, 'ut': ut, 'vt': vt, 'la': la, 's_r': s_r,
    }


def run_fast_physics_torch_inference(pred_chi, pred_s, x_scalars, precomp, env_wnds, utran, vtran, lats,
                                     pred_chi_is_eff=False):
    """
    推理模式：完整 run_fast_with_init（48h 初始化 + F 衰减），用于评估。
    与 run_fast_reference 一致。此路径不参与训练，可含 NumPy。
    """
    if precomp is None:
        B, T = pred_chi.shape[0], pred_chi.shape[1]
        dev = pred_chi.device
        return torch.zeros(B, T, 1, device=dev), torch.zeros(B, T, 1, device=dev), torch.zeros(B, T, 1, device=dev)

    B, T = pred_chi.shape[0], pred_chi.shape[1]
    dev = pred_chi.device

    chi_eff = pred_chi.squeeze(-1) if pred_chi_is_eff else chi_calibrated_torch(pred_chi.squeeze(-1))
    pred_xs = torch.clamp(chi_eff * pred_s.squeeze(-1), min=XS_NAN_FALLBACK)

    Vtarget = torch.from_numpy(precomp['Vtarget']).float().to(dev)
    t_start = precomp['t_start']
    t_init_start = precomp['t_init_start']
    coeff_arr = torch.from_numpy(precomp['coeff_arr']).float().to(dev)
    alpha = torch.from_numpy(precomp['alpha']).float().to(dev)
    beta = torch.from_numpy(precomp['beta']).float().to(dev)
    gamma = torch.from_numpy(precomp['gamma']).float().to(dev)
    vp_used = torch.from_numpy(precomp['vp_used']).float().to(dev)
    F_init_end = torch.from_numpy(precomp['F_init_end']).float().to(dev)

    v_fast = torch.full((B, T), float('nan'), device=dev)
    m_series = torch.full((B, T), float('nan'), device=dev)

    for i in range(B):
        ti = int(t_init_start[i])
        ts = int(t_start[i])
        if ti >= T:
            continue
        v0 = Vtarget[i, ti].clone()
        v0 = torch.where(torch.isnan(v0) | (v0 <= 0), torch.tensor(5.0, device=dev), v0)
        V = v0.clone()
        dv_dt = Vtarget[i, ti + 1] - Vtarget[i, ti] if ti + 1 < T else torch.tensor(0.0, device=dev)
        dv_dt = torch.where(torch.isnan(dv_dt), torch.zeros_like(dv_dt), dv_dt)
        m = calculate_m0_from_fast_torch(
            V.unsqueeze(0), dv_dt.unsqueeze(0),
            alpha[i:i+1, ti], beta[i:i+1, ti], gamma[i:i+1, ti],
            vp_used[i:i+1, ti], coeff_arr[i:i+1, ti]
        ).squeeze(0)
        m = torch.clamp(m, 0.01, 1.0)

        for t in range(ti, T):
            coeff_t = coeff_arr[i, t]
            xs_t = pred_xs[i, t]
            vp_t = vp_used[i, t]
            # 隐患3修复：vp>0.1 为海洋，与 training 一致
            is_ocean_t = ((~torch.isnan(vp_t)) & (vp_t > 0.1)).float()
            vp_t = torch.where(torch.isnan(vp_t) | (vp_t < 0), torch.zeros_like(vp_t), vp_t)
            if t < ts:
                Vtar_t = torch.where(torch.isnan(Vtarget[i, t]), V, Vtarget[i, t])
                Vtar_next = Vtarget[i, t + 1] if t + 1 < T else Vtar_t
                Vtar_next = torch.where(torch.isnan(Vtar_next), Vtar_t, Vtar_next)
                for _ in range(4):
                    dm = coeff_t * (is_ocean_t * (1.0 - m) * Vtar_next - xs_t * m)
                    m = torch.clamp(m + dm * STEP_SIZE, 0.01, 1.0)
                V = Vtar_next
            else:
                lead_h = t - ts
                decay_val = -2.0 * (lead_h / T0_DECAY_HOURS) ** 2
                decay = torch.exp(torch.tensor(decay_val, device=V.device, dtype=V.dtype))
                dV_extra = (F_init_end[i] * decay).unsqueeze(0)
                is_ocean_1 = is_ocean_t.unsqueeze(0)
                for _ in range(4):
                    V, m = _fast_step_torch(
                        xs_t.unsqueeze(0), V.unsqueeze(0), m.unsqueeze(0),
                        alpha[i:i+1, t], beta[i:i+1, t], gamma[i:i+1, t], vp_t.unsqueeze(0),
                        coeff_arr[i:i+1, t], dV_extra=dV_extra, is_ocean=is_ocean_1
                    )
                    V, m = V.squeeze(0), m.squeeze(0)
            v_fast[i, t] = V
            m_series[i, t] = m

    v_fast = torch.nan_to_num(v_fast, nan=0.0)
    m_series = torch.nan_to_num(m_series, nan=0.5)
    v_max = _axi_to_max_wind_torch(v_fast.unsqueeze(-1), pred_s, env_wnds, utran, vtran, lats)
    return v_max, v_fast.unsqueeze(-1), m_series.unsqueeze(-1)


def run_fast_physics_torch(pred_chi, pred_s, x_scalars, precomp, env_wnds, utran, vtran, lats,
                           training_mode=True, v_init=None, v_gt=None, f_init_end=None, m_init=None,
                           t_start=None, v_axisym_at_tstart=None, pred_chi_is_eff=False):
    """
    统一入口。
    training_mode=True:  纯 PyTorch 训练路径，梯度完整回传。
      v_axisym_at_tstart 不为 None 时：逐样本从 t_start 开始积分（完整对齐模式）。
      v_axisym_at_tstart 为 None 时：从 t=0 批量积分（兼容模式）。
    training_mode=False: 完整 run_fast_with_init（推理路径，仅用于图表 Baseline 绿虚线）。
    """
    if training_mode:
        if v_init is None and v_axisym_at_tstart is None:
            B, T = pred_chi.shape[0], pred_chi.shape[1]
            dev = pred_chi.device
            return torch.zeros(B, T, 1, device=dev), torch.zeros(B, T, 1, device=dev), torch.zeros(B, T, 1, device=dev)
        return run_fast_physics_torch_training(pred_chi, pred_s, x_scalars, v_init, env_wnds, utran, vtran, lats,
                                               v_gt=v_gt, f_init_end=f_init_end, m_init=m_init,
                                               t_start=t_start, v_axisym_at_tstart=v_axisym_at_tstart,
                                               pred_chi_is_eff=pred_chi_is_eff)
    else:
        return run_fast_physics_torch_inference(pred_chi, pred_s, x_scalars, precomp, env_wnds, utran, vtran, lats,
                                                pred_chi_is_eff=pred_chi_is_eff)


def _axi_to_max_wind_torch(tc_v, pred_s, env_wnds, utran, vtran, lats):
    """与 run_fast_reference.axi_to_max_wind_numpy 一致。"""
    lats_flat = lats if lats.dim() >= 2 else lats.unsqueeze(-1)
    G = torch.clamp(0.8 + 0.35 * (1.0 + torch.tanh((torch.abs(lats_flat) - 35.0) / 10.0)), max=1.0)
    if G.dim() == 3:
        G = G.squeeze(-1)
    u_shr = env_wnds[..., 0] - env_wnds[..., 2]
    v_shr = env_wnds[..., 1] - env_wnds[..., 3]
    shear_mag = torch.sqrt(u_shr**2 + v_shr**2 + 1e-12)
    has_env = ~(torch.isnan(u_shr) | torch.isnan(v_shr) | (shear_mag < 1e-6))
    u_dir = torch.where(has_env, u_shr / shear_mag, torch.zeros_like(u_shr))
    v_dir = torch.where(has_env, v_shr / shear_mag, torch.zeros_like(v_shr))
    s_safe = torch.nan_to_num(pred_s.squeeze(-1), nan=0.0)
    shear_coeff = 0.1 * s_safe * tc_v.squeeze(-1) / 15.0
    U_inc = G * utran.squeeze(-1) + shear_coeff * u_dir
    V_inc = G * vtran.squeeze(-1) + shear_coeff * v_dir
    mag_inc = torch.sqrt(U_inc**2 + V_inc**2 + 1e-12)
    mag_fac = torch.clamp((tc_v.squeeze(-1) * 0.5) / mag_inc, max=1.0)
    theta_opt = torch.atan2(-U_inc, V_inc)
    ug = tc_v.squeeze(-1) * (-torch.sin(theta_opt)) + U_inc * mag_fac
    vg = tc_v.squeeze(-1) * torch.cos(theta_opt) + V_inc * mag_fac
    return torch.sqrt(ug**2 + vg**2 + 1e-12).unsqueeze(-1)