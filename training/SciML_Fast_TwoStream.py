"""
TwoStream FAST - 完整自包含实现（干净版）
=========================================
四阶段训练课程:
  Phase A - 只训练 S 流 (U,V,Z → S，MLP 瞬时映射)
  Phase B - 只训练 Chi 流 (T,Q,SST,MSLP → Chi，MLP 瞬时映射)
  Phase C - 全冻结 ODE 物理完整性验证 + 画图
  Phase D - 全解冻端到端 ODE 物理微调

关键设计:
  - S/Chi 是诊断量，MLP 并行预测（无 LSTM，无均值作弊通道）
  - Avg+Max 双池化 + 3D attention，增强极端结构提取
  - LayerNorm 防止 CNN 特征爆炸导致 ODE 饱和
  - BatchNorm 冻结流 eval 防止运行统计量泄漏
  - 延迟加载空间场（仅按需读磁盘，避免 ~45GB RAM OOM）
  - DDP 多卡 + 梯度累加 + no_sync 优化

运行方式:
  单卡: python SciML_Fast_TwoStream.py [args]
  多卡: torchrun --nproc_per_node=4 SciML_Fast_TwoStream.py [args]
"""
import os, sys, csv, pickle, warnings, argparse, tempfile
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    print("wandb not installed. Run: pip install wandb")

# ── W&B 配置（仅 rank-0 使用）─────────────────────────────────────────────────
# Credentials come from the environment (WANDB_ENTITY/WANDB_PROJECT/WANDB_API_KEY
# or a normal `wandb login`) -- never hardcode keys in the repo.
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY", "xiaosj1103-georgia")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "Rapid Intensity")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fast_physics_torch import (
    chi_calibrated_torch, precompute_run_fast_init,
    run_fast_physics_torch, LOSS_TRIM_STEPS,
)
from SciML_Fast import (
    axi_to_max_wind_torch, plot_comparison_one_sample,
    _run_fast_reference_batch, _run_physics_ref, FAST_Physics_Engine,
    MS_TO_KNOTS,
)

# ── 全局常量 ──────────────────────────────────────────────────────────────────
SPATIAL_H, SPATIAL_W = 72, 72
N_3D_VARS, N_2D_VARS, N_LEVELS = 5, 2, 7
TARGET_SEQ_LEN      = 480    # 序列长度（步 = 小时）
# ── 阈值统一为 45kts（与 run_fast_reference.VMAX_START_KTS 完全一致）─────────
# 45kts 是物理预报起点：序列保留 t_45 前 48h（init）+ t_45 之后（预报）
# MIN_VMAX_KTS 也统一为 45.0，确保所有入库风暴都能做 48h 初始化+预报对比
MIN_VMAX_KTS        = 45.0   # 峰值强度过滤（kts）：必须至少达到起报阈值
MIN_DURATION_H      = 120    # 预报段最短时长（小时，t_45 之后）
START_FROM_VMAX_KTS = 45.0   # 弃用（保留接口兼容），内部统一使用 VMAX_START_KTS=45
TRAIN_YEARS = list(range(2003, 2023))  # 2003-2022
VAL_YEARS   = [2023, 2024]
TEST_YEARS  = []
CNN_CHUNK = 32   # CNN 分块大小，控制单次 GPU 峰值显存

# ─────────────────────────────────────────────────────────────────────────────
# DDP 工具
# ─────────────────────────────────────────────────────────────────────────────
def _is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def _setup_ddp():
    if 'LOCAL_RANK' not in os.environ:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return dev, 0, 1
    lr = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lr)
    dist.init_process_group(backend='nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        print(f"DDP: {world} GPUs")
    return torch.device(f'cuda:{lr}'), rank, world

# ─────────────────────────────────────────────────────────────────────────────
# 数据加载（延迟加载空间场，避免 ~45GB RAM OOM）
# ─────────────────────────────────────────────────────────────────────────────
def _trim(data, t0):
    """从 t0 开始裁剪所有数组。"""
    def sl(arr, ax=1):
        if arr is None: return None
        if arr.ndim == 1: return arr[t0:]
        s = [slice(None)]*arr.ndim; s[ax] = slice(t0, None)
        return arr[tuple(s)]
    d2 = {k: sl(v) for k, v in data.items() if isinstance(v, np.ndarray)}
    d2['hurricane'] = data.get('hurricane', '')
    if 'times' in data:
        d2['times'] = np.asarray(data['times']).ravel()[t0:]
    return d2

def _pad(d, tlen):
    """填充到固定长度。"""
    T = d['v_gt'].shape[1]
    def f(arr, ax=1):
        if arr is None: return None
        if arr.ndim == 1:
            n = arr.shape[0]
            if n >= tlen: return arr[:tlen]
            return np.concatenate([arr, np.repeat(arr[-1:], tlen - n)])
        if arr.ndim <= ax: return arr
        n = arr.shape[ax]
        if n >= tlen:
            s = [slice(None)]*arr.ndim; s[ax] = slice(0, tlen); return arr[tuple(s)]
        ls = [slice(None)]*arr.ndim; ls[ax] = slice(n-1, n)
        return np.concatenate([arr, np.repeat(arr[tuple(ls)], tlen-n, axis=ax)], axis=ax)
    d2 = {k: f(v) for k, v in d.items() if isinstance(v, np.ndarray)}
    d2['hurricane'] = d.get('hurricane', '')
    if 'times' in d and pd is not None:
        try:
            ta = np.asarray(d['times']).ravel()[:T]
            if len(ta) < tlen:
                extra = pd.date_range(start=pd.Timestamp(ta[-1]) + pd.Timedelta(hours=1),
                                      periods=tlen-len(ta), freq='h')
                ta = np.concatenate([ta, extra.to_numpy()])
            d2['times'] = ta[:tlen]
        except: d2['times'] = d.get('times')
    return d2, min(T, tlen)

def _load_spatial(ds):
    """按需从磁盘读取空间场（延迟加载），做 trim+pad。"""
    sp_path = ds.get('_spatial_path')
    t0, tlen = ds.get('_t0', 0), ds.get('_seq_len', TARGET_SEQ_LEN)
    if sp_path is None:
        return ds.get('spatial_3d'), ds.get('spatial_2d')
    with open(sp_path, 'rb') as f:
        sp = pickle.load(f)
    s3d = sp.get('spatial_3d_1km')
    s2d = sp.get('spatial_2d_1km')
    if s3d is None or s2d is None:
        return (np.zeros((1, tlen, N_3D_VARS, N_LEVELS, SPATIAL_H, SPATIAL_W), np.float32),
                np.zeros((1, tlen, N_2D_VARS, SPATIAL_H, SPATIAL_W), np.float32))
    s3d = s3d[:, t0:].astype(np.float32)
    s2d = s2d[:, t0:].astype(np.float32)
    def fp(arr, ax=1):
        n = arr.shape[ax]
        if n >= tlen:
            sl = [slice(None)]*arr.ndim; sl[ax] = slice(0, tlen); return arr[tuple(sl)]
        ls = [slice(None)]*arr.ndim; ls[ax] = slice(n-1, n)
        return np.concatenate([arr, np.repeat(arr[tuple(ls)], tlen-n, axis=ax)], axis=ax)
    return fp(s3d), fp(s2d)

def load_1km_storms(data_dir='training_data', seq_len=TARGET_SEQ_LEN,
                    min_vmax_kts=MIN_VMAX_KTS, start_from=START_FROM_VMAX_KTS,
                    min_duration_h=MIN_DURATION_H,
                    train_years=None, val_years=None, test_years=None,
                    storm_include=None, storm_exclude=None, keep_weak=False):
    """
    加载所有同时有 *_dataset.pkl 和 *_spatial_1000km.pkl 的风暴。

    序列时间轴（与 run_fast_reference 完全对齐）：
      t=0               t_start(~48)               T-1
      |← 48h init period →|←── 预报段（计 loss）──→|

    - t0 = max(0, t_45 - 48)：保留 48h 历史供 nudging init 使用
    - 过滤：预报段（t_45 之后）>= min_duration_h
    - 训练 loss 仅对 t >= t_start 计算（_vm use_tstart=True）
    """
    ty = set(train_years or TRAIN_YEARS)
    vy = set(val_years   or VAL_YEARS)
    tey= set(test_years  or TEST_YEARS)
    inc= set(storm_include) if storm_include else None
    # storm_exclude 可以是文件路径（逐行读取）或 list/set（直接用）
    exc: set = set()
    if storm_exclude:
        p_exc = Path(storm_exclude) if isinstance(storm_exclude, str) else None
        if p_exc is not None and p_exc.exists():
            for _line in p_exc.read_text().splitlines():
                _sid = _line.split('#')[0].strip()
                if _sid: exc.add(_sid)
        elif not isinstance(storm_exclude, str):
            exc = set(storm_exclude)
    mms= min_vmax_kts / MS_TO_KNOTS
    tr, va, te, cnt = [], [], [], {}
    skip_weak, skip_short = 0, 0

    for yd in sorted(Path(data_dir).iterdir()):
        if not yd.is_dir() or not yd.name.isdigit(): continue
        yr = int(yd.name); cnt[yr] = 0
        for sd in sorted(yd.iterdir()):
            if not sd.is_dir(): continue
            sid = f"{yr}/{sd.name}"
            if exc and sid in exc: continue
            if inc and sid not in inc: continue
            dpkl = list(sd.glob('*_dataset.pkl'))
            spkl = list(sd.glob('*_spatial_1000km.pkl'))
            if not dpkl: continue
            try:
                with open(dpkl[0], 'rb') as f: data = pickle.load(f)
            except: continue
            # Two supported layouts:
            #  (old) separate *_spatial_1000km.pkl with 'spatial_3d_1km' (72x72)
            #  (new, PINN single-pkl) dataset carries 'spatial_3d' (72x72) directly
            if spkl:
                try:
                    with open(spkl[0], 'rb') as f: sp_meta = pickle.load(f)
                except: continue
                s3d_test = sp_meta.get('spatial_3d_1km')
                if s3d_test is None or s3d_test.shape[-2:] != (SPATIAL_H, SPATIAL_W): continue
            else:
                s3d_test = data.get('spatial_3d')
                if s3d_test is None or np.asarray(s3d_test).shape[-2:] != (SPATIAL_H, SPATIAL_W): continue

            # lats/lons 统一 [1, T]
            for k in ['lats', 'lons']:
                if k in data:
                    a = np.asarray(data[k])
                    if a.ndim == 1: data[k] = a.reshape(1, -1)

            vgt = data['v_gt'][0, :, 0]

            # ── 过滤 1：峰值强度 ─────────────────────────────────────────────
            vmax_ms = float(np.nanmax(vgt)) if np.any(np.isfinite(vgt)) else 0.0
            if vmax_ms < mms:
                skip_weak += 1; continue

            kts = vgt * MS_TO_KNOTS
            # ── 找 t_45：首次 >= 45kts（VMAX_START_KTS，与 run_fast_reference 一致）
            t_45 = next((t for t in range(len(kts))
                         if np.isfinite(kts[t]) and kts[t] >= 45.0), None)
            if t_45 is None:
                if keep_weak:
                    t_45 = 0   # ensemble inference: keep weak storms (<45kt), start at h0
                else:
                    skip_weak += 1; continue  # training: 从未达到 45kts，排除

            # ── t0 = max(0, t_45 - 48)：保留 48h 历史给 nudging init ────────
            t0 = max(0, t_45 - 48)

            # ── 过滤 2：预报段时长（t_45 之后）>= min_duration_h ─────────────
            forecast_h = len(kts) - t_45
            if forecast_h < min_duration_h:
                skip_short += 1; continue
            if forecast_h < 3: continue

            d_trim = _trim(data, t0)
            d_pad, seq_valid = _pad(d_trim, seq_len)
            T = seq_len

            # ── 预计算台风惯性强迫力 F_init_end（廉价：单风暴 ~0.01s）─────────
            # F_init_end 表示该台风在预报起点前 48h 积累的内部动量储备。
            # 缓存于 ds['f_init_end']，训练时注入 ODE，与 FAST 推理模式对齐，
            # 防止 ML 模型为"弥补惯性缺失"而扭曲 chi/S 的预测。
            try:
                _pc = precompute_run_fast_init(
                    d_pad.get('scalars'), d_pad.get('v_gt'), d_pad.get('env_wnds'),
                    d_pad.get('utran'),   d_pad.get('vtran'), d_pad.get('lats'),
                    d_pad.get('s_ref'),   d_pad.get('lons'),  d_pad.get('chi_ref'),
                )
                f_init_end_val       = float(_pc['F_init_end'][0])           if _pc is not None else 0.0
                m0_ref_val           = float(_pc['m0_init'][0])              if _pc is not None else 0.5
                m_at_tstart_val      = float(_pc['m_at_tstart'][0])          if _pc is not None else 0.5
                v_axisym_tstart_val  = float(_pc['v_axisym_at_tstart'][0])   if _pc is not None else 5.0
                t_start_val          = int(_pc['t_start'][0])                if _pc is not None else 0
            except Exception:
                f_init_end_val       = 0.0
                m0_ref_val           = 0.5
                m_at_tstart_val      = 0.5
                v_axisym_tstart_val  = 5.0
                t_start_val          = 0

            ds = {
                'hurricane':     data.get('hurricane', sd.name),
                'year':          yr,
                # old layout: lazy-load from separate spatial pkl;
                # new single-pkl layout: spatial_3d already trimmed/padded in d_pad
                '_spatial_path': (str(spkl[0]) if spkl else None),
                '_t0':           t0,
                '_seq_len':      seq_len,
                'f_init_end':          f_init_end_val,      # 台风惯性强迫力
                't_start':             t_start_val,         # 预报起报点（在 trimmed 序列中的索引，≈48）
                'm_at_tstart':         m_at_tstart_val,     # 48h spin-up 结束时的 m
                'v_axisym_at_tstart':  v_axisym_tstart_val, # 48h nudging 后的轴对称风速 V_axisym
                'scalars':  d_pad.get('scalars',  np.zeros((1,T,4),np.float32)),
                'chi_ref':  d_pad.get('chi_ref',  np.zeros((1,T,1),np.float32)),
                's_ref':    d_pad.get('s_ref',    np.zeros((1,T,1),np.float32)),
                'xs_ref':   d_pad.get('xs_ref',   np.zeros((1,T,1),np.float32)),
                'v_init':   d_pad.get('v_init',   np.zeros((1,T,1),np.float32)),
                'v_gt':     d_pad.get('v_gt',     np.zeros((1,T,1),np.float32)),
                'env_wnds': d_pad.get('env_wnds', np.zeros((1,T,4),np.float32)),
                'utran':    d_pad.get('utran',    np.zeros((1,T,1),np.float32)),
                'vtran':    d_pad.get('vtran',    np.zeros((1,T,1),np.float32)),
                'lats':     d_pad.get('lats',     np.zeros((1,T),  np.float32)),
                'lons':     d_pad.get('lons',     np.zeros((1,T),  np.float32)),
                'seq_len':  seq_valid,
                'times':    d_pad.get('times'),
            }
            # new single-pkl layout: carry the (already trimmed+padded) spatial in-dict
            if not spkl:
                ds['spatial_3d'] = d_pad.get('spatial_3d')
                ds['spatial_2d'] = d_pad.get('spatial_2d')
            # m0_ref: 供旧接口使用（48h init 开头的 m）
            ds['m0_ref'] = data['m0_ref'] if 'm0_ref' in data else np.array([[m0_ref_val]], dtype=np.float32)
            bucket = tr if yr in ty else (va if yr in vy else (te if yr in tey else None))
            if bucket is not None:
                bucket.append(ds); cnt[yr] += 1

    if _is_main():
        print(f"1km storms: {[(y,n) for y,n in sorted(cnt.items()) if n>0]}")
        print(f"train={len(tr)}  val={len(va)}  test={len(te)}"
              f"  (filtered: weak={skip_weak}, short<{min_duration_h}h={skip_short})")
    return tr, va, te

# ─────────────────────────────────────────────────────────────────────────────
# 归一化（逐变量逐层，保留垂直梯度敏感性）
# ─────────────────────────────────────────────────────────────────────────────
def compute_spatial_stats(datasets, max_samples=30):
    """采样 max_samples 个风暴计算归一化统计，避免全量加载 OOM。"""
    if not datasets: return None
    n = min(max_samples, len(datasets))
    if _is_main(): print(f"  Computing spatial stats from {n}/{len(datasets)} storms ...")
    s3l, s2l = [], []
    for d in datasets[:n]:
        s3, s2 = _load_spatial(d)
        if s3 is not None: s3l.append(s3)
        if s2 is not None: s2l.append(s2)
    if not s3l: return None
    s3 = np.concatenate(s3l, 0); s2 = np.concatenate(s2l, 0)
    def _stat(x):
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if len(x)>0 else 0., max(float(np.std(x)) if len(x)>1 else 1., 1e-8)
    st3 = [[_stat(s3[:,:,vi,li].ravel()) for li in range(N_LEVELS)] for vi in range(N_3D_VARS)]
    st2 = [_stat(s2[:,:,vi].ravel()) for vi in range(N_2D_VARS)]
    return {'3d': st3, '2d': st2}

def normalize_spatial(s3, s2, stats):
    s3 = np.array(s3, dtype=np.float32, copy=True)
    s2 = np.array(s2, dtype=np.float32, copy=True)
    for vi, lvl in enumerate(stats['3d']):
        for li, (mu, std) in enumerate(lvl):
            s3[:,:,vi,li] = np.nan_to_num((s3[:,:,vi,li]-mu)/std, nan=0., posinf=0., neginf=0.)
    for vi, (mu, std) in enumerate(stats['2d']):
        s2[:,:,vi] = np.nan_to_num((s2[:,:,vi]-mu)/std, nan=0., posinf=0., neginf=0.)
    return s3, s2

def create_batch(datasets, batch_size, device, shuffle=True, spatial_stats=None, rank=0, world=1):
    """按需加载空间场并打包 batch。"""
    n = min(batch_size, len(datasets))
    idx = np.random.choice(len(datasets), n, replace=False) if shuffle else np.arange(n)
    # 延迟加载空间场
    s3l, s2l = [], []
    for i in idx:
        s3, s2 = _load_spatial(datasets[i])
        s3l.append(s3); s2l.append(s2)
    s3 = np.concatenate(s3l, 0); s2 = np.concatenate(s2l, 0)
    if spatial_stats: s3, s2 = normalize_spatial(s3, s2, spatial_stats)
    T = s3.shape[1]
    def cat(k, sh):
        arrs = []
        for i in idx:
            v = datasets[i].get(k)
            arrs.append(v if v is not None else np.zeros(sh, np.float32))
        return np.concatenate(arrs, 0)
    b = {
        'spatial_3d': torch.from_numpy(s3).float().to(device),
        'spatial_2d': torch.from_numpy(s2).float().to(device),
        'scalars':  torch.from_numpy(cat('scalars', (1,T,4))).float().to(device),
        'chi_ref':  torch.from_numpy(cat('chi_ref', (1,T,1))).float().to(device),
        's_ref':    torch.from_numpy(cat('s_ref',   (1,T,1))).float().to(device),
        'xs_ref':   torch.from_numpy(cat('xs_ref',  (1,T,1))).float().to(device),
        'v_init':   torch.from_numpy(cat('v_init',  (1,T,1))).float().to(device),
        'v_gt':     torch.from_numpy(cat('v_gt',    (1,T,1))).float().to(device),
        'env_wnds': torch.from_numpy(cat('env_wnds',(1,T,4))).float().to(device),
        'utran':    torch.from_numpy(cat('utran',   (1,T,1))).float().to(device),
        'vtran':    torch.from_numpy(cat('vtran',   (1,T,1))).float().to(device),
        'lats':     torch.from_numpy(cat('lats',    (1,T)  )).float().to(device),
        'lons':     torch.from_numpy(cat('lons',    (1,T)  )).float().to(device),
    }
    # chi_ref_raw: 数据集原始 chi；chi_ref_eff: 与 SciML_Fast/_chi_calibrated_multiply 一致的 0~4 监督口径
    # 关键：保留 raw 有效性，避免 NaN 在校准后变成“伪有效小值”
    b['chi_ref_raw'] = b['chi_ref']
    chi_valid = torch.isfinite(b['chi_ref_raw']) & (b['chi_ref_raw'] > 0)
    chi_eff_all = chi_calibrated_torch(torch.where(chi_valid, b['chi_ref_raw'], torch.zeros_like(b['chi_ref_raw'])))
    b['chi_ref_eff'] = torch.where(chi_valid, chi_eff_all, torch.full_like(chi_eff_all, float('nan')))
    # 逐时刻有效掩码：用于统一屏蔽范围外占位步（chi/s 为 NaN）
    s_valid = torch.isfinite(b['s_ref']) & (b['s_ref'] > 0)
    b['step_valid'] = (chi_valid & s_valid).squeeze(-1)  # [B, T]
    sl = torch.tensor([datasets[i].get('seq_len',T) for i in idx], dtype=torch.long, device=device)
    b['seq_len']    = sl
    b['valid_mask'] = torch.arange(T, device=device).unsqueeze(0) < sl.unsqueeze(1)
    # 台风惯性强迫力：[B] float tensor，训练时注入 ODE dV_extra
    b['f_init_end'] = torch.tensor(
        [datasets[i].get('f_init_end', 0.0) for i in idx],
        dtype=torch.float32, device=device)
    # t_start [B]：预报起报点（首次 >=45kts）
    b['t_start'] = torch.tensor(
        [datasets[i].get('t_start', 0) for i in idx],
        dtype=torch.long, device=device)
    # m0_ref [B,1]：48h spin-up 开始时的 m（历史保留接口）
    b['m0_ref'] = torch.from_numpy(
        np.concatenate([datasets[i].get('m0_ref', np.array([[.5]])) for i in idx], 0)
    ).float().to(device)
    # m_at_tstart [B,1]：48h spin-up 结束时的 m（预报起点，物理最准确）
    b['m_at_tstart'] = torch.tensor(
        [datasets[i].get('m_at_tstart', 0.5) for i in idx],
        dtype=torch.float32, device=device).unsqueeze(-1)  # [B,1]
    # v_axisym_at_tstart [B,1]：V_axisym 在预报起点（来自 precompute Vtarget[t_start]）
    b['v_axisym_at_tstart'] = torch.tensor(
        [datasets[i].get('v_axisym_at_tstart', 5.0) for i in idx],
        dtype=torch.float32, device=device).unsqueeze(-1)  # [B,1]
    b['times'] = [datasets[i].get('times') for i in idx]
    return b

# ─────────────────────────────────────────────────────────────────────────────
# 模型（MLP，无 LSTM，并行预测）
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_enc(encoder, x):
    return (encoder(x) if x.shape[0] <= CNN_CHUNK
            else torch.cat([encoder(c) for c in x.split(CNN_CHUNK, 0)], 0))


class Enc3D(nn.Module):
    """
    [B,1,7,H,W] → [B,out_dim]
    Avg+Max 双池化：兼顾整体统计与高值极端信息。
    低值分支：显式编码低值极端（avg-min），提升弱信号捕捉。
    3D channel attention（残差缩放）：避免低值被过度抑制。
    LayerNorm 防止特征爆炸导致 ODE 饱和。
    """
    def __init__(self, out_dim=64, use_vert_diff=False):
        super().__init__()
        self.use_vert_diff = use_vert_diff
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, (2,3,3), stride=(1,2,2), padding=(0,1,1)), nn.BatchNorm3d(16),  nn.GELU(),
            nn.Conv3d(16,32, (2,3,3), stride=(1,2,2), padding=(0,1,1)), nn.BatchNorm3d(32),  nn.GELU(),
            nn.Conv3d(32,64, (2,3,3), stride=(1,2,2), padding=(0,1,1)), nn.BatchNorm3d(64),  nn.GELU(),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(64, 16), nn.GELU(),
            nn.Linear(16, 64), nn.Tanh(),
        )
        self.pool_avg = nn.AdaptiveAvgPool3d((4, 2, 2))
        self.pool_max = nn.AdaptiveMaxPool3d((4, 2, 2))
        # extreme-aware: 同时编码高值极端(max-avg)与低值极端(avg-min)
        diff_dim = 16 * 2 * 2 if use_vert_diff else 0
        if use_vert_diff:
            # 显式中层-低层差分分支（重点服务通风相关层结信号）
            self.diff_proj = nn.Sequential(
                nn.Conv2d(1, 8, 3, 2, 1), nn.BatchNorm2d(8), nn.GELU(),
                nn.Conv2d(8, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
        self.fc = nn.Sequential(nn.Linear(64*4*2*2*2 + 64*2 + diff_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim))

    def forward(self, x):
        h = self.conv(x)
        g_avg = F.adaptive_avg_pool3d(h, (1,1,1)).flatten(1)
        g_max = F.adaptive_max_pool3d(h, (1,1,1)).flatten(1)
        g_min = torch.amin(h, dim=(2, 3, 4))
        gate = self.attn_mlp(0.5 * (g_avg + g_max)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 残差缩放：范围约 [0.5, 1.5]，保留弱信号底座
        h = h * (1.0 + 0.5 * gate)
        p_avg = self.pool_avg(h).flatten(1)
        p_max = self.pool_max(h).flatten(1)
        p_ext_hi = (g_max - g_avg)
        p_ext_lo = (g_avg - g_min)
        feats = [p_avg, p_max, p_ext_hi, p_ext_lo]
        if self.use_vert_diff:
            # x:[N,1,7,H,W] -> low(1000/850) vs mid(700/600/500)
            low = x[:, :, 0:2].mean(dim=2)   # [N,1,H,W]
            mid = x[:, :, 2:5].mean(dim=2)   # [N,1,H,W]
            vdiff = self.diff_proj(mid - low).flatten(1)
            feats.append(vdiff)
        return self.fc(torch.cat(feats, dim=1))


class Enc2D(nn.Module):
    """
    [B,1,H,W] → [B,out_dim]
    Avg+Max 双池化 + 低值分支：保留四象限统计并强化极值/弱值信号。
    """
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16,32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32,64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.pool_avg = nn.AdaptiveAvgPool2d((2, 2))
        self.pool_max = nn.AdaptiveMaxPool2d((2, 2))
        self.fc = nn.Sequential(nn.Linear(64*2*2*2 + 64*2, out_dim), nn.GELU(), nn.LayerNorm(out_dim))

    def forward(self, x):
        h = self.conv(x)
        g_avg = F.adaptive_avg_pool2d(h, (1,1)).flatten(1)
        g_max = F.adaptive_max_pool2d(h, (1,1)).flatten(1)
        g_min = torch.amin(h, dim=(2, 3))
        p_avg = self.pool_avg(h).flatten(1)
        p_max = self.pool_max(h).flatten(1)
        p_ext_hi = (g_max - g_avg)
        p_ext_lo = (g_avg - g_min)
        return self.fc(torch.cat([p_avg, p_max, p_ext_hi, p_ext_lo], dim=1))


class TwoStreamFASTModel(nn.Module):
    """
    S 流  (U,V,Z → S):      MLP 瞬时预测（无 LSTM，无均值记忆通道）
    Chi 流 (T,Q,SST,MSLP → Chi): MLP 瞬时预测

    S_PARAMS   = CNN encoders for S  + head_s
    CHI_PARAMS = CNN encoders for Chi + head_chi
    """
    S_PARAMS   = ('enc_u', 'enc_v', 'enc_z', 'head_s')
    CHI_PARAMS = ('enc_t', 'enc_q', 'enc_sst', 'enc_mslp', 'head_chi')
    # CNN encoder 前缀（Phase D1 冻结这些，只微调 head）
    ENC_PARAMS = ('enc_u', 'enc_v', 'enc_z', 'enc_t', 'enc_q', 'enc_sst', 'enc_mslp')

    def __init__(self, feat_dim=64, scalar_dim=5):
        super().__init__()
        self.enc_u = Enc3D(feat_dim); self.enc_v = Enc3D(feat_dim); self.enc_z = Enc3D(feat_dim)
        # S 采用 base + delta：先拟合主趋势，再学习小幅纠偏（适配低值段）
        self.head_s_backbone = nn.Sequential(
            nn.Linear(feat_dim*3, 128), nn.GELU(),
            nn.Linear(128, 64),         nn.GELU(),
        )
        self.head_s_base = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        self.head_s_delta = nn.Linear(64, 1)
        # T/Q 编码器启用中低层差分分支，显式强化通风层结识别
        self.enc_t = Enc3D(feat_dim, use_vert_diff=True); self.enc_q = Enc3D(feat_dim, use_vert_diff=True)
        self.enc_sst = Enc2D(feat_dim); self.enc_mslp = Enc2D(feat_dim)
        self.head_chi = nn.Sequential(
            nn.Linear(feat_dim*4 + scalar_dim, 128), nn.GELU(),
            nn.Linear(128, 64),                      nn.GELU(),
            nn.Linear(64, 1),                        nn.Sigmoid(),
        )
        # 初始偏置 -2.0：Sigmoid(-2) ≈ 0.12，接近 chi_ref(raw) 的常见低值背景
        nn.init.constant_(self.head_chi[-2].bias, -2.0)
        nn.init.zeros_(self.head_s_delta.weight)
        nn.init.zeros_(self.head_s_delta.bias)

    def forward(self, x3d, x2d, sc, v_init,
                precomp=None, env_wnds=None, utran=None, vtran=None, lats=None,
                v_gt=None, use_run_fast_physics=True, f_init_end=None, m_init=None, t_start=None,
                v_axisym_at_tstart=None, plot_mode=False):
        """
        plot_mode=True：推理路径，前 48h 跟踪观测（与 run_fast_reference 一致），
                        t>=t_start 后用 ML 预测的 pred_s/pred_chi 自由演化。
                        仅用于图表可视化，保证与 reference 公平对比。
        plot_mode=False（默认）：训练路径，从 t=0 纯 ODE 积分，梯度可回传。
        """
        B, T = x3d.shape[0], x3d.shape[1]
        f3 = rearrange(x3d, 'b t c l h w -> (b t) c l h w')
        f2 = rearrange(x2d, 'b t c h w   -> (b t) c h w')

        fu=_chunk_enc(self.enc_u,f3[:,2:3]); fv=_chunk_enc(self.enc_v,f3[:,3:4])
        fz=_chunk_enc(self.enc_z,f3[:,4:5]); ft=_chunk_enc(self.enc_t,f3[:,0:1])
        fq=_chunk_enc(self.enc_q,f3[:,1:2]); fs=_chunk_enc(self.enc_sst,f2[:,0:1])
        fm=_chunk_enc(self.enc_mslp,f2[:,1:2])

        dyn    = rearrange(torch.cat([fu,fv,fz],1), '(b t) d -> b t d', b=B)  # [B,T,3f]
        thermo_cat = torch.cat([ft,fq,fs,fm],1)
        thermo = rearrange(thermo_cat, '(b t) (k d) -> b t k d', b=B, k=4)  # [B,T,4,f]

        thermo = rearrange(thermo, 'b t k d -> b t (k d)')   # [B,T,4f]

        # 熵亏损代理：Delta_s ~= SST - s_mid_proxy(T,Q)
        sst_mean = x2d[:, :, 0:1].mean(dim=(-2, -1))              # [B,T,1]
        mid_t = x3d[:, :, 0, 2:5].mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)  # [B,T,1]
        mid_q = x3d[:, :, 1, 2:5].mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)  # [B,T,1]
        delta_s = torch.tanh(sst_mean - (mid_t - 2.5 * mid_q))                        # [B,T,1]

        hs = self.head_s_backbone(dyn)
        s_base = self.head_s_base(hs)
        s_delta = 0.25 * torch.tanh(self.head_s_delta(hs))  # 小幅纠偏，防止早期不稳定
        pred_s   = torch.clamp(s_base + s_delta, 0.1, 50.)                         # [B,T,1]
        # χ 头内部先给 raw χ，再校准为 χ_eff（0~4）作为统一监督与物理输入
        # chi 独立分支：不使用风场切变输入，仅使用热力特征 + 物理标量 + 熵亏损代理
        chi_aux = torch.cat([sc, delta_s], dim=-1)                                              # [B,T,5]
        pred_chi_raw = torch.clamp(self.head_chi(torch.cat([thermo, chi_aux], dim=-1)), 0.0, 1.0)  # 仅用于诊断
        pred_chi_eff = torch.clamp(5.0 * pred_chi_raw, 0.0, 4.0)                                # [B,T,1]
        pred_xs  = pred_chi_eff * pred_s                                                       # [B,T,1]

        if use_run_fast_physics and env_wnds is not None:
            if plot_mode and precomp is not None:
                # ── 图表模式：48h 跟踪观测（run_fast_reference 逻辑）+ ML S/Chi 预报 ─
                # 与 reference 公平比较：两者都从观测初始化，然后分别用 ref_xs 和 pred_xs
                vm, vf, ms = run_fast_physics_torch(
                    pred_chi_eff, pred_s, sc, precomp, env_wnds, utran, vtran, lats,
                    training_mode=False, pred_chi_is_eff=True)
            else:
                # ── 训练/验证 loss 模式：从 t_start 开始积分（完整对齐模式）──
                vm, vf, ms = run_fast_physics_torch(
                    pred_chi_eff, pred_s, sc, None, env_wnds, utran, vtran, lats,
                    training_mode=True, v_init=v_init, v_gt=v_gt,
                    f_init_end=f_init_end, m_init=m_init, t_start=t_start,
                    v_axisym_at_tstart=v_axisym_at_tstart, pred_chi_is_eff=True)
            return {'v':vf,'m':ms,'s':pred_s,'chi':pred_chi_eff,'xs':pred_xs,'v_max':vm}

        return {'v':torch.zeros_like(pred_s),'m':torch.zeros_like(pred_s),
                's':pred_s,'chi':pred_chi_eff,'xs':pred_xs,'v_max':None}


def _raw(m): return m.module if isinstance(m, (DDP, nn.DataParallel)) else m


def set_stream_trainable(model, s=True, chi=True, enc=True):
    """
    冻结/解冻参数 + BN eval/train。
    enc=False 时冻结所有 CNN encoder（保留特征提取能力，只微调 head_s / head_chi）。
    requires_grad=False 不能阻止 BN running_stats 更新，必须显式设 eval()。
    """
    raw = _raw(model)
    BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for nm, p in raw.named_parameters():
        is_s   = any(nm.startswith(k) for k in TwoStreamFASTModel.S_PARAMS)
        is_chi = any(nm.startswith(k) for k in TwoStreamFASTModel.CHI_PARAMS)
        is_enc = any(nm.startswith(k) for k in TwoStreamFASTModel.ENC_PARAMS)
        if is_enc:
            # enc=False 时强制冻结 CNN，不受 s/chi 控制
            p.requires_grad = enc and (s if is_s else chi)
        elif is_s:
            p.requires_grad = s
        elif is_chi:
            p.requires_grad = chi
    for nm, mod in raw.named_modules():
        if not isinstance(mod, BN): continue
        is_s   = any(nm.startswith(k) for k in TwoStreamFASTModel.S_PARAMS)
        is_chi = any(nm.startswith(k) for k in TwoStreamFASTModel.CHI_PARAMS)
        is_enc = any(nm.startswith(k) for k in TwoStreamFASTModel.ENC_PARAMS)
        trainable = enc and (s if is_s else chi) if is_enc else (s if is_s else chi if is_chi else True)
        mod.train() if trainable else mod.eval()


def _fix_bn(model, s, chi, enc=True):
    """model.train() 后恢复冻结流的 BN eval 状态。"""
    raw = _raw(model)
    BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for nm, mod in raw.named_modules():
        if not isinstance(mod, BN): continue
        is_s   = any(nm.startswith(k) for k in TwoStreamFASTModel.S_PARAMS)
        is_chi = any(nm.startswith(k) for k in TwoStreamFASTModel.CHI_PARAMS)
        is_enc = any(nm.startswith(k) for k in TwoStreamFASTModel.ENC_PARAMS)
        trainable = enc and (s if is_s else chi) if is_enc else (s if is_s else chi if is_chi else True)
        if not trainable: mod.eval()


def load_ckpt(model, path, device):
    """strict=False 加载，兼容任意旧版本 checkpoint。"""
    if not Path(path).exists(): return False
    try:
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
        miss, unexp = _raw(model).load_state_dict(sd, strict=False)
        if _is_main():
            print(f"  Loaded {path}: miss={len(miss)} unexp={len(unexp)}"
                  + (" (partial)" if miss or unexp else " OK"))
        return True
    except Exception as e:
        if _is_main(): print(f"  Warning: cannot load {path}: {e}")
        return False


def save_ckpt(model, path, upload_to_wandb=False):
    if not _is_main(): return
    if not any(torch.isnan(p).any() or torch.isinf(p).any() for p in _raw(model).parameters()):
        torch.save(_raw(model).state_dict(), path)
        if upload_to_wandb:
            _wandb_save_file(path, policy="now")

# ─────────────────────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────────────────────
def _safe(t): return t if torch.isfinite(t).all() else torch.zeros(1, device=t.device).squeeze()
def _sfn(n):  return (n.replace('/','_').replace(' ','_').strip() or 'unknown')

def _mfilt(sc, dev):
    from run_fast_reference import _median_filter_1d
    sc = sc.clone()
    for b in range(sc.shape[0]):
        sc[b,:,3] = torch.from_numpy(_median_filter_1d(sc[b,:,3].cpu().numpy(),3)).float().to(dev)
    return sc

def _pc(sc, vb):
    if not all(k in vb for k in ['env_wnds','utran','vtran','lats']): return None
    return precompute_run_fast_init(
        sc.cpu().numpy(), vb['v_gt'].cpu().numpy(), vb['env_wnds'].cpu().numpy(),
        vb['utran'].cpu().numpy(), vb['vtran'].cpu().numpy(), vb['lats'].cpu().numpy(),
        vb['s_ref'].cpu().numpy(), vb['lons'].cpu().numpy() if 'lons' in vb else None,
        vb['chi_ref'].cpu().numpy() if 'chi_ref' in vb else None)

def _vm(batch, dev, use_tstart=False):
    """
    use_tstart=True（ODE phase）：每个样本从各自 t_start 开始计 loss，
        确保 loss 只来自 48h spin-up 之后的自由预报段，与 reference 完全对齐。
    use_tstart=False（Phase A/B）：保留前 LOSS_TRIM_STEPS 步截断逻辑。
    """
    vm = batch['valid_mask'].unsqueeze(-1)  # [B, T, 1]
    if 'step_valid' in batch:
        vm = vm & batch['step_valid'].unsqueeze(-1)
    if use_tstart and 't_start' in batch:
        T = vm.shape[1]
        t_arr = torch.arange(T, device=dev).unsqueeze(0)  # [1, T]
        ts    = batch['t_start'].unsqueeze(1)              # [B, 1]
        return vm & (t_arr >= ts).unsqueeze(-1)            # [B, T, 1]
    return vm & (torch.arange(vm.shape[1],device=dev).unsqueeze(0)>=LOSS_TRIM_STEPS).unsqueeze(-1)

# Monitor：图表模式——前 48h 跟踪观测 + ML S/Chi 预报，与 reference 公平对比
def _monitor(model, ds, stats, dev, epoch, outdir, cache=None):
    if not _is_main(): return cache
    os.makedirs(outdir, exist_ok=True)
    raw = _raw(model); raw.eval()
    with torch.no_grad():
        vb = create_batch([ds], 1, dev, shuffle=False, spatial_stats=stats)
        sc = _mfilt(vb['scalars'].clone(), dev)
        pc = cache or _pc(sc, vb)
        # plot_mode=True：48h 跟踪观测初始化，t>=t_start 后用 ML pred_s/pred_chi 自由演化
        out = raw(vb['spatial_3d'],vb['spatial_2d'],sc,vb['v_gt'][:,0:1,:],
                  precomp=pc, env_wnds=vb.get('env_wnds'),utran=vb.get('utran'),
                  vtran=vb.get('vtran'),lats=vb.get('lats'),use_run_fast_physics=True,
                  plot_mode=True)
        pv = out['v_max'] if out.get('v_max') is not None else \
             axi_to_max_wind_torch(out['v'],out['s'],vb['env_wnds'],vb['utran'],vb['vtran'],vb['lats'])
        rv,rm = _run_fast_reference_batch(vb['s_ref'],vb['chi_ref'],vb['xs_ref'],
                                          vb['scalars'].clone(),vb['v_gt'],vb['env_wnds'],
                                          vb['utran'],vb['vtran'],vb['lats'],vb.get('lons'))
    vfn = rv[0] if rv is not None else out['v'][0,:,0].cpu().numpy()
    mfn = rm[0] if rm is not None else out['m'][0,:,0].cpu().numpy()
    plot_comparison_one_sample(
        vb['v_gt'][0,:,0].cpu().numpy(), vb['s_ref'][0,:,0].cpu().numpy(),
        vb['chi_ref_eff'][0,:,0].cpu().numpy(), pv[0,:,0].cpu().numpy(),
        out['s'][0,:,0].cpu().numpy(), out['chi'][0,:,0].cpu().numpy(),
        out['xs'][0,:,0].cpu().numpy(), out['m'][0,:,0].cpu().numpy(), vfn, mfn,
        vb['seq_len'][0].item(), ds.get('hurricane','monitor'),
        os.path.join(outdir,f"ep{epoch:04d}.png"),
        ds.get('m0_ref',np.array([[.5]]))[0,0] if 'm0_ref' in ds else None,
        ds.get('times'))
    raw.train()
    return pc

# ─────────────────────────────────────────────────────────────────────────────
# W&B 工具
# ─────────────────────────────────────────────────────────────────────────────

def _wandb_init(config: dict):
    """初始化 wandb run（仅 rank-0 调用）。"""
    if not _WANDB_AVAILABLE or not _is_main():
        return
    try:
        if WANDB_API_KEY:
            wandb.login(key=WANDB_API_KEY, relogin=False)
        else:
            wandb.login()   # rely on ~/.netrc or WANDB_API_KEY env from wandb
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            config=config,
            resume='allow',
        )
        # 每次 run 保存核心代码快照，便于后续恢复
        code_files = [
            Path(__file__).resolve(),
            Path(__file__).resolve().parent / 'run_fast_physics_torch.py',
            Path(__file__).resolve().parent / 'run_fast_reference.py',
            Path(__file__).resolve().parent / 'SciML_Fast.py',
        ]
        for fp in code_files:
            if fp.exists():
                try:
                    wandb.save(str(fp), policy="now")
                except Exception:
                    pass
        print(f"  W&B initialized: {WANDB_PROJECT} / {wandb.run.name}")
    except Exception as e:
        print(f"  W&B init failed: {e}")


def _wandb_log(metrics: dict, step: int):
    """记录标量指标（仅 rank-0）。"""
    if not _WANDB_AVAILABLE or not _is_main(): return
    try:
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception: pass


def _wandb_save_file(path, policy="now"):
    """上传文件到 wandb（仅 rank-0）。"""
    if not _WANDB_AVAILABLE or not _is_main():
        return
    try:
        if wandb.run is None:
            return
        fp = Path(path).resolve()
        if fp.exists():
            wandb.save(str(fp), policy=policy)
    except Exception:
        pass


def _wandb_log_images(model, val_list, stats, dev, epoch, n_storms=5, use_ode=True):
    """
    随机从验证集选 n_storms 个风暴生成对比图，上传到 wandb。
    仅 rank-0 调用。
    """
    if not _WANDB_AVAILABLE or not _is_main() or wandb.run is None: return
    if not val_list: return

    import matplotlib
    matplotlib.use('Agg')

    raw = _raw(model); raw.eval()
    n = min(n_storms, len(val_list))
    chosen = np.random.choice(len(val_list), n, replace=False)
    wandb_images = {}

    with torch.no_grad():
        for idx in chosen:
            ds = val_list[int(idx)]
            name = ds.get('hurricane', f'val_{idx}')
            tmp_path = os.path.join(tempfile.gettempdir(), f"wandb_ep{epoch}_{_sfn(name)}.png")
            try:
                vb = create_batch([ds], 1, dev, shuffle=False, spatial_stats=stats)
                sc = _mfilt(vb['scalars'].clone(), dev)
                pc = _pc(sc, vb) if use_ode else None
                # plot_mode=True：与 reference 公平对比（48h 跟踪 + ML 预报）
                out = raw(vb['spatial_3d'], vb['spatial_2d'], sc, vb['v_gt'][:, 0:1, :],
                          precomp=pc,
                          env_wnds=vb.get('env_wnds') if use_ode else None,
                          utran=vb.get('utran'), vtran=vb.get('vtran'), lats=vb.get('lats'),
                          use_run_fast_physics=use_ode,
                          plot_mode=use_ode)
                pv = out['v_max'] if out.get('v_max') is not None else \
                    axi_to_max_wind_torch(out['v'], out['s'], vb['env_wnds'],
                                          vb['utran'], vb['vtran'], vb['lats'])
                rv, rm = _run_fast_reference_batch(
                    vb['s_ref'], vb['chi_ref'], vb['xs_ref'], vb['scalars'].clone(),
                    vb['v_gt'], vb['env_wnds'], vb['utran'], vb['vtran'],
                    vb['lats'], vb.get('lons'))
                vfn = rv[0] if rv is not None else out['v'][0, :, 0].cpu().numpy()
                mfn = rm[0] if rm is not None else out['m'][0, :, 0].cpu().numpy()

                plot_comparison_one_sample(
                    vb['v_gt'][0, :, 0].cpu().numpy(), vb['s_ref'][0, :, 0].cpu().numpy(),
                    vb['chi_ref_eff'][0, :, 0].cpu().numpy(), pv[0, :, 0].cpu().numpy(),
                    out['s'][0, :, 0].cpu().numpy(), out['chi'][0, :, 0].cpu().numpy(),
                    out['xs'][0, :, 0].cpu().numpy(), out['m'][0, :, 0].cpu().numpy(),
                    vfn, mfn, vb['seq_len'][0].item(), name, tmp_path,
                    ds.get('m0_ref', np.array([[.5]]))[0, 0] if 'm0_ref' in ds else None,
                    ds.get('times'))

                wandb_images[f"val_plot/{_sfn(name)}"] = wandb.Image(tmp_path, caption=f"{name} ep{epoch}")
            except Exception as e:
                print(f"  W&B image error ({name}): {e}")

    if wandb_images:
        wandb.log(wandb_images, step=epoch)
    raw.train()


# ─────────────────────────────────────────────────────────────────────────────
# 通用 Phase 训练器
# ─────────────────────────────────────────────────────────────────────────────
def _phase(model, tr, va, stats, dev, rank, world,
           epochs, lr, grad_clip, bs, accum,
           use_ode, ws, wchi, wphys,
           name, log, mon_ds, mon_dir, mon_every, val_every, patience, ckpt,
           wadv=0.0, adv_margin=0.0, adv_warmup_frac=0.30,
           s_tr=True, chi_tr=True, enc_tr=True,
           time_weight_tau=0.0, huber_v_delta=0.0,
           warmup_epochs=0, weight_decay=1e-5,
           _wandb_step_offset=0):   # enc_tr=False 时冻结所有 CNN encoder，只微调 head

    S_DELTA, CHI_DELTA = 5.0, 0.5
    loss_s   = nn.HuberLoss(delta=S_DELTA)
    loss_chi = nn.HuberLoss(delta=CHI_DELTA)
    CHI_SC   = 50.0

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    # warmup 期间手动设 lr，cosine 仅在 warmup 之后接管，T_max 取 epochs - warmup
    sch_T_max = max(epochs - warmup_epochs, 1)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, sch_T_max, eta_min=lr*0.02)
    bv, ni, mon_pc, _vd = float('inf'), 0, None, ""
    best_key_name = "val_loss"

    if _is_main():
        adv_warmup_ep_disp = max(1, int(adv_warmup_frac * max(1, epochs)))
        print(f"\n{'='*55}\n  {name}: ep={epochs} lr={lr:.1e} ode={use_ode} ws={ws} wchi={wchi} wphys={wphys} "
              f"wadv={wadv} (ramp→{adv_warmup_ep_disp}ep) margin={adv_margin}\n  "
              f"warmup={warmup_epochs} wd={weight_decay:.0e} time_tau={time_weight_tau} huber_v_delta={huber_v_delta}\n{'='*55}")
        if use_ode:
            print("  Model selection key: val_v_mse (primary physics target)")
            best_key_name = "val_v_mse"
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        if not Path(log).exists():
            with open(log,'w') as f: f.write('phase,ep,lr,tot,Strain,Chitrain,Vtrain,Advtrain,Val\n')

    for ep in range(epochs):
        # ── LR warmup（线性 ramp from lr/100 → lr）─────────────────────────────
        if warmup_epochs > 0 and ep < warmup_epochs:
            warm_lr = lr * (0.01 + 0.99 * float(ep + 1) / float(warmup_epochs))
            for g in opt.param_groups: g['lr'] = warm_lr

        model.train(); _fix_bn(model, s_tr, chi_tr, enc_tr)
        opt.zero_grad()
        ttot=ts=tc=tv=ta=0.; n=0
        # wadv 线性升温：前 adv_warmup_frac × epoch 从 0 升到设定值，后续保持
        adv_warmup_ep = max(1, int(adv_warmup_frac * max(1, epochs)))
        wadv_eff = (wadv * min(1.0, float(ep + 1) / float(adv_warmup_ep))) if use_ode else 0.0

        for acc in range(accum):
            b = create_batch(tr, bs, dev, spatial_stats=stats, rank=rank, world=world)
            sc = _mfilt(b['scalars'].clone(), dev)
            # S/Chi 的监督掩码（包含 step_valid）
            vm = _vm(b, dev, use_tstart=use_ode)
            # V 的监督掩码（不依赖 step_valid，避免 chi/s 缺失牵连 v 监督）
            vm_v = b['valid_mask'].unsqueeze(-1)
            if use_ode and 't_start' in b:
                T = vm_v.shape[1]
                t_arr = torch.arange(T, device=dev).unsqueeze(0)
                ts_arr = b['t_start'].unsqueeze(1)
                vm_v = vm_v & (t_arr >= ts_arr).unsqueeze(-1)
            else:
                vm_v = vm_v & (torch.arange(vm_v.shape[1], device=dev).unsqueeze(0) >= LOSS_TRIM_STEPS).unsqueeze(-1)
            vm_v = vm_v & torch.isfinite(b['v_gt']) & (b['v_gt'] > 0)
            out = _raw(model)(b['spatial_3d'],b['spatial_2d'],sc,b['v_gt'][:,0:1,:],
                              precomp=None,env_wnds=b.get('env_wnds') if use_ode else None,
                              utran=b.get('utran'),vtran=b.get('vtran'),lats=b.get('lats'),
                              v_gt=b['v_gt'],use_run_fast_physics=use_ode,
                              f_init_end=b.get('f_init_end') if use_ode else None,
                              m_init=b.get('m_at_tstart') if use_ode else b.get('m0_ref'),
                              t_start=b.get('t_start'),
                              v_axisym_at_tstart=b.get('v_axisym_at_tstart') if use_ode else None)

            sm = vm & ~torch.isnan(b['s_ref'])   & (b['s_ref']   > 0)
            chi_ref_sup = b.get('chi_ref_eff', b['chi_ref'])
            cm = vm & ~torch.isnan(chi_ref_sup) & (chi_ref_sup > 0)
            ls = _safe(loss_s(out['s'][sm],   b['s_ref'][sm])   if sm.sum()>0 else torch.zeros(1,device=dev))
            lc = _safe(loss_chi(out['chi'][cm],chi_ref_sup[cm])*CHI_SC if cm.sum()>0 else torch.zeros(1,device=dev))
            lv = torch.zeros(1,device=dev)
            la = torch.zeros(1,device=dev)
            if use_ode and out.get('v_max') is not None:
                # ── V loss：可选 Huber + 时间衰减权重 (D2 端到端稳定化) ─────────────
                diff = out['v_max'] - b['v_gt']
                if huber_v_delta > 0:
                    absd = diff.abs()
                    per = torch.where(absd < huber_v_delta,
                                      0.5 * diff * diff,
                                      huber_v_delta * (absd - 0.5 * huber_v_delta))
                else:
                    per = diff * diff
                if time_weight_tau > 0:
                    Tlen = vm_v.shape[1]
                    t_idx = torch.arange(Tlen, device=dev, dtype=torch.float32).unsqueeze(0)
                    t_st  = b['t_start'].float().unsqueeze(1) if 't_start' in b else torch.zeros(vm_v.shape[0], 1, device=dev)
                    tau   = (t_idx - t_st).clamp(min=0)
                    w_t   = torch.exp(-tau / time_weight_tau).unsqueeze(-1)
                    msk   = vm_v.float() * w_t
                else:
                    msk   = vm_v.float()
                lv = _safe((per * msk).sum() / msk.sum().clamp(1))
                # 显式“超越 FAST”目标：若 ML 的 v-MSE 高于 FAST baseline，则产生惩罚
                if wadv_eff > 0:
                    rv_fast, _ = _run_fast_reference_batch(
                        b['s_ref'], b['chi_ref'], b['xs_ref'],
                        b['scalars'].clone(), b['v_gt'], b['env_wnds'],
                        b['utran'], b['vtran'], b['lats'], b.get('lons'))
                    if rv_fast is not None:
                        v_fast = torch.from_numpy(rv_fast.astype(np.float32)).unsqueeze(-1).to(dev)
                        lv_fast = _safe(((v_fast-b['v_gt'])**2*vm_v.float()).sum()/vm_v.float().sum().clamp(1))
                        la = F.relu(lv - lv_fast + adv_margin)
            tot = _safe(ws*ls + wchi*lc + wphys*lv + wadv_eff*la) / accum
            if torch.isfinite(tot) and tot.item() < 1e7/accum:
                import contextlib
                _ctx = (model.no_sync() if isinstance(model, DDP) and acc < accum - 1
                        else contextlib.nullcontext())
                with _ctx:
                    tot.backward()
            ttot+=tot.item()*accum; ts+=ls.item(); tc+=lc.item(); tv+=lv.item(); ta+=la.item(); n+=1

        gn = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        if torch.isfinite(gn): opt.step()
        else: opt.zero_grad()
        opt.zero_grad()
        # warmup 期间不让 cosine scheduler 推进（手动设的 lr 由下个 ep 头部覆盖）
        if ep >= warmup_epochs:
            sch.step()

        vl = float('nan')
        sel_key = float('inf')
        val_v_mse = float('nan')
        val_fast_mse = float('nan')
        val_gain_vs_fast_mse = float('nan')
        if va and (ep+1)%val_every==0:
            _raw(model).eval()
            with torch.no_grad():
                vb=create_batch(va,len(va),dev,shuffle=False,spatial_stats=stats)
                sv=_mfilt(vb['scalars'].clone(),dev)
                pvc=_pc(sv,vb) if use_ode else None
                ov=_raw(model)(vb['spatial_3d'],vb['spatial_2d'],sv,vb['v_gt'][:,0:1,:],
                               precomp=None,env_wnds=vb.get('env_wnds') if use_ode else None,
                               utran=vb.get('utran'),vtran=vb.get('vtran'),lats=vb.get('lats'),use_run_fast_physics=use_ode,
                               f_init_end=vb.get('f_init_end') if use_ode else None,
                               m_init=vb.get('m_at_tstart') if use_ode else vb.get('m0_ref'),
                               t_start=vb.get('t_start'),
                               v_axisym_at_tstart=vb.get('v_axisym_at_tstart') if use_ode else None)
                # val loss 同样按 t_start 截断，与训练 loss 对齐
                vmv=_vm(vb, dev, use_tstart=use_ode)
                vmv_v = vb['valid_mask'].unsqueeze(-1)
                if use_ode and 't_start' in vb:
                    T = vmv_v.shape[1]
                    t_arr = torch.arange(T, device=dev).unsqueeze(0)
                    ts_arr = vb['t_start'].unsqueeze(1)
                    vmv_v = vmv_v & (t_arr >= ts_arr).unsqueeze(-1)
                else:
                    vmv_v = vmv_v & (torch.arange(vmv_v.shape[1], device=dev).unsqueeze(0) >= LOSS_TRIM_STEPS).unsqueeze(-1)
                vmv_v = vmv_v & torch.isfinite(vb['v_gt']) & (vb['v_gt'] > 0)
                sm2=vmv&(vb['s_ref']>0)&~torch.isnan(vb['s_ref'])
                chi_ref_sup_v = vb.get('chi_ref_eff', vb['chi_ref'])
                cm2=vmv&(chi_ref_sup_v>0)&~torch.isnan(chi_ref_sup_v)
                vls=loss_s(ov['s'][sm2],vb['s_ref'][sm2]) if sm2.sum()>0 else torch.zeros(1,device=dev)
                vlc=loss_chi(ov['chi'][cm2],chi_ref_sup_v[cm2])*CHI_SC if cm2.sum()>0 else torch.zeros(1,device=dev)
                vl=(ws*vls + wchi*vlc).item()
                vla = torch.zeros(1, device=dev)
                if use_ode and wphys>0 and ov.get('v_max') is not None:
                    vl_v = ((ov['v_max']-vb['v_gt'])**2*vmv_v.float()).sum()/vmv_v.float().sum().clamp(1)
                    val_v_mse = float(vl_v.item())
                    vl += wphys * vl_v.item()
                    if wadv_eff > 0:
                        rv_fast_v, _ = _run_fast_reference_batch(
                            vb['s_ref'], vb['chi_ref'], vb['xs_ref'],
                            vb['scalars'].clone(), vb['v_gt'], vb['env_wnds'],
                            vb['utran'], vb['vtran'], vb['lats'], vb.get('lons'))
                        if rv_fast_v is not None:
                            v_fast_v = torch.from_numpy(rv_fast_v.astype(np.float32)).unsqueeze(-1).to(dev)
                            vl_fast_v = ((v_fast_v-vb['v_gt'])**2*vmv_v.float()).sum()/vmv_v.float().sum().clamp(1)
                            val_fast_mse = float(vl_fast_v.item())
                            val_gain_vs_fast_mse = float((vl_fast_v - vl_v).item())
                            vla = F.relu(vl_v - vl_fast_v + adv_margin)
                            vl += wadv_eff * vla.item()
                _vd = f"S:{vls.item():.3f} Chi:{vlc.item():.3f} Adv:{vla.item():.3f}"
                if use_ode and ov.get('v_max') is not None:
                    _vd += f" V:{((ov['v_max']-vb['v_gt'])**2*vmv_v.float()).sum().item()/vmv_v.float().sum().clamp(1).item():.3f}"
                # D 阶段以风速物理目标优先：保存/早停依据 val_v_mse
                if use_ode and np.isfinite(val_v_mse):
                    sel_key = val_v_mse
                else:
                    sel_key = vl
            model.train(); _fix_bn(model, s_tr, chi_tr)
            if np.isfinite(sel_key) and sel_key < bv:
                if _is_main():
                    bv, ni = sel_key, 0; save_ckpt(model, ckpt, upload_to_wandb=True)
                    print(f"  -> best {best_key_name}={bv:.6f} ep={ep+1}")
                else:
                    bv, ni = sel_key, 0
            else:
                ni += 1

        clr = sch.get_last_lr()[0]
        global_step = _wandb_step_offset + ep + 1

        # ── monitor 图（本地 + wandb）────────────────────────────────────────
        if _is_main() and ((ep+1)%mon_every==0 or ep==0):
            mon_pc = _monitor(_raw(model), mon_ds, stats, dev, ep+1, mon_dir, mon_pc)
            # 每 mon_every 步从验证集随机选 5 个上传 wandb
            _wandb_log_images(_raw(model), va, stats, dev, global_step,
                              n_storms=5, use_ode=use_ode)

        if _is_main():
            vs = f" | Val={vl:.4f} ({_vd})" if np.isfinite(vl) else ""
            print(f"[{name}] {ep+1:04d} lr:{clr:.1e} wadv_eff:{wadv_eff:.3f} | Tot:{ttot/n:.3f} S:{ts/n:.3f} Chi:{tc/n:.3f} V:{tv/n:.3f} Adv:{ta/n:.3f}{vs}")
            _vlstr = f"{vl:.4f}" if np.isfinite(vl) else "nan"
            with open(log,'a') as f:
                f.write(f"{name},{ep+1},{clr:.2e},{ttot/n:.4f},{ts/n:.4f},{tc/n:.4f},{tv/n:.4f},{ta/n:.4f},{_vlstr}\n")

            # ── wandb 标量日志 ────────────────────────────────────────────────
            _wlog = {
                f"{name}/train_loss_S":     ts/n,
                f"{name}/train_loss_Chi":   tc/n,
                f"{name}/train_loss_V":     tv/n,
                f"{name}/train_loss_Adv":   ta/n,
                f"{name}/train_loss_total": ttot/n,
                f"{name}/wadv_eff":         wadv_eff,
                f"{name}/lr":               clr,
                f"{name}/phase":            name,
            }
            if np.isfinite(vl):
                _wlog[f"{name}/val_loss"] = vl
                if np.isfinite(val_v_mse):
                    _wlog[f"{name}/val_v_mse"] = val_v_mse
                    _wlog[f"{name}/val_v_rmse_kts"] = (val_v_mse ** 0.5) * MS_TO_KNOTS
                if np.isfinite(val_fast_mse):
                    _wlog[f"{name}/val_fast_mse"] = val_fast_mse
                    _wlog[f"{name}/val_fast_rmse_kts"] = (val_fast_mse ** 0.5) * MS_TO_KNOTS
                if np.isfinite(val_gain_vs_fast_mse):
                    _wlog[f"{name}/val_gain_vs_fast_mse"] = val_gain_vs_fast_mse
                    _wlog[f"{name}/val_gain_vs_fast_rmse_kts"] = (
                        ((val_fast_mse ** 0.5) - (val_v_mse ** 0.5)) * MS_TO_KNOTS
                        if np.isfinite(val_fast_mse) and np.isfinite(val_v_mse) else float('nan')
                    )
                # 分项 val 指标
                if 'S:' in _vd:
                    try:
                        _wlog[f"{name}/val_loss_S"]   = float(_vd.split('S:')[1].split()[0])
                        _wlog[f"{name}/val_loss_Chi"]  = float(_vd.split('Chi:')[1].split()[0])
                        _wlog[f"{name}/val_loss_Adv"]  = float(_vd.split('Adv:')[1].split()[0])
                        if 'V:' in _vd:
                            _wlog[f"{name}/val_loss_V"] = float(_vd.split('V:')[1].split()[0])
                    except: pass
            _wandb_log(_wlog, step=global_step)

        # 广播 early stop 信号，确保所有 DDP rank 同步退出，避免死锁
        _stop = torch.tensor(1 if (ni >= patience and ep >= 20) else 0,
                             dtype=torch.long, device=dev)
        if dist.is_initialized():
            dist.broadcast(_stop, src=0)
        if _stop.item():
            if _is_main(): print(f"  Early stop {name} ep={ep+1}")
            break

    load_ckpt(model, ckpt, dev)
    # 同步 bv 到所有 rank（DDP 下仅 rank-0 更新过 bv）
    if dist.is_initialized():
        buf = [bv]; dist.broadcast_object_list(buf, src=0); bv = buf[0]
    return bv

# ─────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────────────────────
def train_twostream(data_dir='training_data', seq_len=TARGET_SEQ_LEN,
                    min_vmax_kts=MIN_VMAX_KTS, start_from=START_FROM_VMAX_KTS,
                    min_duration_h=MIN_DURATION_H,
                    train_years=None, val_years=None, test_years=None,
                    storm_include=None, storm_exclude=None,
                    epochs_a=150, epochs_b=150, epochs_d1=200, epochs_d2=100,
                    lr_ab=1e-3, lr_d=3e-4, lr_d2=5e-5, grad_clip=1.0,
                    bs=4, accum=4, val_every=1, mon_every=20, patience=25,
                    skip_final_eval=False, d_only=False,
                    d2_ws=None, d2_wchi=None, d2_wphys=None, d2_wadv=0.0,
                    d2_time_weight_tau=0.0, d2_huber_v_delta=0.0,
                    d2_warmup=0, d2_weight_decay=1e-5):

    dev, rank, world = _setup_ddp()
    torch.manual_seed(42+rank); np.random.seed(42+rank)
    if _is_main(): print(f"{world} GPU(s), eff_batch={bs*accum*world}")

    tr, va, te = load_1km_storms(data_dir, seq_len, min_vmax_kts, start_from,
                                 min_duration_h,
                                 train_years, val_years, test_years,
                                 storm_include, storm_exclude)
    if not tr:
        if _is_main(): print("No data. Run batch_prepare_spatial_1000km.py first.")
        return

    stats = compute_spatial_stats(tr) if _is_main() else None
    if dist.is_initialized():
        import pickle as pk
        buf = [pk.dumps(stats)]; dist.broadcast_object_list(buf, src=0); stats = pk.loads(buf[0])

    if _is_main():
        for vi,vn in enumerate(['T','Q','U','V','Z']):
            lev_hpa = [1000,850,700,600,500,250,200]
            rows = []
            for li in range(N_LEVELS):
                mu,std = stats['3d'][vi][li]
                fmt = ".2e" if 0<abs(mu)<0.1 else ".2f"
                rows.append(f"{lev_hpa[li]}hPa μ={mu:{fmt}}")
            print(f"  {vn}: " + "  ".join(rows))

    mon_ds = (va if va else tr)[0]
    if _is_main(): print(f"Monitor: {mon_ds.get('hurricane','?')}")

    # 模型（优先从最近 checkpoint 恢复）
    raw = TwoStreamFASTModel().to(dev)
    ck_order = ['ckpt_phase_d1.pth','ckpt_phase_d.pth','ckpt_phase_b.pth','ckpt_phase_a.pth'] if d_only \
        else ['ckpt_phase_d.pth','ckpt_phase_b.pth','ckpt_phase_a.pth']
    for ck in ck_order:
        if load_ckpt(raw, ck, dev): break

    model = DDP(raw, device_ids=[dev.index], find_unused_parameters=True) \
        if dist.is_initialized() else (nn.DataParallel(raw) if torch.cuda.device_count()>1 else raw)

    log_dir = 'training_logs'
    mon_dir = os.path.join(log_dir, 'monitor_plots')
    os.makedirs(mon_dir, exist_ok=True)
    log = os.path.join(log_dir, 'phases.csv')

    # ── 初始化 W&B（仅 rank-0）───────────────────────────────────────────────
    _wandb_init({
        'train_years':    str(train_years or TRAIN_YEARS),
        'val_years':      str(val_years or VAL_YEARS),
        'test_years':     str(test_years or TEST_YEARS),
        'seq_len':        seq_len,
        'min_vmax_kts':   min_vmax_kts,
        'min_duration_h': min_duration_h,
        'n_train':        len(tr),
        'n_val':          len(va),
        'n_test':         len(te),
        'epochs_a':       epochs_a,
        'epochs_b':       epochs_b,
        'epochs_d1':      epochs_d1,
        'epochs_d2':      epochs_d2,
        'lr_ab':          lr_ab,
        'lr_d':           lr_d,
        'lr_d2':          lr_d2,
        'grad_clip':      grad_clip,
        'eff_batch':      bs * accum * world,
        'architecture':   'TwoStream-MLP-4phase',
        'pool':           'AvgMaxPool(3D:4x2x2,2D:2x2)+extreme-aware',
    })

    kw = dict(stats=stats, dev=dev, rank=rank, world=world, grad_clip=grad_clip,
              bs=bs, accum=accum, mon_ds=mon_ds, mon_dir=mon_dir, mon_every=mon_every,
              val_every=val_every, patience=patience, log=log, tr=tr, va=va)

    if not d_only:
        # ── Phase A: 只训练 S 流 ────────────────────────────────────────────
        set_stream_trainable(model, s=True, chi=False)
        _phase(model, epochs=epochs_a, lr=lr_ab, use_ode=False,
               ws=1., wchi=0., wphys=0., name='PhaseA_S', ckpt='ckpt_phase_a.pth',
               s_tr=True, chi_tr=False, _wandb_step_offset=0, **kw)

        # ── Phase B: 只训练 Chi 流 ──────────────────────────────────────────
        set_stream_trainable(model, s=False, chi=True)
        _phase(model, epochs=epochs_b, lr=lr_ab, use_ode=False,
               ws=0., wchi=1., wphys=0., name='PhaseB_Chi', ckpt='ckpt_phase_b.pth',
               s_tr=False, chi_tr=True, _wandb_step_offset=epochs_a, **kw)

        # ── Phase C: ODE 物理完整性验证 ─────────────────────────────────────
        if _is_main():
            print("\n" + "="*55 + "\n  Phase C: ODE Sanity Check\n" + "="*55)
            set_stream_trainable(model, s=False, chi=False); raw.eval()
            phc = os.path.join(mon_dir,'phase_c'); os.makedirs(phc, exist_ok=True)
            with torch.no_grad():
                for i, ds in enumerate((te or va or tr)[:3]):
                    _monitor(raw, ds, stats, dev, 9000+i, phc)
            print("  Phase C → training_logs/monitor_plots/phase_c/")
    elif _is_main():
        print("\n" + "="*55 + "\n  D-only mode: skip Phase A/B/C, run Phase D only\n" + "="*55)

    # ── Phase D1: 冻结 CNN，只微调 head_s + head_chi → ODE ──────────────────
    # CNN 特征提取器保持 A/B 学到的能力；head 适配 ODE 物理约束
    set_stream_trainable(model, s=True, chi=True, enc=False)
    best_val_d1 = _phase(model, epochs=epochs_d1, lr=lr_d, use_ode=True,
                         ws=0.10, wchi=0.10, wphys=1.0, wadv=0.00, adv_margin=0.0,
                         name='PhaseD1_HeadODE', ckpt='ckpt_phase_d1.pth',
                         s_tr=True, chi_tr=True, enc_tr=False,
                         _wandb_step_offset=epochs_a+epochs_b, **kw)

    # ── Phase D2: 全解冻，极小学习率端到端收尾 ───────────────────────────────
    # CNN 与 head 一起做轻微调整，让特征空间最终对齐 ODE 物理
    set_stream_trainable(model, s=True, chi=True, enc=True)
    _ws_d2    = 0.03 if d2_ws    is None else d2_ws
    _wchi_d2  = 0.03 if d2_wchi  is None else d2_wchi
    _wphys_d2 = 1.0  if d2_wphys is None else d2_wphys
    best_val_d2 = _phase(model, epochs=epochs_d2, lr=lr_d2, use_ode=True,
                         ws=_ws_d2, wchi=_wchi_d2, wphys=_wphys_d2,
                         wadv=d2_wadv, adv_margin=0.0,
                         time_weight_tau=d2_time_weight_tau,
                         huber_v_delta=d2_huber_v_delta,
                         warmup_epochs=d2_warmup,
                         weight_decay=d2_weight_decay,
                         name='PhaseD2_FullODE', ckpt='ckpt_phase_d.pth',
                         s_tr=True, chi_tr=True, enc_tr=True,
                         _wandb_step_offset=epochs_a+epochs_b+epochs_d1, **kw)
    if _is_main():
        print(f"Best val: D1={best_val_d1:.4f} | D2={best_val_d2:.4f}")
    _wandb_log({
        "phase_summary/best_val_d1": float(best_val_d1),
        "phase_summary/best_val_d2": float(best_val_d2),
    }, step=epochs_a+epochs_b+epochs_d1+epochs_d2)

    # 同时保留 D1 与 D2 各自最佳权重
    # - ckpt_phase_d1.pth: D1 best
    # - ckpt_phase_d.pth : D2 best
    # 同时导出两个 final 文件，避免覆盖导致回溯困难
    if _is_main():
        load_ckpt(model, 'ckpt_phase_d1.pth', dev)
        save_ckpt(model, 'twostream_final_d1.pth', upload_to_wandb=True)
        load_ckpt(model, 'ckpt_phase_d.pth', dev)
        save_ckpt(model, 'twostream_final_d2.pth', upload_to_wandb=True)
        # 默认 final 指向 D2（保持原训练流程习惯）
        save_ckpt(model, 'twostream_final.pth', upload_to_wandb=True)
        print(f"Saved twostream_final_d1.pth / twostream_final_d2.pth / twostream_final.pth")

    # 所有 rank 在最终评估前同步，避免 rank 提前退出导致收尾异常
    if dist.is_initialized():
        dist.barrier()

    # ── 统一评估：验证集+测试集，分别评估 D1(A) / D2(B) / FAST baseline ───────────
    if _is_main() and (not skip_final_eval):
        eval_results = {}
        def _eval_split_with_ckpt(split_name, ds_list, ckpt_path, model_tag):
            if not ds_list:
                return
            if not load_ckpt(raw, ckpt_path, dev):
                print(f"[{split_name}/{model_tag}] skip (missing ckpt: {ckpt_path})")
                return
            raw.eval()
            with torch.no_grad():
                b = create_batch(ds_list, len(ds_list), dev, shuffle=False, spatial_stats=stats)
                sc = _mfilt(b['scalars'].clone(), dev)
                pc = _pc(sc, b)
                # 评估用 plot_mode=True：48h 跟踪 + ML 预报
                p = raw(b['spatial_3d'], b['spatial_2d'], sc, b['v_gt'][:, 0:1, :],
                        precomp=pc, env_wnds=b.get('env_wnds'), utran=b.get('utran'),
                        vtran=b.get('vtran'), lats=b.get('lats'),
                        use_run_fast_physics=True, plot_mode=True)
                pv = p['v_max'] if p.get('v_max') is not None else \
                     axi_to_max_wind_torch(p['v'], p['s'], b['env_wnds'], b['utran'], b['vtran'], b['lats'])

                vm = b['valid_mask'].unsqueeze(-1)
                if 'step_valid' in b:
                    vm = vm & b['step_valid'].unsqueeze(-1)
                ml_mse = ((pv - b['v_gt'])**2 * vm.float()).sum() / vm.float().sum().clamp(1)
                ml_rmse_kts = float(ml_mse.sqrt().item() * MS_TO_KNOTS)

                rv, rm = _run_fast_reference_batch(
                    b['s_ref'], b['chi_ref'], b['xs_ref'], b['scalars'].clone(), b['v_gt'],
                    b['env_wnds'], b['utran'], b['vtran'], b['lats'], b.get('lons')
                )
                if rv is not None:
                    vr = torch.from_numpy(rv.astype(np.float32)).unsqueeze(-1).to(dev)
                    ph_mse = ((vr - b['v_gt'])**2 * vm.float()).sum() / vm.float().sum().clamp(1)
                    ph_rmse_kts = float(ph_mse.sqrt().item() * MS_TO_KNOTS)
                else:
                    vr = pv
                    ph_rmse_kts = float('nan')
                mr = torch.from_numpy(rm.astype(np.float32)).unsqueeze(-1).to(dev) if rm is not None else p['m']

                out_dir = f"eval_{split_name}_twostream_{model_tag.lower()}"
                os.makedirs(out_dir, exist_ok=True)
                for i in range(len(ds_list)):
                    plot_comparison_one_sample(
                        b['v_gt'][i,:,0].cpu().numpy(), b['s_ref'][i,:,0].cpu().numpy(),
                        b['chi_ref_eff'][i,:,0].cpu().numpy(), pv[i,:,0].cpu().numpy(),
                        p['s'][i,:,0].cpu().numpy(), p['chi'][i,:,0].cpu().numpy(),
                        p['xs'][i,:,0].cpu().numpy(), p['m'][i,:,0].cpu().numpy(),
                        vr[i,:,0].cpu().numpy(), mr[i,:,0].cpu().numpy(),
                        b['seq_len'][i].item(), ds_list[i].get('hurricane', f'{split_name}_{i}'),
                        os.path.join(out_dir, _sfn(ds_list[i].get('hurricane', f'{split_name}_{i}')) + '.png'),
                        b['m0_ref'][i,0].item() if 'm0_ref' in b else None,
                        b['times'][i] if 'times' in b else None
                    )
                print(f"[{split_name}/{model_tag}] ML RMSE={ml_rmse_kts:.2f} kts | FAST RMSE={ph_rmse_kts:.2f} kts")
                print(f"Saved {len(ds_list)} plots → {out_dir}/")
                gain = ph_rmse_kts - ml_rmse_kts if np.isfinite(ph_rmse_kts) else float('nan')
                eval_results[(split_name, model_tag)] = {
                    "ml_rmse_kts": ml_rmse_kts,
                    "fast_rmse_kts": ph_rmse_kts,
                    "gain_vs_fast_kts": gain,
                }

                if _WANDB_AVAILABLE and wandb.run is not None:
                    w = {
                        f"{split_name}/{model_tag}/ml_rmse_kts": ml_rmse_kts,
                        f"{split_name}/{model_tag}/fast_rmse_kts": ph_rmse_kts,
                        f"{split_name}/{model_tag}/gain_vs_fast_kts": gain,
                    }
                    # 上传前 5 张图，避免日志过重
                    n_img = min(5, len(ds_list))
                    for i in range(n_img):
                        sid = _sfn(ds_list[i].get('hurricane', f'{split_name}_{i}'))
                        w[f"{split_name}/{model_tag}/plot_{sid}"] = wandb.Image(
                            os.path.join(out_dir, sid + '.png'))
                    wandb.log(w)

        eval_splits = [('val', va), ('test', te)]
        eval_models = [('A_D1', 'ckpt_phase_d1.pth'), ('B_D2', 'ckpt_phase_d.pth')]
        for sp_name, sp_ds in eval_splits:
            for tag, ck in eval_models:
                _eval_split_with_ckpt(sp_name, sp_ds, ck, tag)

        # 自动判定：是否超越 FAST，并给出 A/B 推荐
        print("\n=== Final Eval Summary (vs FAST baseline) ===")
        for sp_name, _ in eval_splits:
            a = eval_results.get((sp_name, 'A_D1'))
            b2 = eval_results.get((sp_name, 'B_D2'))
            if a is None and b2 is None:
                continue
            print(f"[{sp_name}]")
            if a is not None:
                ok_a = "YES" if np.isfinite(a["gain_vs_fast_kts"]) and a["gain_vs_fast_kts"] > 0 else "NO"
                print(f"  A_D1: ML={a['ml_rmse_kts']:.2f}, FAST={a['fast_rmse_kts']:.2f}, "
                      f"gain={a['gain_vs_fast_kts']:.2f} kts, beat_fast={ok_a}")
            if b2 is not None:
                ok_b = "YES" if np.isfinite(b2["gain_vs_fast_kts"]) and b2["gain_vs_fast_kts"] > 0 else "NO"
                print(f"  B_D2: ML={b2['ml_rmse_kts']:.2f}, FAST={b2['fast_rmse_kts']:.2f}, "
                      f"gain={b2['gain_vs_fast_kts']:.2f} kts, beat_fast={ok_b}")

            if (a is not None) and (b2 is not None):
                best_tag = 'A_D1' if a['ml_rmse_kts'] <= b2['ml_rmse_kts'] else 'B_D2'
                best_gain = a['gain_vs_fast_kts'] if best_tag == 'A_D1' else b2['gain_vs_fast_kts']
                print(f"  Recommend: {best_tag} (gain_vs_fast={best_gain:.2f} kts)")

        # 将评估结果落盘，便于明确对比 2024 test 与 FAST baseline
        os.makedirs('training_logs', exist_ok=True)
        eval_csv = os.path.join('training_logs', 'eval_summary.csv')
        with open(eval_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['split', 'model_tag', 'ml_rmse_kts', 'fast_rmse_kts', 'gain_vs_fast_kts', 'beat_fast'])
            for (sp, tag), met in sorted(eval_results.items()):
                gain = met['gain_vs_fast_kts']
                beat = int(np.isfinite(gain) and gain > 0)
                w.writerow([sp, tag, f"{met['ml_rmse_kts']:.4f}", f"{met['fast_rmse_kts']:.4f}", f"{gain:.4f}", beat])

        test_csv = os.path.join('training_logs', 'test_2024_vs_fast.csv')
        with open(test_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['year', 'model_tag', 'ml_rmse_kts', 'fast_rmse_kts', 'gain_vs_fast_kts', 'beat_fast'])
            for tag in ['A_D1', 'B_D2']:
                met = eval_results.get(('test', tag))
                if met is None:
                    continue
                gain = met['gain_vs_fast_kts']
                beat = int(np.isfinite(gain) and gain > 0)
                w.writerow([2024, tag, f"{met['ml_rmse_kts']:.4f}", f"{met['fast_rmse_kts']:.4f}", f"{gain:.4f}", beat])
        print(f"Saved eval summaries → {eval_csv} / {test_csv}")

        if _WANDB_AVAILABLE and wandb.run is not None:
            _wandb_save_file(eval_csv, policy="now")
            _wandb_save_file(test_csv, policy="now")

    # 最终评估后再次同步，确保所有 rank 一致退出
    if dist.is_initialized():
        dist.barrier()

    # ── 关闭 wandb ────────────────────────────────────────────────────────────
    if _is_main() and _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

    if dist.is_initialized(): dist.destroy_process_group()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TwoStream FAST: PhaseA(S)→B(Chi)→C(check)→D(ODE)')
    p.add_argument('--data_dir',      default='training_data')
    p.add_argument('--seq_len',       type=int,   default=TARGET_SEQ_LEN)
    p.add_argument('--min_vmax_kts',   type=float, default=MIN_VMAX_KTS,
                   help='峰值强度过滤阈值（kts），低于此值的风暴排除')
    p.add_argument('--min_duration_h', type=int,   default=MIN_DURATION_H,
                   help='最短有效持续时长（小时），低于此值的风暴排除')
    p.add_argument('--start_from',    type=float, default=START_FROM_VMAX_KTS)
    p.add_argument('--train_years',   default='')
    p.add_argument('--val_years',     default='')
    p.add_argument('--test_years',    default='')
    p.add_argument('--storm_include', default='')
    p.add_argument('--storm_exclude', default='storm_exclude.txt')
    p.add_argument('--epochs_a',      type=int,   default=150)
    p.add_argument('--epochs_b',      type=int,   default=150)
    p.add_argument('--epochs_d1',     type=int,   default=200,
                   help='Phase D1：冻结 CNN 只微调 head（头部适配 ODE）')
    p.add_argument('--epochs_d2',     type=int,   default=100,
                   help='Phase D2：全解冻极小 lr 端到端收尾')
    p.add_argument('--lr_ab',         type=float, default=1e-3)
    p.add_argument('--lr_d',          type=float, default=3e-4,
                   help='Phase D1 学习率（只训练 head）')
    p.add_argument('--lr_d2',         type=float, default=5e-5,
                   help='Phase D2 学习率（全模型微调，应远小于 lr_d）')
    p.add_argument('--grad_clip',     type=float, default=1.0)
    p.add_argument('--bs',            type=int,   default=6,   help='每GPU batch size')
    p.add_argument('--accum',         type=int,   default=4,   help='梯度累加步数')
    p.add_argument('--val_every',     type=int,   default=1)
    p.add_argument('--mon_every',     type=int,   default=20)
    p.add_argument('--patience',      type=int,   default=25)
    p.add_argument('--skip_final_eval', action='store_true',
                   help='跳过训练结束后的统一评估（建议用独立评估脚本另跑，避免作业收尾崩溃）')
    p.add_argument('--d_only', action='store_true',
                   help='仅运行 Phase D 微调（跳过 A/B/C）')
    # ── D2 纯 V_MSE 端到端稳定化参数 ───────────────────────────────────────────
    p.add_argument('--ws_d2',       type=float, default=None, help='D2 ws override (default 0.03)')
    p.add_argument('--wchi_d2',     type=float, default=None, help='D2 wchi override (default 0.03)')
    p.add_argument('--wphys_d2',    type=float, default=None, help='D2 wphys override (default 1.0)')
    p.add_argument('--wadv_d2',     type=float, default=0.0,  help='D2 wadv (default 0)')
    p.add_argument('--time_weight_tau_d2', type=float, default=0.0,
                   help='D2 V loss 时间衰减权重 tau（小时，0=均匀），exp(-(t-t_start)/tau)')
    p.add_argument('--huber_v_delta_d2',   type=float, default=0.0,
                   help='D2 V loss Huber delta（m/s，0=纯 MSE），限制单点梯度幅度')
    p.add_argument('--warmup_d2',          type=int,   default=0,
                   help='D2 LR warmup epoch 数（线性 ramp lr*0.01 → lr）')
    p.add_argument('--weight_decay_d2',    type=float, default=1e-5,
                   help='D2 weight decay（正则化，防 chi/s 漂移）')
    args = p.parse_args()

    def _py(s): return [int(x) for x in s.split(',') if x.strip()] if s.strip() else None

    train_twostream(
        data_dir=args.data_dir, seq_len=args.seq_len,
        min_vmax_kts=args.min_vmax_kts, start_from=args.start_from,
        min_duration_h=args.min_duration_h,
        train_years=_py(args.train_years), val_years=_py(args.val_years),
        test_years=_py(args.test_years),
        storm_include=args.storm_include.strip() or None,
        storm_exclude=args.storm_exclude.strip() or None,
        epochs_a=args.epochs_a, epochs_b=args.epochs_b,
        epochs_d1=args.epochs_d1, epochs_d2=args.epochs_d2,
        lr_ab=args.lr_ab, lr_d=args.lr_d, lr_d2=args.lr_d2, grad_clip=args.grad_clip,
        bs=args.bs, accum=args.accum,
        val_every=args.val_every, mon_every=args.mon_every, patience=args.patience,
        skip_final_eval=args.skip_final_eval, d_only=args.d_only,
        d2_ws=args.ws_d2, d2_wchi=args.wchi_d2, d2_wphys=args.wphys_d2, d2_wadv=args.wadv_d2,
        d2_time_weight_tau=args.time_weight_tau_d2,
        d2_huber_v_delta=args.huber_v_delta_d2,
        d2_warmup=args.warmup_d2,
        d2_weight_decay=args.weight_decay_d2)