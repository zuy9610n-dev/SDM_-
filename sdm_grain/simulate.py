"""粒度条件から合成 EDS 点分析データを生成する簡略シミュレータ。

実データ取得前に「粒度をどう振ると組成空間カバレッジや相互拡散帯がどう変わるか」を
定性的に検討するためのもの。物理モデルは意図的に簡略化している:

1. **粒子配置** : 各元素粉末を、粒径 D50 と体積比に応じた個数だけ視野内にランダム配置し、
   加法重み付き Voronoi (power diagram) で粒界を決める。粒径の大きい粉末は大きなセルを得る。
2. **拡散**     : 各元素の指示関数を、実効拡散距離 L_i = 2√(Σ D_i t) に相当する
   Gaussian で平滑化する (Fick 拡散の Green 関数近似)。元素ごとの相対移動度
   (Al, Cu が速く Ti, Fe が遅い) と、微粉ほど接触面積が増えて焼結接合が進む効果
   (contact factor) を考慮する。
3. **化合物形成**: 相互拡散帯の組成を、系内で予想される化合物相の代表組成へ
   `phase_attraction` の強さで引き寄せ、点密度のピーク (相候補) を作る。
4. **測定**     : 視野内ランダム位置で組成を読み取り、EDS 定量ノイズを加える。

*出力は実験計画の目安であり、定量的な状態図予測ではない。*
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .config import (
    DEFAULT_D0_M2_S,
    DEFAULT_Q_J_MOL,
    GrainCondition,
    SimulationConfig,
)

# 元素ごとの相対移動度 (Al マトリクス中の拡散係数のオーダー差を粗く反映)
DEFAULT_MOBILITY: Dict[str, float] = {"Al": 1.0, "Cu": 0.9, "Si": 0.6, "Fe": 0.3, "Ti": 0.2}

# 拡散長の見積もりに使う Arrhenius 既定値 (config の値より活性化エネルギーを下げ、
# 焼結温度 ~870 K で拡散長が数〜十数 µm となるよう調整)
SIM_D0_M2_S: float = DEFAULT_D0_M2_S
SIM_Q_J_MOL: float = 1.7e5


@dataclass
class SimulationResult:
    condition: GrainCondition
    points: pd.DataFrame                 # x_um, y_um, 元素列
    grain_map: np.ndarray                # (n_pix, n_pix) 焼結前の元素 index
    comp_grid: np.ndarray                # (n_el, n_pix, n_pix) 拡散後の組成 [at%]
    pixel_um: float
    diffusion_length_um: Dict[str, float]
    meta: Dict[str, float] = field(default_factory=dict)


def _n_particles(cond: GrainCondition, cfg: SimulationConfig, elements: Sequence[str]) -> Dict[str, int]:
    """視野を埋める粒子個数を元素別に見積もる。個数 ∝ 体積比 / D50²。"""
    vf = cfg.volume_fractions or {el: 1.0 for el in elements}
    tot_vf = sum(vf[el] for el in elements)
    area = cfg.field_size_um ** 2
    counts: Dict[str, int] = {}
    for el in elements:
        frac = vf[el] / tot_vf
        d = cond.d50_um[el]
        n = frac * area / (math.pi * (d / 2.0) ** 2) * cfg.n_grains_scale
        counts[el] = max(1, int(round(n)))
    return counts


def _power_voronoi_grain_map(
    cond: GrainCondition,
    cfg: SimulationConfig,
    elements: Sequence[str],
    n_pix: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """加法重み付き Voronoi で粒子配置を作る。戻り値は元素 index の 2D 配列。"""
    counts = _n_particles(cond, cfg, elements)
    seeds: List[np.ndarray] = []
    radii: List[float] = []
    labels: List[int] = []
    for i, el in enumerate(elements):
        n = counts[el]
        d50 = cond.d50_um[el]
        span = cond.span.get(el, 1.0)
        # 対数正規分布で粒径をばらつかせる (span → σ_ln)
        sigma_ln = max(0.05, span / 2.5)
        diam = d50 * np.exp(rng.normal(0.0, sigma_ln, n) - sigma_ln ** 2 / 2)
        pos = rng.uniform(0.0, cfg.field_size_um, size=(n, 2))
        seeds.append(pos)
        radii.extend((diam / 2.0).tolist())
        labels.extend([i] * n)
    S = np.vstack(seeds)                 # (M, 2)
    R = np.asarray(radii)                # (M,)
    L = np.asarray(labels)               # (M,)

    pix = cfg.field_size_um / n_pix
    ys, xs = np.mgrid[0:n_pix, 0:n_pix]
    P = np.stack([(xs + 0.5) * pix, (ys + 0.5) * pix], axis=-1).reshape(-1, 2)  # (n_pix², 2)

    # メモリ節約のためチャンク処理で argmin(|p-s|² - r²)
    best = np.empty(len(P), dtype=np.int64)
    chunk = 8192
    R2 = R ** 2
    for start in range(0, len(P), chunk):
        p = P[start : start + chunk]
        d2 = ((p[:, None, :] - S[None, :, :]) ** 2).sum(axis=2) - R2[None, :]
        best[start : start + chunk] = d2.argmin(axis=1)
    return L[best].reshape(n_pix, n_pix)


def effective_diffusion_lengths(
    cond: GrainCondition,
    mobility: Optional[Dict[str, float]] = None,
    d0: float = SIM_D0_M2_S,
    q: float = SIM_Q_J_MOL,
    contact_exponent: float = 0.25,
    d50_ref_um: float = 30.0,
) -> Dict[str, float]:
    """元素ごとの実効拡散距離 [µm]。

    L_i = 2√(ΣDt) × mobility_i × (D50_ref / D50_i)^contact_exponent

    微粉ほど粒子間接触点が多く焼結初期のネック形成が速いことを、
    contact_exponent による緩やかな増幅で表現する。
    """
    mob = mobility or DEFAULT_MOBILITY
    base = cond.diffusion_length_um(d0, q)
    out: Dict[str, float] = {}
    for el in cond.elements:
        m = mob.get(el, 0.5)
        contact = (d50_ref_um / cond.d50_um[el]) ** contact_exponent
        out[el] = base * m * contact
    return out


def _apply_phase_attraction(
    comp: np.ndarray,
    elements: Sequence[str],
    phases: List[Dict[str, float]],
    strength: float,
    capture_radius_at: float = 20.0,
    pure_threshold_at: float = 95.0,
) -> np.ndarray:
    """相互拡散帯の組成を最寄り化合物組成へ引き寄せる。comp: (N, n_el) [at%]"""
    if not phases or strength <= 0:
        return comp
    P = np.array([[ph.get(el, 0.0) for el in elements] for ph in phases])  # (K, n_el)
    P = P / P.sum(axis=1, keepdims=True) * 100.0
    out = comp.copy()
    mixed = comp.max(axis=1) < pure_threshold_at
    idx = np.where(mixed)[0]
    if idx.size == 0:
        return out
    c = comp[idx]
    d = np.linalg.norm(c[:, None, :] - P[None, :, :], axis=2)  # (n, K)
    k = d.argmin(axis=1)
    dmin = d[np.arange(len(idx)), k]
    w = strength * np.clip(1.0 - dmin / capture_radius_at, 0.0, 1.0)
    out[idx] = c + w[:, None] * (P[k] - c)
    return out


def simulate_condition(
    cond: GrainCondition,
    cfg: Optional[SimulationConfig] = None,
    n_pix: int = 256,
    mobility: Optional[Dict[str, float]] = None,
    contact_exponent: float = 0.25,
    seed: Optional[int] = None,
) -> SimulationResult:
    """1 つの粒度条件について合成 EDS 点分析データを生成する。"""
    cfg = cfg or SimulationConfig()
    elements = cond.elements
    rng = np.random.default_rng(cfg.random_state if seed is None else seed)
    pix = cfg.field_size_um / n_pix

    # 1. 粒子配置
    grain_map = _power_voronoi_grain_map(cond, cfg, elements, n_pix, rng)

    # 2. 拡散 (元素別 Gaussian 平滑化)
    L = effective_diffusion_lengths(cond, mobility, contact_exponent=contact_exponent)
    fields = np.zeros((len(elements), n_pix, n_pix), dtype=float)
    for i, el in enumerate(elements):
        ind = (grain_map == i).astype(float)
        sigma_pix = max(L[el] / pix / 2.0, 1e-3)  # 2√(Dt) ≈ 2σ とみなす
        fields[i] = gaussian_filter(ind, sigma=sigma_pix, mode="reflect")
    total = fields.sum(axis=0, keepdims=True)
    total[total == 0] = 1.0
    comp_grid = fields / total * 100.0

    # 3. 化合物相への引き寄せ
    flat = comp_grid.reshape(len(elements), -1).T
    flat = _apply_phase_attraction(flat, elements, cfg.intermetallic_phases, cfg.phase_attraction)
    comp_grid = flat.T.reshape(len(elements), n_pix, n_pix)

    # 4. 点分析サンプリング + ノイズ
    xy = rng.uniform(0.0, cfg.field_size_um, size=(cfg.n_points, 2))
    ix = np.clip((xy[:, 0] / pix).astype(int), 0, n_pix - 1)
    iy = np.clip((xy[:, 1] / pix).astype(int), 0, n_pix - 1)
    c = comp_grid[:, iy, ix].T                                   # (n_points, n_el)
    c = c + rng.normal(0.0, cfg.eds_noise_at, size=c.shape)
    c = np.clip(c, 0.0, None)
    c = c / c.sum(axis=1, keepdims=True) * 100.0

    df = pd.DataFrame({"x_um": xy[:, 0], "y_um": xy[:, 1]})
    for i, el in enumerate(elements):
        df[el] = c[:, i]
    df["condition"] = cond.name

    meta = {
        "n_pix": float(n_pix),
        "field_size_um": cfg.field_size_um,
        "mean_d50_um": cond.mean_d50_um,
        "d50_ratio": cond.d50_ratio,
        "base_diffusion_length_um": cond.diffusion_length_um(SIM_D0_M2_S, SIM_Q_J_MOL),
    }
    return SimulationResult(
        condition=cond,
        points=df,
        grain_map=grain_map,
        comp_grid=comp_grid,
        pixel_um=pix,
        diffusion_length_um=L,
        meta=meta,
    )
