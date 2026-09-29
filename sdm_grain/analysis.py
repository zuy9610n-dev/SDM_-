"""解析コア: 1 つの点分析データセットから粒度依存指標を算出する。

算出する主な指標
----------------
組成空間カバレッジ系
  * ``n_points``               : 有効点数
  * ``unique_bins``            : bin_width 刻みで離散化した組成空間の占有ビン数
  * ``coverage_fraction``      : 占有ビン数 / 理論上到達可能なビン数 (単体格子点数)
  * ``composition_entropy``    : 占有ビンの点数分布の Shannon エントロピー (bit)
  * ``mean_nn_distance_at``    : 組成空間における最近接点距離の平均 [at%] (小さいほど密)
  * ``convex_hull_volume``     : 相互拡散した点 (主元素 < pure_threshold) の組成空間凸包体積。
                                 純元素点を含めると常に単体全体になるため除外し、
                                 「実際に拡散で到達した組成範囲の広さ」を表す

拡散・均質化系
  * ``pure_fraction``          : 主元素 ≥ pure_threshold の「未拡散」点の割合
  * ``interdiffusion_fraction``: 有意元素数 ≥ interdiffusion_min_elements の「相互拡散帯」点の割合
  * ``n_elements_mean``        : 1 点あたりの有意元素数の平均 (多元化の進み具合)
  * ``quinary_fraction``       : 5 元素すべてが有意な点の割合 (5 元系適用の核心指標)
  * ``homogeneity_index``      : 1 - 平均組成からの距離の CV。1 に近いほど均質化 (=情報量減少)
  * ``spatial_gradient_at_um`` : 面内座標がある場合、隣接点間の組成勾配の中央値 [at%/µm]

相候補 (点密度) 系
  * ``n_clusters``             : DBSCAN による高密度クラスタ (相候補) 数
  * ``cluster_fraction``       : クラスタに属する点の割合
  * ``clusters``               : 各クラスタの代表組成・点数・広がり

総合
  * ``sdm_score``              : 上記を正規化・重み付けした総合スコア (compare モジュールで条件間相対評価)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, cKDTree
from sklearn.cluster import DBSCAN

from .config import AnalysisConfig
from .io import has_coordinates, normalize_compositions


@dataclass
class ClusterInfo:
    label: int
    n_points: int
    fraction: float
    centroid_at: Dict[str, float]
    spread_at: float                     # 重心からの RMS 距離 [at%]
    nearest_phase: Optional[str] = None  # 既知化合物との照合結果 (任意)
    nearest_phase_distance_at: Optional[float] = None


@dataclass
class DatasetMetrics:
    name: str
    n_points: int
    elements: List[str]

    # coverage
    unique_bins: int
    total_bins: int
    coverage_fraction: float
    composition_entropy: float
    max_entropy: float
    mean_nn_distance_at: float
    convex_hull_volume: float

    # diffusion / homogenisation
    pure_fraction: float
    pure_fraction_by_element: Dict[str, float]
    interdiffusion_fraction: float
    n_elements_mean: float
    n_elements_histogram: Dict[int, float]
    quinary_fraction: float
    homogeneity_index: float
    mean_composition_at: Dict[str, float]
    std_composition_at: Dict[str, float]
    spatial_gradient_at_um: Optional[float]

    # clusters
    n_clusters: int
    cluster_fraction: float
    clusters: List[ClusterInfo] = field(default_factory=list)

    # misc
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["n_elements_histogram"] = {str(k): v for k, v in self.n_elements_histogram.items()}
        return d

    def to_flat_series(self) -> pd.Series:
        """比較表用に主要スカラー指標を Series 化。"""
        s = {
            "name": self.name,
            "n_points": self.n_points,
            "unique_bins": self.unique_bins,
            "coverage_fraction": self.coverage_fraction,
            "composition_entropy": self.composition_entropy,
            "mean_nn_distance_at": self.mean_nn_distance_at,
            "convex_hull_volume": self.convex_hull_volume,
            "pure_fraction": self.pure_fraction,
            "interdiffusion_fraction": self.interdiffusion_fraction,
            "n_elements_mean": self.n_elements_mean,
            "quinary_fraction": self.quinary_fraction,
            "homogeneity_index": self.homogeneity_index,
            "spatial_gradient_at_um": self.spatial_gradient_at_um if self.spatial_gradient_at_um is not None else np.nan,
            "n_clusters": self.n_clusters,
            "cluster_fraction": self.cluster_fraction,
        }
        return pd.Series(s)


# ---------------------------------------------------------------------------
# 個別指標
# ---------------------------------------------------------------------------

def _simplex_bin_count(n_elements: int, bin_width_at: float) -> int:
    """合計 100 at% の制約下で bin_width 刻みの格子点数 (組合せ C(m+n-1, n-1))。"""
    from math import comb

    m = int(round(100.0 / bin_width_at))
    return comb(m + n_elements - 1, n_elements - 1)


def composition_bins(comp: np.ndarray, bin_width_at: float) -> np.ndarray:
    """組成行列 (N, n_el) を bin index の整数配列に変換。"""
    return np.floor(comp / bin_width_at + 1e-9).astype(np.int64)


def coverage_metrics(comp: np.ndarray, cfg: AnalysisConfig) -> Dict[str, Any]:
    n_el = comp.shape[1]
    bins = composition_bins(comp, cfg.bin_width_at)
    uniq, counts = np.unique(bins, axis=0, return_counts=True)
    total = _simplex_bin_count(n_el, cfg.bin_width_at)
    p = counts / counts.sum()
    entropy = float(-(p * np.log2(p)).sum())
    max_entropy = float(np.log2(total))

    # 最近接距離 (自身を除く)
    if len(comp) >= 2:
        tree = cKDTree(comp)
        d, _ = tree.query(comp, k=2)
        mean_nn = float(np.mean(d[:, 1]))
    else:
        mean_nn = float("nan")

    # 凸包体積: 純元素 (未拡散) 点を除いた相互拡散点のみで計算。
    # 単体制約で 1 次元落ちるので最後の元素を除いた座標で計算。
    hull_vol = 0.0
    mixed = comp[comp.max(axis=1) < cfg.pure_threshold_at]
    reduced = mixed[:, :-1]
    if len(reduced) > reduced.shape[1] + 1:
        try:
            hull_vol = float(ConvexHull(reduced, qhull_options="QJ").volume)
        except (QhullError, ValueError):
            hull_vol = 0.0

    return dict(
        unique_bins=int(len(uniq)),
        total_bins=int(total),
        coverage_fraction=float(len(uniq) / total),
        composition_entropy=entropy,
        max_entropy=max_entropy,
        mean_nn_distance_at=mean_nn,
        convex_hull_volume=hull_vol,
        bin_counts=counts,
    )


def diffusion_metrics(comp: np.ndarray, elements: Sequence[str], cfg: AnalysisConfig) -> Dict[str, Any]:
    n_el = comp.shape[1]
    max_el = comp.max(axis=1)
    arg_el = comp.argmax(axis=1)
    pure_mask = max_el >= cfg.pure_threshold_at
    pure_by_el = {el: float(np.mean(pure_mask & (arg_el == i))) for i, el in enumerate(elements)}

    sig = comp >= cfg.significant_at
    n_sig = sig.sum(axis=1)
    hist = {int(k): float(np.mean(n_sig == k)) for k in range(0, n_el + 1)}
    inter_mask = n_sig >= cfg.interdiffusion_min_elements
    quinary = float(np.mean(n_sig == n_el))

    mean_c = comp.mean(axis=0)
    dist = np.linalg.norm(comp - mean_c, axis=1)
    # 完全均質: 全点が平均組成 -> dist=0 -> index=1。理論最大距離で正規化。
    max_dist = float(np.sqrt(2.0) * 100.0)
    homogeneity = float(1.0 - np.mean(dist) / max_dist)

    return dict(
        pure_fraction=float(pure_mask.mean()),
        pure_fraction_by_element=pure_by_el,
        interdiffusion_fraction=float(inter_mask.mean()),
        n_elements_mean=float(n_sig.mean()),
        n_elements_histogram=hist,
        quinary_fraction=quinary,
        homogeneity_index=homogeneity,
        mean_composition_at={el: float(v) for el, v in zip(elements, mean_c)},
        std_composition_at={el: float(v) for el, v in zip(elements, comp.std(axis=0))},
    )


def spatial_gradient(df: pd.DataFrame, elements: Sequence[str], k: int = 4) -> Optional[float]:
    """面内で近い k 点との組成差 / 距離 の中央値 [at%/µm]。座標が無ければ None。"""
    if not has_coordinates(df) or len(df) < k + 1:
        return None
    xy = df[["x_um", "y_um"]].to_numpy(float)
    comp = df[list(elements)].to_numpy(float)
    tree = cKDTree(xy)
    d, idx = tree.query(xy, k=k + 1)
    d = d[:, 1:]
    idx = idx[:, 1:]
    valid = d > 1e-9
    dc = np.linalg.norm(comp[idx] - comp[:, None, :], axis=2)
    grads = (dc[valid] / d[valid])
    return float(np.median(grads)) if grads.size else None


def cluster_metrics(
    comp: np.ndarray,
    elements: Sequence[str],
    cfg: AnalysisConfig,
    known_phases: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """DBSCAN で組成空間の高密度領域 (相候補) を抽出する。"""
    if len(comp) < cfg.dbscan_min_samples:
        return dict(n_clusters=0, cluster_fraction=0.0, clusters=[], labels=np.full(len(comp), -1))
    db = DBSCAN(eps=cfg.dbscan_eps_at, min_samples=cfg.dbscan_min_samples).fit(comp)
    labels = db.labels_
    uniq = [l for l in np.unique(labels) if l >= 0]
    infos: List[ClusterInfo] = []
    for l in uniq:
        m = labels == l
        c = comp[m].mean(axis=0)
        spread = float(np.sqrt(np.mean(np.sum((comp[m] - c) ** 2, axis=1))))
        info = ClusterInfo(
            label=int(l),
            n_points=int(m.sum()),
            fraction=float(m.mean()),
            centroid_at={el: float(v) for el, v in zip(elements, c)},
            spread_at=spread,
        )
        if known_phases:
            best, bd = None, np.inf
            for pname, pcomp in known_phases.items():
                vec = np.array([pcomp.get(el, 0.0) for el in elements])
                dd = float(np.linalg.norm(vec - c))
                if dd < bd:
                    best, bd = pname, dd
            info.nearest_phase = best
            info.nearest_phase_distance_at = bd
        infos.append(info)
    infos.sort(key=lambda i: -i.n_points)
    return dict(
        n_clusters=len(infos),
        cluster_fraction=float(np.mean(labels >= 0)),
        clusters=infos,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def analyze_dataset(
    df: pd.DataFrame,
    cfg: Optional[AnalysisConfig] = None,
    name: str = "dataset",
    known_phases: Optional[Dict[str, Dict[str, float]]] = None,
) -> DatasetMetrics:
    """1 条件分の点分析 DataFrame を解析して DatasetMetrics を返す。"""
    cfg = cfg or AnalysisConfig()
    elements = list(cfg.elements)
    missing = [e for e in elements if e not in df.columns]
    if missing:
        raise ValueError(f"DataFrame に元素列がありません: {missing}")
    if cfg.normalize:
        df = normalize_compositions(df, elements)
    if len(df) == 0:
        raise ValueError("有効な点分析データがありません")

    comp = df[elements].to_numpy(dtype=float)

    cov = coverage_metrics(comp, cfg)
    dif = diffusion_metrics(comp, elements, cfg)
    clu = cluster_metrics(comp, elements, cfg, known_phases)
    grad = spatial_gradient(df, elements)

    metrics = DatasetMetrics(
        name=name,
        n_points=int(len(df)),
        elements=elements,
        unique_bins=cov["unique_bins"],
        total_bins=cov["total_bins"],
        coverage_fraction=cov["coverage_fraction"],
        composition_entropy=cov["composition_entropy"],
        max_entropy=cov["max_entropy"],
        mean_nn_distance_at=cov["mean_nn_distance_at"],
        convex_hull_volume=cov["convex_hull_volume"],
        pure_fraction=dif["pure_fraction"],
        pure_fraction_by_element=dif["pure_fraction_by_element"],
        interdiffusion_fraction=dif["interdiffusion_fraction"],
        n_elements_mean=dif["n_elements_mean"],
        n_elements_histogram=dif["n_elements_histogram"],
        quinary_fraction=dif["quinary_fraction"],
        homogeneity_index=dif["homogeneity_index"],
        mean_composition_at=dif["mean_composition_at"],
        std_composition_at=dif["std_composition_at"],
        spatial_gradient_at_um=grad,
        n_clusters=clu["n_clusters"],
        cluster_fraction=clu["cluster_fraction"],
        clusters=clu["clusters"],
        extra={"cluster_labels": clu["labels"].tolist(), "bin_counts": cov["bin_counts"].tolist()},
    )
    return metrics


def default_known_phases() -> Dict[str, Dict[str, float]]:
    """Al-Si-Ti-Fe-Cu 系で照合に用いる代表化合物 (at%)。"""
    return {
        "Al2Cu(θ)": {"Al": 66.7, "Cu": 33.3},
        "Al3Ti": {"Al": 75.0, "Ti": 25.0},
        "Al13Fe4": {"Al": 76.5, "Fe": 23.5},
        "Al5FeSi(β)": {"Al": 71.4, "Fe": 14.3, "Si": 14.3},
        "Al8Fe2Si(α)": {"Al": 72.7, "Fe": 18.2, "Si": 9.1},
        "FeSi": {"Fe": 50.0, "Si": 50.0},
        "Fe3Si": {"Fe": 75.0, "Si": 25.0},
        "TiSi2": {"Ti": 33.3, "Si": 66.7},
        "Ti5Si3": {"Ti": 62.5, "Si": 37.5},
        "Al7Cu2Fe": {"Al": 70.0, "Cu": 20.0, "Fe": 10.0},
        "TiCu": {"Ti": 50.0, "Cu": 50.0},
        "Fe2Ti": {"Fe": 66.7, "Ti": 33.3},
        "Al(fcc)": {"Al": 100.0},
        "Si(dia)": {"Si": 100.0},
        "Ti(hcp)": {"Ti": 100.0},
        "Fe(bcc)": {"Fe": 100.0},
        "Cu(fcc)": {"Cu": 100.0},
    }
