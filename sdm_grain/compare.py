"""複数粒度条件の指標を比較・ランキングし、粒度依存性 (トレンド) を自動判定する。

総合スコア (``sdm_score``)
--------------------------
焼結拡散マルチプル法にとって「良い」試料とは、

* 組成空間を広く・密にカバーしている (coverage, entropy, hull volume が大きい)
* 5 元素が同時に共存する相互拡散帯が十分ある (quinary_fraction, interdiffusion_fraction が大きい)
* 未反応の純元素残留が少ない (pure_fraction が小さい)
* しかし完全に均質化して情報が失われていない (homogeneity_index が高すぎない)
* 点密度ピーク (相候補) が明瞭に検出される (n_clusters, cluster_fraction)

というもの。各指標を条件間で min-max 正規化し、重み付き和で 0-100 のスコアにする。
重みは ``ScoreWeights`` で変更可能。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import DatasetMetrics
from .config import GrainCondition


@dataclass
class ScoreWeights:
    coverage_fraction: float = 2.0
    composition_entropy: float = 1.5
    convex_hull_volume: float = 1.0
    quinary_fraction: float = 2.5
    interdiffusion_fraction: float = 1.5
    pure_fraction: float = -1.5          # 負: 小さいほど良い
    homogeneity_penalty: float = -1.0    # 過均質化ペナルティ (閾値超過分)
    n_clusters: float = 1.0
    cluster_fraction: float = 0.5
    homogeneity_soft_threshold: float = 0.85  # これを超える homogeneity_index を減点

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TrendResult:
    """指標 vs 粒度パラメータの単調性・相関の判定。"""

    metric: str
    predictor: str
    spearman_rho: float
    p_value: float
    slope_per_decade: float               # log10(predictor) に対する線形回帰勾配
    direction: str                        # "increase" / "decrease" / "flat"
    n: int


@dataclass
class ComparisonResult:
    table: pd.DataFrame                   # 条件 × 指標 (生値)
    normalized: pd.DataFrame              # min-max 正規化後
    scores: pd.Series                     # sdm_score (0-100)
    ranking: List[str]
    trends: List[TrendResult]
    weights: ScoreWeights
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "table": self.table.reset_index().to_dict(orient="records"),
            "scores": self.scores.to_dict(),
            "ranking": self.ranking,
            "trends": [asdict(t) for t in self.trends],
            "weights": self.weights.to_dict(),
            "recommendations": self.recommendations,
        }


# 指標の説明 (レポート表示用)
METRIC_LABELS: Dict[str, str] = {
    "n_points": "点数",
    "unique_bins": "占有組成ビン数",
    "coverage_fraction": "組成空間カバレッジ",
    "composition_entropy": "組成エントロピー [bit]",
    "mean_nn_distance_at": "平均最近接距離 [at%]",
    "convex_hull_volume": "凸包体積 (組成空間)",
    "pure_fraction": "未拡散 (純元素) 点割合",
    "interdiffusion_fraction": "相互拡散帯 (≥3元素) 割合",
    "n_elements_mean": "平均有意元素数",
    "quinary_fraction": "5元素共存点割合",
    "homogeneity_index": "均質化指数",
    "spatial_gradient_at_um": "面内組成勾配 [at%/µm]",
    "n_clusters": "相候補クラスタ数",
    "cluster_fraction": "クラスタ所属点割合",
    "mean_d50_um": "平均 D50 [µm]",
    "d50_ratio": "D50 比 (max/min)",
    "diffusion_length_um": "特性拡散距離 [µm]",
    "sdm_score": "SDM 総合スコア",
}


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def build_table(
    metrics: Sequence[DatasetMetrics],
    conditions: Optional[Sequence[GrainCondition]] = None,
) -> pd.DataFrame:
    rows = [m.to_flat_series() for m in metrics]
    df = pd.DataFrame(rows).set_index("name")
    if conditions:
        cmap = {c.name: c for c in conditions}
        df["mean_d50_um"] = [cmap[n].mean_d50_um if n in cmap else np.nan for n in df.index]
        df["d50_ratio"] = [cmap[n].d50_ratio if n in cmap else np.nan for n in df.index]
        df["diffusion_length_um"] = [cmap[n].diffusion_length_um() if n in cmap else np.nan for n in df.index]
        # 元素別 D50 も付与
        for el in conditions[0].elements:
            df[f"d50_{el}_um"] = [cmap[n].d50_um.get(el, np.nan) if n in cmap else np.nan for n in df.index]
    return df


def compute_scores(table: pd.DataFrame, weights: Optional[ScoreWeights] = None) -> Tuple[pd.Series, pd.DataFrame]:
    w = weights or ScoreWeights()
    norm = pd.DataFrame(index=table.index)
    contrib = pd.DataFrame(index=table.index)

    def add(metric: str, weight: float) -> None:
        if metric not in table.columns or weight == 0:
            return
        n = _minmax(table[metric].astype(float).fillna(table[metric].astype(float).median()))
        norm[metric] = n
        contrib[metric] = n * weight

    add("coverage_fraction", w.coverage_fraction)
    add("composition_entropy", w.composition_entropy)
    add("convex_hull_volume", w.convex_hull_volume)
    add("quinary_fraction", w.quinary_fraction)
    add("interdiffusion_fraction", w.interdiffusion_fraction)
    add("pure_fraction", w.pure_fraction)
    add("n_clusters", w.n_clusters)
    add("cluster_fraction", w.cluster_fraction)

    if "homogeneity_index" in table.columns and w.homogeneity_penalty != 0:
        excess = (table["homogeneity_index"] - w.homogeneity_soft_threshold).clip(lower=0.0)
        rng = excess.max()
        pen = excess / rng if rng > 1e-12 else excess * 0.0
        norm["homogeneity_excess"] = pen
        contrib["homogeneity_excess"] = pen * w.homogeneity_penalty

    weight_values = [
        w.coverage_fraction, w.composition_entropy, w.convex_hull_volume, w.quinary_fraction,
        w.interdiffusion_fraction, w.pure_fraction, w.homogeneity_penalty, w.n_clusters, w.cluster_fraction,
    ]
    pos = sum(v for v in weight_values if v > 0)
    neg = -sum(v for v in weight_values if v < 0)
    raw = contrib.sum(axis=1)
    # raw ∈ [-neg, pos] → 0-100 へ線形変換
    score = (raw + neg) / (pos + neg) * 100.0
    return score.rename("sdm_score"), norm


def compute_trends(
    table: pd.DataFrame,
    predictors: Sequence[str] = ("mean_d50_um",),
    metrics: Sequence[str] = (
        "coverage_fraction",
        "composition_entropy",
        "quinary_fraction",
        "interdiffusion_fraction",
        "pure_fraction",
        "homogeneity_index",
        "n_clusters",
    ),
    alpha: float = 0.1,
) -> List[TrendResult]:
    out: List[TrendResult] = []
    for pred in predictors:
        if pred not in table.columns:
            continue
        x_raw = table[pred].astype(float)
        for met in metrics:
            if met not in table.columns:
                continue
            y = table[met].astype(float)
            mask = np.isfinite(x_raw) & np.isfinite(y)
            if mask.sum() < 3 or x_raw[mask].nunique() < 3:
                continue
            x = np.log10(x_raw[mask].to_numpy())
            yy = y[mask].to_numpy()
            if np.ptp(yy) < 1e-12:
                # 指標が全条件で一定 → 相関は定義されない
                out.append(TrendResult(met, pred, 0.0, 1.0, 0.0, "flat", int(mask.sum())))
                continue
            rho, p = stats.spearmanr(x, yy)
            slope = float(np.polyfit(x, yy, 1)[0]) if np.ptp(x) > 0 else 0.0
            if not np.isfinite(rho):
                rho, p = 0.0, 1.0
            if p < alpha and rho > 0:
                direction = "increase"
            elif p < alpha and rho < 0:
                direction = "decrease"
            else:
                direction = "flat"
            out.append(TrendResult(met, pred, float(rho), float(p), slope, direction, int(mask.sum())))
    return out


def make_recommendations(
    table: pd.DataFrame,
    scores: pd.Series,
    trends: Sequence[TrendResult],
    weights: ScoreWeights,
) -> List[str]:
    """数値結果から日本語の所見を自動生成する。"""
    rec: List[str] = []
    best = scores.idxmax()
    worst = scores.idxmin()
    rec.append(f"総合スコア最大は「{best}」({scores[best]:.1f} 点)、最小は「{worst}」({scores[worst]:.1f} 点)。")

    tmap = {(t.predictor, t.metric): t for t in trends}

    def trend_text(metric: str, jp: str) -> None:
        t = tmap.get(("mean_d50_um", metric))
        if t is None:
            return
        if t.direction == "increase":
            rec.append(f"平均 D50 を大きくすると{jp}は増加する傾向 (Spearman ρ={t.spearman_rho:+.2f}, p={t.p_value:.2g})。")
        elif t.direction == "decrease":
            rec.append(f"平均 D50 を大きくすると{jp}は減少する傾向 (Spearman ρ={t.spearman_rho:+.2f}, p={t.p_value:.2g})。")

    trend_text("coverage_fraction", "組成空間カバレッジ")
    trend_text("quinary_fraction", "5 元素共存点の割合")
    trend_text("pure_fraction", "未拡散 (純元素) 残留の割合")
    trend_text("homogeneity_index", "均質化指数")

    # 過均質化・未拡散の警告
    if "homogeneity_index" in table.columns:
        over = table.index[table["homogeneity_index"] > weights.homogeneity_soft_threshold].tolist()
        if over:
            rec.append(
                f"{', '.join(over)} は均質化指数が {weights.homogeneity_soft_threshold} を超えており、"
                "拡散が進みすぎて組成情報が失われている可能性がある。粒度を粗くするか熱処理時間を短縮する余地がある。"
            )
    if "pure_fraction" in table.columns:
        under = table.index[table["pure_fraction"] > 0.5].tolist()
        if under:
            rec.append(
                f"{', '.join(under)} は点分析の半数以上が未拡散の純元素領域であり、"
                "相互拡散帯の情報量が不足している。粒度を細かくするか熱処理を強化する余地がある。"
            )
    # 混合粒度の効果
    if "d50_ratio" in table.columns and table["d50_ratio"].max() > 1.5:
        mixed = table[table["d50_ratio"] > 1.5]
        uniform = table[table["d50_ratio"] <= 1.5]
        if len(mixed) and len(uniform):
            qm, qu = mixed["quinary_fraction"].mean(), uniform["quinary_fraction"].mean()
            if qm > qu * 1.1:
                rec.append(
                    f"元素間で粒度を変えた混合条件 (D50 比 > 1.5) は均一粒度条件より 5 元素共存点割合が高い "
                    f"({qm:.3f} vs {qu:.3f})。低移動度元素 (Ti, Fe) を微粉化する配置が有効な可能性がある。"
                )
            elif qu > qm * 1.1:
                rec.append(
                    f"均一粒度条件のほうが混合粒度条件より 5 元素共存点割合が高い ({qu:.3f} vs {qm:.3f})。"
                )
    if "quinary_fraction" in table.columns:
        qbest = table["quinary_fraction"].idxmax()
        rec.append(
            f"5 元系適用の核心指標である 5 元素共存点割合は「{qbest}」が最大 "
            f"({table.loc[qbest, 'quinary_fraction']:.3f})。"
        )
    return rec


def compare_conditions(
    metrics: Sequence[DatasetMetrics],
    conditions: Optional[Sequence[GrainCondition]] = None,
    weights: Optional[ScoreWeights] = None,
) -> ComparisonResult:
    """複数条件の DatasetMetrics を比較する。"""
    if len(metrics) == 0:
        raise ValueError("metrics が空です")
    w = weights or ScoreWeights()
    table = build_table(metrics, conditions)
    scores, norm = compute_scores(table, w)
    table["sdm_score"] = scores
    ranking = scores.sort_values(ascending=False).index.tolist()
    predictors = [p for p in ("mean_d50_um", "d50_ratio", "diffusion_length_um") if p in table.columns]
    trends = compute_trends(table, predictors=predictors) if predictors else []
    rec = make_recommendations(table, scores, trends, w)
    return ComparisonResult(
        table=table,
        normalized=norm,
        scores=scores,
        ranking=ranking,
        trends=trends,
        weights=w,
        recommendations=rec,
    )
