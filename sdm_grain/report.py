"""図表生成と HTML レポート出力。

matplotlib (Agg) で PNG を生成し、単一の自己完結 HTML にまとめる。
"""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import DatasetMetrics  # noqa: E402
from .compare import METRIC_LABELS, ComparisonResult  # noqa: E402
from .config import GrainCondition  # noqa: E402

# 日本語フォントが無い環境でも警告を抑えるため、図中ラベルは英語にする
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _fig_to_b64(fig: plt.Figure, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save_fig(fig: plt.Figure, path: Optional[Path], dpi: int = 130) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 個別図
# ---------------------------------------------------------------------------

def plot_metric_bars(comp: ComparisonResult, out: Optional[Path] = None) -> str:
    metrics = [
        "sdm_score",
        "coverage_fraction",
        "composition_entropy",
        "quinary_fraction",
        "interdiffusion_fraction",
        "pure_fraction",
        "homogeneity_index",
        "n_clusters",
    ]
    metrics = [m for m in metrics if m in comp.table.columns]
    n = len(metrics)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    order = comp.ranking
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(order)))
    for ax, m in zip(axes, metrics):
        vals = comp.table.loc[order, m].astype(float)
        ax.bar(range(len(order)), vals.values, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7)
        ax.set_title(m, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Metrics by grain-size condition (sorted by SDM score)", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out)
    return _fig_to_b64(fig)


def plot_trends(comp: ComparisonResult, out: Optional[Path] = None, predictor: str = "mean_d50_um") -> Optional[str]:
    if predictor not in comp.table.columns:
        return None
    metrics = ["coverage_fraction", "quinary_fraction", "pure_fraction", "homogeneity_index", "composition_entropy", "n_clusters"]
    metrics = [m for m in metrics if m in comp.table.columns]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.ravel()
    x = comp.table[predictor].astype(float)
    tmap = {t.metric: t for t in comp.trends if t.predictor == predictor}
    for ax, m in zip(axes, metrics):
        y = comp.table[m].astype(float)
        ax.scatter(x, y, c="tab:blue", s=40, zorder=3)
        for name, xi, yi in zip(comp.table.index, x, y):
            ax.annotate(name, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
        if m in tmap and np.isfinite(x).sum() >= 3:
            t = tmap[m]
            xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
            coef = np.polyfit(np.log10(x), y, 1)
            ax.plot(xs, np.polyval(coef, np.log10(xs)), "r--", lw=1, label=f"ρ={t.spearman_rho:+.2f}, p={t.p_value:.2g}")
            ax.legend(fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel(predictor)
        ax.set_title(m, fontsize=9)
    fig.suptitle(f"Metric trends vs {predictor}", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out)
    return _fig_to_b64(fig)


def plot_element_histogram(metrics: Sequence[DatasetMetrics], out: Optional[Path] = None) -> str:
    """有意元素数分布 (1 点に何元素が共存しているか) を条件別に比較。"""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    n_el = len(metrics[0].elements)
    width = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        ks = list(range(1, n_el + 1))
        vals = [m.n_elements_histogram.get(k, 0.0) for k in ks]
        ax.bar(np.array(ks) + (i - len(metrics) / 2 + 0.5) * width, vals, width=width, label=m.name)
    ax.set_xlabel("Number of significant elements per point")
    ax.set_ylabel("Fraction of points")
    ax.set_xticks(range(1, n_el + 1))
    ax.legend(fontsize=7)
    ax.set_title("Multi-element coexistence distribution")
    fig.tight_layout()
    _save_fig(fig, out)
    return _fig_to_b64(fig)


def plot_pairwise_compositions(
    datasets: Dict[str, pd.DataFrame],
    elements: Sequence[str],
    pairs: Optional[List[tuple]] = None,
    out: Optional[Path] = None,
    max_points: int = 1500,
) -> str:
    """組成空間の 2 元素射影 (散布図) を条件別に重ね描き。"""
    if pairs is None:
        pairs = [("Al", "Cu"), ("Al", "Fe"), ("Ti", "Si"), ("Fe", "Si"), ("Al", "Ti"), ("Cu", "Fe")]
        pairs = [p for p in pairs if p[0] in elements and p[1] in elements][:6]
    ncol = 3
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.cm.tab10
    for ax, (a, b) in zip(axes, pairs):
        for i, (name, df) in enumerate(datasets.items()):
            d = df.sample(min(max_points, len(df)), random_state=0) if len(df) > max_points else df
            ax.scatter(d[a], d[b], s=4, alpha=0.35, color=cmap(i % 10), label=name if ax is axes[0] else None)
        ax.set_xlabel(f"{a} [at%]")
        ax.set_ylabel(f"{b} [at%]")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.plot([0, 100], [100, 0], "k:", lw=0.6)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    axes[0].legend(fontsize=6, markerscale=3, loc="upper right")
    fig.suptitle("Composition-space projections", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out)
    return _fig_to_b64(fig)


def plot_spatial_maps(
    datasets: Dict[str, pd.DataFrame],
    elements: Sequence[str],
    out: Optional[Path] = None,
) -> Optional[str]:
    """面内座標がある場合、主元素マップと有意元素数マップを条件別に描く。"""
    usable = {k: v for k, v in datasets.items() if {"x_um", "y_um"}.issubset(v.columns)}
    if not usable:
        return None
    n = len(usable)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.2), squeeze=False)
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(elements))
    for j, (name, df) in enumerate(usable.items()):
        comp = df[list(elements)].to_numpy(float)
        major = comp.argmax(axis=1)
        nsig = (comp >= 2.0).sum(axis=1)
        ax = axes[0, j]
        sc = ax.scatter(df["x_um"], df["y_um"], c=major, cmap=cmap, s=6, vmin=-0.5, vmax=len(elements) - 0.5)
        ax.set_title(f"{name}\nmajor element", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1, j]
        sc2 = ax.scatter(df["x_um"], df["y_um"], c=nsig, cmap="magma", s=6, vmin=1, vmax=len(elements))
        ax.set_title("# significant elements", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(sc, ax=axes[0, :].tolist(), ticks=range(len(elements)), fraction=0.02, pad=0.01)
    cb.ax.set_yticklabels(list(elements))
    fig.colorbar(sc2, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01)
    _save_fig(fig, out)
    return _fig_to_b64(fig)


def plot_cluster_summary(metrics: Sequence[DatasetMetrics], out: Optional[Path] = None, top: int = 6) -> str:
    """各条件の上位クラスタ (相候補) の重心組成を積み上げ棒で表示。"""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False, sharey=True)
    elements = metrics[0].elements
    colors = plt.cm.tab10(np.arange(len(elements)))
    for ax, m in zip(axes[0], metrics):
        cl = m.clusters[:top]
        bottom = np.zeros(len(cl))
        for i, el in enumerate(elements):
            vals = np.array([c.centroid_at[el] for c in cl])
            ax.bar(range(len(cl)), vals, bottom=bottom, color=colors[i], label=el if ax is axes[0][0] else None)
            bottom += vals
        labels = [
            f"#{c.label}\nn={c.n_points}" + (f"\n{c.nearest_phase}" if c.nearest_phase else "") for c in cl
        ]
        ax.set_xticks(range(len(cl)))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(f"{m.name}\n{m.n_clusters} clusters", fontsize=8)
        ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("at%")
    axes[0][0].legend(fontsize=6, ncol=len(elements), loc="lower left", bbox_to_anchor=(0, 1.15))
    fig.tight_layout()
    _save_fig(fig, out)
    return _fig_to_b64(fig)


def plot_simulation_maps(sim_results: Sequence, out: Optional[Path] = None) -> Optional[str]:
    """SimulationResult の焼結前粒子マップと拡散後の主元素マップを並べる。"""
    if not sim_results:
        return None
    n = len(sim_results)
    elements = sim_results[0].condition.elements
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.0), squeeze=False)
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(elements))
    for j, r in enumerate(sim_results):
        ax = axes[0, j]
        ax.imshow(r.grain_map, cmap=cmap, vmin=-0.5, vmax=len(elements) - 0.5, origin="lower")
        ax.set_title(f"{r.condition.name}\nas-packed", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1, j]
        nsig = (r.comp_grid >= 2.0).sum(axis=0)
        im = ax.imshow(nsig, cmap="magma", vmin=1, vmax=len(elements), origin="lower")
        ax.set_title("after diffusion: # elements", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01)
    _save_fig(fig, out)
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML レポート
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Sans','Noto Sans JP',sans-serif;
     margin:0;background:#f5f6f8;color:#222}
.wrap{max-width:1200px;margin:0 auto;padding:24px}
h1{font-size:22px;border-bottom:3px solid #2b6cb0;padding-bottom:6px}
h2{font-size:17px;margin-top:32px;border-left:5px solid #2b6cb0;padding-left:8px}
table{border-collapse:collapse;width:100%;font-size:12px;background:#fff}
th,td{border:1px solid #d0d5dd;padding:4px 6px;text-align:right;white-space:nowrap}
th{background:#e8eef7}
td:first-child,th:first-child{text-align:left;font-weight:600}
.card{background:#fff;border:1px solid #d0d5dd;border-radius:6px;padding:14px;margin:12px 0}
.rank{font-size:15px}
.rank li{margin:4px 0}
.badge{display:inline-block;background:#2b6cb0;color:#fff;border-radius:10px;padding:1px 8px;font-size:11px;margin-left:6px}
img{max-width:100%;border:1px solid #ddd;background:#fff}
.rec li{margin:6px 0;line-height:1.5}
.small{color:#666;font-size:11px}
.up{color:#c53030}.down{color:#2b6cb0}.flat{color:#777}
"""


def _fmt(v, nd=3) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "–"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    return html.escape(str(v))


def _table_html(df: pd.DataFrame, cols: Sequence[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    head = "".join(f"<th title='{html.escape(c)}'>{html.escape(METRIC_LABELS.get(c, c))}</th>" for c in cols)
    rows = []
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in cols)
        rows.append(f"<tr><td>{html.escape(str(idx))}</td>{cells}</tr>")
    return f"<table><thead><tr><th>条件</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_report(
    comp: ComparisonResult,
    metrics: Sequence[DatasetMetrics],
    datasets: Dict[str, pd.DataFrame],
    out_dir: str | Path,
    conditions: Optional[Sequence[GrainCondition]] = None,
    sim_results: Optional[Sequence] = None,
    title: str = "焼結拡散マルチプル法 粒度調整影響 自動解析レポート",
) -> Path:
    """HTML レポートと図・JSON・CSV を out_dir に書き出し、HTML パスを返す。"""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    elements = metrics[0].elements

    figs: Dict[str, Optional[str]] = {}
    figs["bars"] = plot_metric_bars(comp, fig_dir / "metric_bars.png")
    figs["trends"] = plot_trends(comp, fig_dir / "trends_vs_d50.png")
    figs["hist"] = plot_element_histogram(metrics, fig_dir / "element_count_hist.png")
    figs["pairs"] = plot_pairwise_compositions(datasets, elements, out=fig_dir / "composition_projections.png")
    figs["spatial"] = plot_spatial_maps(datasets, elements, fig_dir / "spatial_maps.png")
    figs["clusters"] = plot_cluster_summary(metrics, fig_dir / "cluster_summary.png")
    figs["sim"] = plot_simulation_maps(sim_results or [], fig_dir / "simulation_maps.png") if sim_results else None

    # データ書き出し
    comp.table.to_csv(out_dir / "metrics_table.csv", float_format="%.5f")
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comp.to_dict(), f, ensure_ascii=False, indent=2, default=_json_default)
    with open(out_dir / "metrics_detail.json", "w", encoding="utf-8") as f:
        json.dump([_strip_extra(m.to_dict()) for m in metrics], f, ensure_ascii=False, indent=2, default=_json_default)

    # HTML 組み立て
    main_cols = [
        "sdm_score", "n_points", "coverage_fraction", "composition_entropy", "convex_hull_volume",
        "quinary_fraction", "interdiffusion_fraction", "pure_fraction", "n_elements_mean",
        "homogeneity_index", "spatial_gradient_at_um", "n_clusters", "cluster_fraction",
    ]
    cond_cols = ["mean_d50_um", "d50_ratio", "diffusion_length_um"] + [f"d50_{el}_um" for el in elements]

    parts: List[str] = []
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(
        f"<p class='small'>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')} / 元素系: {'-'.join(elements)} / 条件数: {len(metrics)}</p>"
    )

    # ランキング
    parts.append("<h2>1. 総合ランキング</h2><div class='card'><ol class='rank'>")
    for name in comp.ranking:
        parts.append(f"<li>{html.escape(name)}<span class='badge'>{comp.scores[name]:.1f}</span></li>")
    parts.append("</ol><p class='small'>SDM 総合スコア: 組成空間カバレッジ・エントロピー・5元素共存割合・相互拡散帯割合・相候補数を加点、未拡散残留と過均質化を減点した 0–100 の相対評価。</p></div>")

    # 所見
    parts.append("<h2>2. 自動所見</h2><div class='card'><ul class='rec'>")
    for r in comp.recommendations:
        parts.append(f"<li>{html.escape(r)}</li>")
    parts.append("</ul></div>")

    # 条件表
    if conditions:
        parts.append("<h2>3. 粒度条件</h2><div class='card'>")
        parts.append(_table_html(comp.table.loc[comp.ranking], cond_cols))
        parts.append("<ul class='small'>")
        for c in conditions:
            heat = f"焼結 {c.sinter_temp_k:.0f} K × {c.sinter_time_s/3600:.1f} h"
            if c.anneal_temp_k:
                heat += f", 焼鈍 {c.anneal_temp_k:.0f} K × {c.anneal_time_s/3600:.1f} h"
            parts.append(f"<li><b>{html.escape(c.name)}</b>: {heat}. {html.escape(c.note)}</li>")
        parts.append("</ul></div>")

    # 指標表
    parts.append("<h2>4. 指標一覧</h2><div class='card'>")
    parts.append(_table_html(comp.table.loc[comp.ranking], main_cols))
    parts.append("</div>")

    # トレンド
    if comp.trends:
        parts.append("<h2>5. 粒度パラメータに対する指標トレンド</h2><div class='card'><table><thead><tr>"
                     "<th>説明変数</th><th>指標</th><th>Spearman ρ</th><th>p 値</th><th>勾配/decade</th><th>判定</th></tr></thead><tbody>")
        for t in comp.trends:
            cls = {"increase": "up", "decrease": "down"}.get(t.direction, "flat")
            parts.append(
                f"<tr><td>{html.escape(METRIC_LABELS.get(t.predictor, t.predictor))}</td>"
                f"<td style='text-align:left'>{html.escape(METRIC_LABELS.get(t.metric, t.metric))}</td>"
                f"<td>{t.spearman_rho:+.2f}</td><td>{t.p_value:.3g}</td><td>{t.slope_per_decade:+.3g}</td>"
                f"<td class='{cls}'>{t.direction}</td></tr>"
            )
        parts.append("</tbody></table></div>")

    # 図
    parts.append("<h2>6. 図</h2>")
    for key, cap in [
        ("bars", "指標の条件別比較 (スコア順)"),
        ("trends", "平均 D50 に対する指標トレンド"),
        ("hist", "1 点あたり有意元素数の分布"),
        ("pairs", "組成空間の 2 元素射影"),
        ("spatial", "面内マップ (主元素 / 有意元素数)"),
        ("clusters", "相候補クラスタの重心組成"),
        ("sim", "シミュレーション: 焼結前粒子配置と拡散後の元素共存数"),
    ]:
        if figs.get(key):
            parts.append(f"<div class='card'><h3 style='font-size:13px;margin:0 0 8px'>{cap}</h3><img src='data:image/png;base64,{figs[key]}'></div>")

    # クラスタ詳細
    parts.append("<h2>7. 相候補クラスタ詳細</h2>")
    for m in metrics:
        parts.append(f"<div class='card'><b>{html.escape(m.name)}</b> — {m.n_clusters} clusters, {m.cluster_fraction:.1%} of points<table><thead><tr>"
                     "<th>#</th><th>n</th><th>割合</th>" + "".join(f"<th>{el}</th>" for el in elements) + "<th>広がり [at%]</th><th>最近傍既知相</th><th>距離 [at%]</th></tr></thead><tbody>")
        for c in m.clusters[:12]:
            parts.append(
                f"<tr><td>{c.label}</td><td>{c.n_points}</td><td>{c.fraction:.3f}</td>"
                + "".join(f"<td>{c.centroid_at[el]:.1f}</td>" for el in elements)
                + f"<td>{c.spread_at:.2f}</td><td>{html.escape(c.nearest_phase or '-')}</td>"
                + f"<td>{_fmt(c.nearest_phase_distance_at) if c.nearest_phase_distance_at is not None else '-'}</td></tr>"
            )
        parts.append("</tbody></table></div>")

    parts.append("<p class='small'>sdm_grain v0.1 — 指標定義は README を参照。シミュレーション結果は簡略モデルによる定性的な目安であり、実測データによる検証が必要。</p>")

    html_doc = f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{_CSS}</style></head><body><div class='wrap'>{''.join(parts)}</div></body></html>"
    out_path = out_dir / "report.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)
    return str(o)


def _strip_extra(d: Dict) -> Dict:
    d = dict(d)
    d.pop("extra", None)
    return d
