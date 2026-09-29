"""コマンドラインインタフェース。

使い方::

    # 1) 既定 5 条件で合成データ生成 → 解析 → レポート (デモ)
    python -m sdm_grain demo -o output/demo

    # 2) 条件ファイルから合成データのみ生成
    python -m sdm_grain simulate -c configs/conditions_example.yaml -o output/sim

    # 3) 実測 EDS CSV (条件ごとに 1 ファイル or condition 列) を解析
    python -m sdm_grain analyze data/fine.csv data/coarse.csv -c configs/conditions_example.yaml -o output/real

    # 4) 1 ファイルの単独解析 (比較なし)
    python -m sdm_grain analyze data/sample.csv -o output/single
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import __version__
from .analysis import DatasetMetrics, analyze_dataset, default_known_phases
from .compare import ScoreWeights, compare_conditions
from .config import (
    ELEMENTS,
    AnalysisConfig,
    GrainCondition,
    SimulationConfig,
    default_conditions,
    load_analysis_config,
    load_conditions,
    load_simulation_config,
)
from .io import load_composition_csv, save_composition_csv, split_by_condition
from .report import build_report
from .simulate import simulate_condition


def _log(msg: str) -> None:
    print(f"[sdm_grain] {msg}", file=sys.stderr, flush=True)


def _analysis_cfg(args) -> AnalysisConfig:
    cfg = load_analysis_config(args.analysis_config) if getattr(args, "analysis_config", None) else AnalysisConfig()
    if getattr(args, "elements", None):
        cfg.elements = args.elements
    if getattr(args, "bin_width", None):
        cfg.bin_width_at = args.bin_width
    return cfg


def _sim_cfg(args) -> SimulationConfig:
    cfg = load_simulation_config(args.sim_config) if getattr(args, "sim_config", None) else SimulationConfig()
    if getattr(args, "n_points", None):
        cfg.n_points = args.n_points
    if getattr(args, "seed", None) is not None:
        cfg.random_state = args.seed
    return cfg


def _run_pipeline(
    datasets: Dict[str, pd.DataFrame],
    conditions: Optional[Sequence[GrainCondition]],
    acfg: AnalysisConfig,
    out_dir: Path,
    sim_results=None,
    title: Optional[str] = None,
) -> Path:
    known = default_known_phases()
    metrics: List[DatasetMetrics] = []
    for name, df in datasets.items():
        m = analyze_dataset(df, acfg, name=name, known_phases=known)
        metrics.append(m)
        _log(
            f"{name}: n={m.n_points} coverage={m.coverage_fraction:.3f} entropy={m.composition_entropy:.2f} "
            f"pure={m.pure_fraction:.2f} quinary={m.quinary_fraction:.3f} clusters={m.n_clusters}"
        )
    conds = None
    if conditions:
        cmap = {c.name: c for c in conditions}
        conds = [cmap[n] for n in datasets if n in cmap]
        if len(conds) != len(datasets):
            _log("警告: 一部データセット名が条件名と一致しません。粒度依存トレンドは一致分のみで計算します。")
            if not conds:
                conds = None
    comp = compare_conditions(metrics, conds)
    kwargs = {"title": title} if title else {}
    path = build_report(comp, metrics, datasets, out_dir, conditions=conds, sim_results=sim_results, **kwargs)
    _log("ランキング: " + " > ".join(f"{n}({comp.scores[n]:.1f})" for n in comp.ranking))
    for r in comp.recommendations:
        _log("所見: " + r)
    _log(f"レポート: {path}")
    return path


# ---------------------------------------------------------------------------
# サブコマンド
# ---------------------------------------------------------------------------

def cmd_simulate(args) -> int:
    conditions = load_conditions(args.conditions) if args.conditions else default_conditions()
    scfg = _sim_cfg(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, cond in enumerate(conditions):
        _log(f"simulate {cond.name} ...")
        r = simulate_condition(cond, scfg, n_pix=args.n_pix, seed=scfg.random_state + i)
        save_composition_csv(r.points, out / f"{cond.name}.csv")
        frames.append(r.points)
    save_composition_csv(pd.concat(frames, ignore_index=True), out / "all_conditions.csv")
    with open(out / "conditions.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in conditions], f, ensure_ascii=False, indent=2)
    _log(f"合成データを {out} に保存しました")
    return 0


def cmd_analyze(args) -> int:
    acfg = _analysis_cfg(args)
    conditions = load_conditions(args.conditions) if args.conditions else None
    datasets: Dict[str, pd.DataFrame] = {}
    for p in args.inputs:
        p = Path(p)
        df = load_composition_csv(p, elements=acfg.elements, normalize=acfg.normalize, unit=args.unit)
        parts = split_by_condition(df)
        for name, sub in parts.items():
            key = p.stem if name == "all" else name
            if key in datasets:
                key = f"{p.stem}:{name}"
            datasets[key] = sub
    if not datasets:
        _log("入力データがありません")
        return 1
    _run_pipeline(datasets, conditions, acfg, Path(args.output), title=args.title)
    return 0


def cmd_demo(args) -> int:
    conditions = load_conditions(args.conditions) if args.conditions else default_conditions()
    scfg = _sim_cfg(args)
    acfg = _analysis_cfg(args)
    out = Path(args.output)
    (out / "synthetic_data").mkdir(parents=True, exist_ok=True)
    datasets: Dict[str, pd.DataFrame] = {}
    sims = []
    for i, cond in enumerate(conditions):
        _log(f"simulate {cond.name} (mean D50={cond.mean_d50_um:.1f} µm) ...")
        r = simulate_condition(cond, scfg, n_pix=args.n_pix, seed=scfg.random_state + i)
        sims.append(r)
        save_composition_csv(r.points, out / "synthetic_data" / f"{cond.name}.csv")
        datasets[cond.name] = r.points.drop(columns=["condition"])
    with open(out / "conditions.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in conditions], f, ensure_ascii=False, indent=2)
    _run_pipeline(datasets, conditions, acfg, out, sim_results=sims, title=args.title)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sdm_grain",
        description="焼結拡散マルチプル法における粉末粒度調整の影響を自動解析するツール",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common_analysis(sp):
        sp.add_argument("-o", "--output", default="output", help="出力ディレクトリ")
        sp.add_argument("-c", "--conditions", help="粒度条件ファイル (YAML/JSON)")
        sp.add_argument("--analysis-config", help="解析設定ファイル (YAML/JSON)")
        sp.add_argument("--elements", nargs="+", help=f"元素列名 (既定: {' '.join(ELEMENTS)})")
        sp.add_argument("--bin-width", type=float, help="組成ビン幅 [at%%]")
        sp.add_argument("--title", help="レポートタイトル")

    def common_sim(sp):
        sp.add_argument("--sim-config", help="シミュレーション設定ファイル (YAML/JSON)")
        sp.add_argument("--n-points", type=int, help="1 条件あたりの点分析数")
        sp.add_argument("--n-pix", type=int, default=256, help="シミュレーション格子解像度")
        sp.add_argument("--seed", type=int, help="乱数シード")

    s = sub.add_parser("simulate", help="粒度条件から合成 EDS データを生成")
    s.add_argument("-o", "--output", default="output/sim")
    s.add_argument("-c", "--conditions")
    common_sim(s)
    s.set_defaults(func=cmd_simulate)

    a = sub.add_parser("analyze", help="EDS/EPMA 点分析 CSV を解析してレポート出力")
    a.add_argument("inputs", nargs="+", help="入力 CSV (複数可)。condition 列があれば分割")
    a.add_argument("--unit", choices=["at", "wt"], default="at", help="入力濃度の単位")
    common_analysis(a)
    a.set_defaults(func=cmd_analyze)

    d = sub.add_parser("demo", help="合成データ生成 → 解析 → レポートを一括実行")
    common_analysis(d)
    common_sim(d)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
