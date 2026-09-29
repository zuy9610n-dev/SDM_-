"""sdm_grain のユニット/統合テスト。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdm_grain import (
    ELEMENTS,
    AnalysisConfig,
    GrainCondition,
    SimulationConfig,
    analyze_dataset,
    compare_conditions,
    load_composition_csv,
    simulate_condition,
)
from sdm_grain.analysis import _simplex_bin_count, default_known_phases
from sdm_grain.cli import main
from sdm_grain.config import default_conditions, load_conditions
from sdm_grain.io import CompositionDataError, convert_wt_to_at, split_by_condition


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def pure_df():
    """全点が純元素 (拡散なし) のデータ。"""
    rows = []
    for i, el in enumerate(ELEMENTS):
        for _ in range(40):
            c = {e: 0.0 for e in ELEMENTS}
            c[el] = 100.0
            rows.append(c)
    return pd.DataFrame(rows)


@pytest.fixture
def uniform_df(rng):
    """組成空間を一様に埋めた (理想的に多様な) データ。"""
    comp = rng.dirichlet(np.ones(len(ELEMENTS)), size=2000) * 100.0
    return pd.DataFrame(comp, columns=ELEMENTS)


@pytest.fixture
def homogeneous_df(rng):
    """完全均質化 (等組成 + ノイズ) のデータ。"""
    comp = 20.0 + rng.normal(0.0, 0.5, size=(500, len(ELEMENTS)))
    comp = comp / comp.sum(axis=1, keepdims=True) * 100.0
    return pd.DataFrame(comp, columns=ELEMENTS)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_grain_condition_properties():
    c = GrainCondition(name="t", d50_um={"Al": 10, "Si": 20, "Ti": 40, "Fe": 40, "Cu": 10})
    assert c.mean_d50_um == pytest.approx(24.0)
    assert c.d50_ratio == pytest.approx(4.0)
    assert all(v == 1.0 for v in c.span.values())
    assert c.diffusion_length_um() > 0


def test_grain_condition_rejects_nonpositive():
    with pytest.raises(ValueError):
        GrainCondition(name="bad", d50_um={"Al": 0.0})


def test_thermal_budget_increases_with_anneal():
    a = GrainCondition(name="a", d50_um={"Al": 10}, sinter_temp_k=873, sinter_time_s=3600)
    b = GrainCondition(name="b", d50_um={"Al": 10}, sinter_temp_k=873, sinter_time_s=3600, anneal_temp_k=823, anneal_time_s=7200)
    assert b.thermal_budget() > a.thermal_budget()


def test_load_conditions_yaml(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "conditions:\n  - name: x\n    d50_um: {Al: 10, Si: 10, Ti: 10, Fe: 10, Cu: 10}\n    sinter_temp_k: 900\n",
        encoding="utf-8",
    )
    conds = load_conditions(p)
    assert len(conds) == 1 and conds[0].name == "x" and conds[0].sinter_temp_k == 900


def test_load_conditions_json(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"name": "j", "d50_um": {"Al": 5}}]), encoding="utf-8")
    assert load_conditions(p)[0].name == "j"


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def test_load_csv_normalizes_and_resolves_columns(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text(
        "X (um),Y (um),al (at%),Si,TI,Fe,cu,condition\n"
        "0,0,50,25,0,0,25,c1\n"
        "1,0,10,10,10,10,10,c1\n"
        "2,0,0,0,0,0,0,c2\n",
        encoding="utf-8",
    )
    df = load_composition_csv(p)
    assert list(df.columns[:2]) == ["x_um", "y_um"]
    assert set(ELEMENTS).issubset(df.columns)
    assert len(df) == 2  # 合計 0 の行は除外
    assert np.allclose(df[ELEMENTS].sum(axis=1), 100.0)
    parts = split_by_condition(df)
    assert set(parts) == {"c1"}


def test_load_csv_missing_element_raises(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text("Al,Si\n1,2\n", encoding="utf-8")
    with pytest.raises(CompositionDataError):
        load_composition_csv(p)


def test_load_csv_negative_raises(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text("Al,Si,Ti,Fe,Cu\n-1,2,3,4,5\n", encoding="utf-8")
    with pytest.raises(CompositionDataError):
        load_composition_csv(p)


def test_wt_to_at_conversion():
    df = pd.DataFrame([{"Al": 50.0, "Cu": 50.0, "Si": 0.0, "Ti": 0.0, "Fe": 0.0}])
    at = convert_wt_to_at(df, ELEMENTS)
    # 等重量なら軽い Al の at% が大きい
    assert at.loc[0, "Al"] > at.loc[0, "Cu"]
    assert at.loc[0, ELEMENTS].sum() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def test_simplex_bin_count():
    # 2 元素, 50% 刻み → (0,100),(50,50),(100,0) = 3
    assert _simplex_bin_count(2, 50.0) == 3
    # 3 元素, 50% 刻み → C(4,2)=6
    assert _simplex_bin_count(3, 50.0) == 6


def test_pure_dataset_metrics(pure_df):
    m = analyze_dataset(pure_df, name="pure")
    assert m.pure_fraction == pytest.approx(1.0)
    assert m.interdiffusion_fraction == 0.0
    assert m.quinary_fraction == 0.0
    assert m.unique_bins == len(ELEMENTS)
    assert m.n_elements_mean == pytest.approx(1.0)
    assert set(m.pure_fraction_by_element) == set(ELEMENTS)
    assert all(v == pytest.approx(0.2) for v in m.pure_fraction_by_element.values())


def test_uniform_dataset_has_high_coverage(uniform_df, pure_df):
    mu = analyze_dataset(uniform_df, name="u")
    mp = analyze_dataset(pure_df, name="p")
    assert mu.coverage_fraction > mp.coverage_fraction
    assert mu.composition_entropy > mp.composition_entropy
    assert mu.convex_hull_volume > mp.convex_hull_volume
    assert mu.pure_fraction < 0.01
    assert mu.quinary_fraction > 0.5


def test_homogeneous_dataset(homogeneous_df, uniform_df):
    mh = analyze_dataset(homogeneous_df, name="h")
    mu = analyze_dataset(uniform_df, name="u")
    assert mh.homogeneity_index > mu.homogeneity_index
    assert mh.quinary_fraction == pytest.approx(1.0)
    assert mh.n_clusters >= 1  # 単一の高密度クラスタ
    assert mh.clusters[0].fraction > 0.9


def test_known_phase_matching():
    rng = np.random.default_rng(1)
    comp = np.array([66.7, 0, 0, 0, 33.3]) + rng.normal(0, 0.5, size=(100, 5))
    comp = np.clip(comp, 0, None)
    df = pd.DataFrame(comp, columns=ELEMENTS)
    m = analyze_dataset(df, name="theta", known_phases=default_known_phases())
    assert m.n_clusters >= 1
    assert m.clusters[0].nearest_phase == "Al2Cu(θ)"
    assert m.clusters[0].nearest_phase_distance_at < 2.0


def test_spatial_gradient_computed_when_coords_present(rng):
    n = 300
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    al = np.clip(x, 0, 100)
    df = pd.DataFrame({"x_um": x, "y_um": y, "Al": al, "Cu": 100 - al, "Si": 0.0, "Ti": 0.0, "Fe": 0.0})
    m = analyze_dataset(df, name="grad")
    assert m.spatial_gradient_at_um is not None and m.spatial_gradient_at_um > 0
    m2 = analyze_dataset(df.drop(columns=["x_um", "y_um"]), name="nograd")
    assert m2.spatial_gradient_at_um is None


def test_analyze_missing_column_raises():
    with pytest.raises(ValueError):
        analyze_dataset(pd.DataFrame({"Al": [1.0]}))


def test_metrics_serialisable(uniform_df):
    m = analyze_dataset(uniform_df, name="u")
    d = m.to_dict()
    json.dumps(d)  # 例外が出なければ OK
    s = m.to_flat_series()
    assert s["name"] == "u"


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_simulation_outputs_valid_compositions():
    cond = GrainCondition(name="s", d50_um={el: 20.0 for el in ELEMENTS})
    cfg = SimulationConfig(n_points=300, field_size_um=200.0, random_state=3)
    r = simulate_condition(cond, cfg, n_pix=64)
    df = r.points
    assert len(df) == 300
    assert np.allclose(df[ELEMENTS].sum(axis=1), 100.0)
    assert (df[ELEMENTS] >= 0).all().all()
    assert r.grain_map.shape == (64, 64)
    assert r.comp_grid.shape == (5, 64, 64)
    assert set(r.diffusion_length_um) == set(ELEMENTS)
    assert (df["condition"] == "s").all()


def test_simulation_is_reproducible():
    cond = GrainCondition(name="s", d50_um={el: 20.0 for el in ELEMENTS})
    cfg = SimulationConfig(n_points=200, field_size_um=150.0, random_state=7)
    a = simulate_condition(cond, cfg, n_pix=48).points
    b = simulate_condition(cond, cfg, n_pix=48).points
    pd.testing.assert_frame_equal(a, b)


def test_finer_powder_reduces_pure_fraction():
    """物理的整合性: 微粉ほど未拡散領域が減り、多元素共存が増える。"""
    cfg = SimulationConfig(n_points=800, field_size_um=300.0, random_state=11)
    fine = GrainCondition(name="f", d50_um={el: 8.0 for el in ELEMENTS})
    coarse = GrainCondition(name="c", d50_um={el: 60.0 for el in ELEMENTS})
    mf = analyze_dataset(simulate_condition(fine, cfg, n_pix=96).points, name="f")
    mc = analyze_dataset(simulate_condition(coarse, cfg, n_pix=96).points, name="c")
    assert mf.pure_fraction < mc.pure_fraction
    assert mf.n_elements_mean > mc.n_elements_mean
    assert mf.coverage_fraction > mc.coverage_fraction


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_ranks_uniform_above_pure(uniform_df, pure_df, homogeneous_df):
    ms = [
        analyze_dataset(uniform_df, name="uniform"),
        analyze_dataset(pure_df, name="pure"),
        analyze_dataset(homogeneous_df, name="homog"),
    ]
    comp = compare_conditions(ms)
    assert comp.ranking[0] == "uniform"
    assert comp.scores.between(0, 100).all()
    assert "sdm_score" in comp.table.columns
    assert len(comp.recommendations) >= 1
    assert comp.trends == []  # 条件情報なし → トレンド無し


def test_compare_with_conditions_produces_trends():
    conds = default_conditions()
    cfg = SimulationConfig(n_points=400, field_size_um=250.0, random_state=5)
    ms = [analyze_dataset(simulate_condition(c, cfg, n_pix=64, seed=i).points, name=c.name) for i, c in enumerate(conds)]
    comp = compare_conditions(ms, conds)
    assert "mean_d50_um" in comp.table.columns
    preds = {t.predictor for t in comp.trends}
    assert "mean_d50_um" in preds
    t = next(t for t in comp.trends if t.predictor == "mean_d50_um" and t.metric == "pure_fraction")
    assert t.spearman_rho > 0  # 粗いほど未拡散が増える
    json.dumps(comp.to_dict())


def test_compare_empty_raises():
    with pytest.raises(ValueError):
        compare_conditions([])


# ---------------------------------------------------------------------------
# CLI / report (integration)
# ---------------------------------------------------------------------------

def test_cli_demo_generates_report(tmp_path: Path):
    out = tmp_path / "demo"
    rc = main(["demo", "-o", str(out), "--n-points", "300", "--n-pix", "48", "--seed", "1"])
    assert rc == 0
    assert (out / "report.html").exists()
    assert (out / "metrics_table.csv").exists()
    assert (out / "comparison.json").exists()
    assert (out / "figures" / "trends_vs_d50.png").exists()
    assert (out / "figures" / "simulation_maps.png").exists()
    assert len(list((out / "synthetic_data").glob("*.csv"))) == 5
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "総合ランキング" in html and "fine_10um" in html


def test_cli_simulate_then_analyze(tmp_path: Path):
    sim = tmp_path / "sim"
    rc = main(["simulate", "-o", str(sim), "-c", "configs/conditions_example.yaml", "--n-points", "250", "--n-pix", "48"])
    assert rc == 0
    files = sorted(p for p in sim.glob("*.csv") if p.stem != "all_conditions")
    assert len(files) == 5
    out = tmp_path / "ana"
    rc = main(["analyze", *map(str, files), "-c", "configs/conditions_example.yaml", "-o", str(out)])
    assert rc == 0
    tbl = pd.read_csv(out / "metrics_table.csv", index_col=0)
    assert set(tbl.index) == {f.stem for f in files}
    assert "mean_d50_um" in tbl.columns


def test_cli_analyze_single_file_with_condition_column(tmp_path: Path):
    sim = tmp_path / "sim"
    main(["simulate", "-o", str(sim), "--n-points", "200", "--n-pix", "40"])
    out = tmp_path / "ana"
    rc = main(["analyze", str(sim / "all_conditions.csv"), "-o", str(out)])
    assert rc == 0
    tbl = pd.read_csv(out / "metrics_table.csv", index_col=0)
    assert len(tbl) == 5
