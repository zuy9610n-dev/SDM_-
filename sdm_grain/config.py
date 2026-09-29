"""設定データクラス群。

粒度条件 (GrainCondition)・解析パラメータ (AnalysisConfig)・
シミュレーションパラメータ (SimulationConfig) を定義し、YAML/JSON からの読込みも提供する。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 研究対象の 5 元系 (at% で扱う)
ELEMENTS: List[str] = ["Al", "Si", "Ti", "Fe", "Cu"]

# 純元素の代表的な融点 [K] (焼結・焼鈍温度に対する相対温度の推定に使用)
MELTING_POINTS_K: Dict[str, float] = {
    "Al": 933.5,
    "Si": 1687.0,
    "Ti": 1941.0,
    "Fe": 1811.0,
    "Cu": 1357.8,
}

# 5 元系相互拡散の「実効拡散係数」を粗く見積もる Arrhenius パラメータ
# D = D0 * exp(-Q / (R T))。個別文献値のオーダー (体積拡散, 固相) を代表させたもので、
# あくまで粒度依存性を相対比較するための簡略モデル用の既定値。
# 実測に基づいて override 可。
DEFAULT_D0_M2_S: float = 1.0e-4   # m^2/s
DEFAULT_Q_J_MOL: float = 2.0e5    # J/mol
R_GAS: float = 8.314462618        # J/(mol K)


@dataclass
class GrainCondition:
    """1 つの粒度調整条件を表す。

    Attributes
    ----------
    name : 条件名 (レポート表示に使用)
    d50_um : 各元素粉末の中位径 D50 [µm]。元素名 -> 値。
    span : 粒度分布幅 (D90-D10)/D50 の目安。元素名 -> 値。省略時は 1.0。
    sinter_temp_k : 焼結温度 [K]
    sinter_time_s : 焼結時間 [s]
    anneal_temp_k : 焼鈍温度 [K] (None なら焼鈍無し)
    anneal_time_s : 焼鈍時間 [s]
    note : 任意メモ
    """

    name: str
    d50_um: Dict[str, float]
    span: Dict[str, float] = field(default_factory=dict)
    sinter_temp_k: float = 873.0
    sinter_time_s: float = 3600.0
    anneal_temp_k: Optional[float] = None
    anneal_time_s: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        for el in self.d50_um:
            self.span.setdefault(el, 1.0)
        for el, v in self.d50_um.items():
            if v <= 0:
                raise ValueError(f"D50 must be positive: {el}={v}")

    @property
    def elements(self) -> List[str]:
        return list(self.d50_um.keys())

    @property
    def mean_d50_um(self) -> float:
        return sum(self.d50_um.values()) / len(self.d50_um)

    @property
    def d50_ratio(self) -> float:
        """最大 D50 / 最小 D50。粒度の元素間不均一さの指標。"""
        vals = list(self.d50_um.values())
        return max(vals) / min(vals)

    def thermal_budget(self, d0: float = DEFAULT_D0_M2_S, q: float = DEFAULT_Q_J_MOL) -> float:
        """焼結 + 焼鈍の積算拡散距離の 2 乗 (Σ D·t) [m^2]。"""
        import math

        total = d0 * math.exp(-q / (R_GAS * self.sinter_temp_k)) * self.sinter_time_s
        if self.anneal_temp_k is not None and self.anneal_time_s > 0:
            total += d0 * math.exp(-q / (R_GAS * self.anneal_temp_k)) * self.anneal_time_s
        return total

    def diffusion_length_um(self, d0: float = DEFAULT_D0_M2_S, q: float = DEFAULT_Q_J_MOL) -> float:
        """特性拡散距離 2√(ΣDt) [µm]。"""
        import math

        return 2.0 * math.sqrt(self.thermal_budget(d0, q)) * 1e6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GrainCondition":
        return cls(**d)


@dataclass
class AnalysisConfig:
    """解析パラメータ。

    Attributes
    ----------
    elements : 解析対象元素 (CSV の列名)
    bin_width_at : 組成空間をグリッド化するときの刻み [at%]
    min_points_per_bin : 「相候補 (高密度クラスタ)」判定の最小点数
    dbscan_eps_at : DBSCAN の近傍半径 [at%] (組成空間)
    dbscan_min_samples : DBSCAN の最小サンプル数
    pure_threshold_at : 主元素濃度がこの値以上なら「未拡散 (純元素残留)」とみなす [at%]
    interdiffusion_min_elements : 「相互拡散帯」とみなす最小有意元素数
    significant_at : 有意とみなす元素濃度 [at%]
    normalize : True なら各点の組成を合計 100 at% に正規化
    grid_size : 面内マップ描画時の格子サイズ
    random_state : 乱数シード
    """

    elements: List[str] = field(default_factory=lambda: list(ELEMENTS))
    bin_width_at: float = 5.0
    min_points_per_bin: int = 5
    dbscan_eps_at: float = 3.0
    dbscan_min_samples: int = 8
    pure_threshold_at: float = 95.0
    interdiffusion_min_elements: int = 3
    significant_at: float = 2.0
    normalize: bool = True
    grid_size: int = 64
    random_state: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisConfig":
        return cls(**d)


@dataclass
class SimulationConfig:
    """合成データ生成のパラメータ。

    Attributes
    ----------
    field_size_um : 解析視野の一辺 [µm]
    n_points : 生成する点分析数
    n_grains_scale : 視野内粒子数を決める係数 (視野面積 / (πD50²/4) に乗じる)
    eds_noise_at : EDS 定量ノイズの標準偏差 [at%]
    volume_fractions : 各元素粉末の体積比 (省略時は等量)
    intermetallic_phases : 拡散帯に形成される化合物相の代表組成リスト
    phase_attraction : 拡散帯の組成が化合物相に引き寄せられる強さ (0-1)
    random_state : 乱数シード
    """

    field_size_um: float = 500.0
    n_points: int = 2000
    n_grains_scale: float = 1.0
    eds_noise_at: float = 0.8
    volume_fractions: Optional[Dict[str, float]] = None
    intermetallic_phases: List[Dict[str, float]] = field(
        default_factory=lambda: [
            # Al-Si-Ti-Fe-Cu 系で予想される代表的な化合物相 (at%)
            {"Al": 66.7, "Cu": 33.3},                       # θ-Al2Cu
            {"Al": 75.0, "Ti": 25.0},                       # Al3Ti
            {"Al": 75.0, "Fe": 25.0},                       # Al13Fe4 近似
            {"Al": 60.0, "Fe": 20.0, "Si": 20.0},           # τ-AlFeSi 系
            {"Fe": 50.0, "Si": 50.0},                       # FeSi
            {"Ti": 33.3, "Si": 66.7},                       # TiSi2 近似
            {"Al": 50.0, "Cu": 25.0, "Fe": 25.0},           # Al7Cu2Fe 近似
        ]
    )
    phase_attraction: float = 0.35
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        return cls(**d)


# ---------------------------------------------------------------------------
# ファイル読み書き
# ---------------------------------------------------------------------------

def _load_structured(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_conditions(path: str | Path) -> List[GrainCondition]:
    """YAML/JSON ファイルから粒度条件リストを読み込む。

    フォーマット例 (YAML)::

        conditions:
          - name: fine
            d50_um: {Al: 10, Si: 10, Ti: 10, Fe: 10, Cu: 10}
            sinter_temp_k: 873
            sinter_time_s: 3600
    """
    data = _load_structured(Path(path))
    items = data.get("conditions", data) if isinstance(data, dict) else data
    return [GrainCondition.from_dict(item) for item in items]


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    data = _load_structured(Path(path))
    return AnalysisConfig.from_dict(data.get("analysis", data))


def load_simulation_config(path: str | Path) -> SimulationConfig:
    data = _load_structured(Path(path))
    return SimulationConfig.from_dict(data.get("simulation", data))


def default_conditions() -> List[GrainCondition]:
    """デモ用の既定粒度条件セット (5 条件)。"""
    base = dict(sinter_temp_k=873.0, sinter_time_s=3600.0, anneal_temp_k=823.0, anneal_time_s=7200.0)
    return [
        GrainCondition(
            name="fine_10um",
            d50_um={el: 10.0 for el in ELEMENTS},
            note="全元素 D50=10 µm の微粉。拡散が進みやすいが均質化しすぎる恐れ",
            **base,
        ),
        GrainCondition(
            name="medium_30um",
            d50_um={el: 30.0 for el in ELEMENTS},
            note="全元素 D50=30 µm",
            **base,
        ),
        GrainCondition(
            name="coarse_75um",
            d50_um={el: 75.0 for el in ELEMENTS},
            note="全元素 D50=75 µm の粗粉。未拡散領域が残りやすい",
            **base,
        ),
        GrainCondition(
            name="mixed_fineAlCu_coarseTiFe",
            d50_um={"Al": 15.0, "Si": 30.0, "Ti": 60.0, "Fe": 60.0, "Cu": 15.0},
            note="低融点元素を微粉、高融点元素を粗粉にした混合粒度",
            **base,
        ),
        GrainCondition(
            name="mixed_coarseAlCu_fineTiFe",
            d50_um={"Al": 60.0, "Si": 30.0, "Ti": 15.0, "Fe": 15.0, "Cu": 60.0},
            note="高融点元素を微粉にした逆配置",
            **base,
        ),
    ]
