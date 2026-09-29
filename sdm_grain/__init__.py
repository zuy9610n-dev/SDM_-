"""
sdm_grain - 焼結拡散マルチプル法 (Sintered Diffusion Multiple, SDM) における
粉末粒度調整の影響を自動解析するツールキット。

対象系: Al-Si-Ti-Fe-Cu 5元系 (元素は設定で変更可)

主な機能
--------
* EDS/EPMA 点分析データ (CSV) の読み込みと検証
* 組成空間カバレッジ・情報エントロピー・点密度 (相候補) 解析
* 均質化度・相互拡散帯の割合など粒度依存指標の算出
* 粒度条件ごとの合成データ生成 (実データが無い段階の検討用)
* 条件間比較・ランキング・HTML レポート出力
"""

from .config import ELEMENTS, AnalysisConfig, GrainCondition, SimulationConfig
from .io import load_composition_csv, save_composition_csv
from .analysis import analyze_dataset, DatasetMetrics
from .simulate import simulate_condition
from .compare import compare_conditions
from .report import build_report

__version__ = "0.1.0"

__all__ = [
    "ELEMENTS",
    "AnalysisConfig",
    "GrainCondition",
    "SimulationConfig",
    "load_composition_csv",
    "save_composition_csv",
    "analyze_dataset",
    "DatasetMetrics",
    "simulate_condition",
    "compare_conditions",
    "build_report",
    "__version__",
]
