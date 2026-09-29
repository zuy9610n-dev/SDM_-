"""EDS/EPMA 点分析データの入出力。

想定 CSV フォーマット (ヘッダ必須)::

    x_um, y_um, Al, Si, Ti, Fe, Cu [, condition]

* ``x_um``/``y_um`` : 試料面内座標 [µm] (省略可。無い場合は空間指標を計算しない)
* 元素列 : at% (合計が 100 でなくても normalize=True で正規化)
* ``condition`` : 任意。複数条件を 1 ファイルにまとめる場合の条件名

wt% で保存されたデータは ``convert_wt_to_at`` で at% に変換できる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ELEMENTS

# 原子量 [g/mol]
ATOMIC_MASS: Dict[str, float] = {
    "Al": 26.9815,
    "Si": 28.0855,
    "Ti": 47.867,
    "Fe": 55.845,
    "Cu": 63.546,
    "Mg": 24.305,
    "Mn": 54.938,
    "Ni": 58.693,
    "Zn": 65.38,
    "Cr": 51.996,
}

COORD_COLUMNS = ("x_um", "y_um")
CONDITION_COLUMN = "condition"


class CompositionDataError(ValueError):
    """入力データの形式不備。"""


def _resolve_element_columns(df: pd.DataFrame, elements: Sequence[str]) -> List[str]:
    """大文字小文字の違いや ' (at%)' 等のサフィックスを吸収して元素列名を解決する。"""
    lower_map = {c.lower().split()[0].split("(")[0].strip(): c for c in df.columns}
    resolved: List[str] = []
    missing: List[str] = []
    for el in elements:
        key = el.lower()
        if key in lower_map:
            resolved.append(lower_map[key])
        else:
            missing.append(el)
    if missing:
        raise CompositionDataError(f"元素列が見つかりません: {missing}. 列: {list(df.columns)}")
    return resolved


def normalize_compositions(df: pd.DataFrame, elements: Sequence[str]) -> pd.DataFrame:
    """各点の元素濃度を合計 100 at% に正規化する (合計 0 の行は除外)。"""
    out = df.copy()
    vals = out[list(elements)].to_numpy(dtype=float)
    total = vals.sum(axis=1)
    keep = total > 0
    vals = vals[keep]
    out = out.loc[keep].copy()
    out[list(elements)] = vals / total[keep, None] * 100.0
    return out.reset_index(drop=True)


def convert_wt_to_at(df: pd.DataFrame, elements: Sequence[str]) -> pd.DataFrame:
    """wt% 列を at% に変換して返す。"""
    out = df.copy()
    w = out[list(elements)].to_numpy(dtype=float)
    m = np.array([ATOMIC_MASS[e] for e in elements])
    mol = w / m
    total = mol.sum(axis=1, keepdims=True)
    total[total == 0] = np.nan
    out[list(elements)] = mol / total * 100.0
    return out.dropna(subset=list(elements)).reset_index(drop=True)


def load_composition_csv(
    path: str | Path,
    elements: Sequence[str] = ELEMENTS,
    normalize: bool = True,
    unit: str = "at",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """点分析 CSV を読み込み、標準化された DataFrame を返す。

    Parameters
    ----------
    path : CSV パス
    elements : 使用する元素列
    normalize : 合計 100 at% に正規化するか
    unit : "at" または "wt"
    drop_incomplete : NaN を含む行を落とすか

    Returns
    -------
    列 ``x_um``, ``y_um`` (存在すれば), 元素列, ``condition`` (存在すれば) を持つ DataFrame。
    元素列名は ``elements`` で与えた名前に統一される。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, comment="#")
    raw.columns = [str(c).strip() for c in raw.columns]

    cols = _resolve_element_columns(raw, elements)
    rename = dict(zip(cols, elements))

    # 座標列のゆらぎ (x, X, x_um, X (um)) を吸収
    for std in COORD_COLUMNS:
        base = std.split("_")[0]
        for c in raw.columns:
            cl = c.lower().replace(" ", "")
            if cl in {base, std, f"{base}(um)", f"{base}(µm)", f"{base}_µm", f"{base}[um]"}:
                rename[c] = std
                break
    df = raw.rename(columns=rename)

    keep = [c for c in COORD_COLUMNS if c in df.columns] + list(elements)
    if CONDITION_COLUMN in df.columns:
        keep.append(CONDITION_COLUMN)
    df = df[keep]

    df[list(elements)] = df[list(elements)].apply(pd.to_numeric, errors="coerce")
    if drop_incomplete:
        df = df.dropna(subset=list(elements))
    if (df[list(elements)] < 0).any().any():
        raise CompositionDataError("負の濃度値が含まれています")

    if unit.lower().startswith("wt"):
        df = convert_wt_to_at(df, elements)
    if normalize:
        df = normalize_compositions(df, elements)
    return df.reset_index(drop=True)


def save_composition_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    return path


def split_by_condition(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """``condition`` 列で分割する。列が無ければ {"all": df}。"""
    if CONDITION_COLUMN not in df.columns:
        return {"all": df}
    return {str(k): g.drop(columns=[CONDITION_COLUMN]).reset_index(drop=True) for k, g in df.groupby(CONDITION_COLUMN)}


def has_coordinates(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in COORD_COLUMNS)


def load_many(paths: Iterable[str | Path], elements: Sequence[str] = ELEMENTS, **kw) -> Dict[str, pd.DataFrame]:
    """複数 CSV を読み込み、ファイル名 (stem) をキーにした辞書を返す。"""
    path_list = [Path(p) for p in paths]
    out: Dict[str, pd.DataFrame] = {}
    for p in path_list:
        df = load_composition_csv(p, elements=elements, **kw)
        for name, sub in split_by_condition(df).items():
            if name == "all":
                key = p.stem
            elif len(path_list) > 1:
                key = f"{p.stem}:{name}"
            else:
                key = name
            out[key] = sub
    return out
