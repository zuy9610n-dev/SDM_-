# sdm_grain — 焼結拡散マルチプル法 粒度調整影響 自動解析プログラム

## 研究背景

私は⾼効率な材料探索⼿法である焼結拡散マルチプル法について研究しています。
材料を構成する元素数が増加するとそれに伴って調べる必要のある組成は急激に増加します。
例えば、5元素材料では元素濃度を1%ずつ変化させると約460万通りの組成となり、すべてを作製・評価することは困難です。
そこで、1つの試料から多数の組成を取得できる本⼿法を開発しています。
焼結拡散マルチプル法は、異なる組成の材料を焼結・焼鈍し、組成分析と点密度解析から状態図の概観を効率的に調べる手法です。
一つの試料から多数の組成情報を取得できるため、新規材料開発の高速化が期待されます。
しかし、適用は現在4元系までに限られているため、Al、Si、Ti、Fe、Cuからなる5元系を対象に、5元素粉末を焼結・焼鈍し、拡散を利⽤して多数の濃度の組合せを作り、組織観察・分析までを行います。
それにより、焼結拡散マルチプル法の5元系への適用条件の解明と新規物質探索を目指し研究しています。

条件の一つとして粒度調整を考えています。
そこで、粒度調整の影響を自動で分析するプログラムを作成しました。

---

## 概要

`sdm_grain` は、粉末粒度 (元素別 D50・分布幅) を変えて作製した焼結拡散マルチプル試料の
EDS/EPMA 点分析データを読み込み、**「その粒度条件が 5 元系 SDM 法にどれだけ適しているか」** を
定量指標と総合スコアで自動評価し、条件間比較・トレンド判定・所見を HTML レポートとして出力します。

実データが揃う前の実験計画段階でも使えるように、粒度条件から合成 EDS データを生成する
簡略シミュレータも同梱しています。

```
粒度条件 (YAML) ──▶ [simulate] ──▶ 合成 EDS CSV ─┐
                                                 ├──▶ [analyze] ──▶ 指標 ──▶ [compare] ──▶ report.html
実測 EDS/EPMA CSV ───────────────────────────────┘                          (ランキング・トレンド・所見・図)
```

## インストール

```bash
pip install -r requirements.txt        # numpy scipy pandas matplotlib scikit-learn pyyaml pytest
# または
pip install -e .                       # `sdm-grain` コマンドが使えるようになる
```

## クイックスタート

### 1. デモ (合成データ → 解析 → レポート)

```bash
python -m sdm_grain demo -o output/demo
# 出力: output/demo/report.html, metrics_table.csv, comparison.json, figures/*.png, synthetic_data/*.csv
```

既定では以下 5 条件を比較します (`configs/conditions_example.yaml` と同一)。

| 条件名 | Al | Si | Ti | Fe | Cu | 狙い |
|---|---|---|---|---|---|---|
| fine_10um | 10 | 10 | 10 | 10 | 10 | 微粉。拡散が進みやすいが過均質化の恐れ |
| medium_30um | 30 | 30 | 30 | 30 | 30 | 中間 |
| coarse_75um | 75 | 75 | 75 | 75 | 75 | 粗粉。未拡散領域が残りやすい |
| mixed_fineAlCu_coarseTiFe | 15 | 30 | 60 | 60 | 15 | 低融点元素を微粉、高融点元素を粗粉 |
| mixed_coarseAlCu_fineTiFe | 60 | 30 | 15 | 15 | 60 | 高融点 (低移動度) 元素を微粉にした逆配置 |

(単位 µm、D50)

### 2. 実測データの解析

条件ごとに 1 つの CSV、または `condition` 列を持つ 1 つの CSV を用意します。

```csv
x_um, y_um, Al, Si, Ti, Fe, Cu
12.3, 45.6, 62.1, 3.2, 0.5, 1.1, 33.1
...
```

* 元素列は at% (wt% の場合は `--unit wt`)。合計が 100 でなくても自動正規化。
* `x_um`, `y_um` は任意。あれば面内組成勾配やマップを出力。
* 列名は大小文字・`Al (at%)` などのサフィックスを許容。

```bash
# 条件ファイルを渡すと D50 依存トレンド (Spearman 相関) が計算される。
# CSV のファイル名 (拡張子除く) または condition 列の値を条件名と一致させてください。
python -m sdm_grain analyze data/fine_10um.csv data/coarse_75um.csv \
    -c configs/conditions_example.yaml -o output/real

# 単独ファイル (比較なし・指標のみ)
python -m sdm_grain analyze data/sample.csv -o output/single
```

### 3. 合成データのみ生成

```bash
python -m sdm_grain simulate -c configs/conditions_example.yaml -o output/sim --n-points 3000
```

### 4. Python API

```python
from sdm_grain import (GrainCondition, SimulationConfig, simulate_condition,
                       analyze_dataset, compare_conditions, build_report)

cond = GrainCondition("fine", d50_um={"Al":10,"Si":10,"Ti":10,"Fe":10,"Cu":10},
                      sinter_temp_k=873, sinter_time_s=3600)
sim = simulate_condition(cond, SimulationConfig(n_points=2000))
m   = analyze_dataset(sim.points, name="fine")
print(m.coverage_fraction, m.quinary_fraction, m.n_clusters)
```

## 算出指標

| 指標 | 意味 | SDM 法にとって |
|---|---|---|
| `coverage_fraction` | 5 at% 刻みで離散化した 5 元組成空間のうち、点分析が占有したビンの割合 | 大きいほど良い |
| `composition_entropy` | 占有ビンの点数分布の Shannon エントロピー [bit] | 大きいほど良い (多様) |
| `convex_hull_volume` | 相互拡散点の組成空間凸包体積 (探索範囲の広さ) | 大きいほど良い |
| `mean_nn_distance_at` | 組成空間での最近接点距離平均 [at%] | 小さいほど密 |
| `pure_fraction` | 主元素 ≥ 95 at% の「未拡散 (純元素残留)」点割合 | 小さいほど良い |
| `interdiffusion_fraction` | 有意元素 (≥ 2 at%) が 3 種以上の「相互拡散帯」点割合 | 大きいほど良い |
| `n_elements_mean` | 1 点あたり有意元素数の平均 | 大きいほど多元化 |
| **`quinary_fraction`** | **5 元素すべてが有意な点の割合 — 5 元系適用の核心指標** | 大きいほど良い |
| `homogeneity_index` | 1 − (平均組成からの平均距離 / 理論最大)。1 で完全均質 | 高すぎると情報喪失 (減点) |
| `spatial_gradient_at_um` | 面内近傍点間の組成勾配中央値 [at%/µm] | 参考 (拡散帯の幅) |
| `n_clusters` / `cluster_fraction` | DBSCAN による組成空間高密度クラスタ (相候補) 数と所属点割合 | 多いほど相が明瞭 |
| `sdm_score` | 上記を条件間で min-max 正規化し重み付き和した 0–100 の総合スコア | 相対評価 |

クラスタ重心は Al2Cu(θ), Al3Ti, Al13Fe4, β-Al5FeSi, FeSi, TiSi2, Al7Cu2Fe 等の既知化合物と照合し、
最近傍相と距離を表示します (`analysis.default_known_phases`)。

スコア重み・閾値は `sdm_grain.compare.ScoreWeights` で、解析パラメータ (ビン幅、閾値、DBSCAN 半径など) は
`configs/analysis_example.yaml` (`--analysis-config`) で変更できます。

## 自動判定される内容

* **ランキング** — `sdm_score` 順に条件を並べる。
* **トレンド** — 平均 D50・D50 比 (max/min)・特性拡散距離に対する各指標の Spearman 相関と
  log-線形勾配を計算し、`increase / decrease / flat` を判定。
* **所見 (日本語)** — 最良/最悪条件、粒度依存の方向、過均質化 (index > 0.85) や未拡散過多
  (pure_fraction > 0.5) の警告、均一粒度 vs 混合粒度の比較、5 元素共存割合最大条件。

## シミュレータの物理モデル (簡略)

1. **粒子配置** — 元素別 D50 と体積比から粒子数を決め、加法重み付き Voronoi (power diagram) で充填。
   粒径は対数正規分布 (`span` → σ)。
2. **拡散** — 各元素の指示関数を実効拡散距離 L_i = 2√(ΣDt)·mobility_i·(D50_ref/D50_i)^0.25 の
   Gaussian で平滑化 (Fick 拡散の Green 関数近似)。Al, Cu は速く、Ti, Fe は遅い。
   微粉ほど接触点が増えて焼結接合が進む効果を指数 0.25 で表現。
3. **化合物形成** — 相互拡散帯の組成を最寄りの既知化合物組成へ `phase_attraction` の強さで引き寄せ、
   点密度ピークを再現。
4. **測定** — 視野内ランダム点で組成を読み取り、EDS 定量ノイズ (σ=0.8 at%) を付加。

> 注意: 定性的な傾向比較を目的とした簡略モデルです。Arrhenius パラメータや移動度は
> `sdm_grain.simulate` の定数で調整でき、実測データによる検証が必要です。

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `report.html` | 自己完結の HTML レポート (図は base64 埋め込み) |
| `metrics_table.csv` | 条件 × 指標の表 (粒度パラメータ・スコア含む) |
| `comparison.json` | ランキング・トレンド・所見・重み |
| `metrics_detail.json` | クラスタ情報などを含む詳細指標 |
| `figures/*.png` | 指標棒グラフ、D50 トレンド、元素数分布、組成射影、面内マップ、クラスタ組成、シミュレーションマップ |
| `synthetic_data/*.csv` | (demo/simulate) 生成した合成 EDS データ |

## ディレクトリ構成

```
sdm_grain/
  config.py     粒度条件・解析/シミュレーション設定 (dataclass, YAML/JSON 読込)
  io.py         EDS/EPMA CSV 入出力、wt%→at% 変換、正規化
  analysis.py   指標算出 (カバレッジ・拡散/均質化・DBSCAN 相候補)
  simulate.py   粒度条件 → 合成 EDS データ
  compare.py    条件間比較・スコア・トレンド・所見
  report.py     図と HTML レポート
  cli.py        simulate / analyze / demo サブコマンド
configs/        条件・解析設定の例 (YAML)
tests/          pytest (26 tests)
```

## テスト

```bash
python -m pytest tests -q
```

## 今後の拡張候補

* 実測データでの拡散長・移動度の同定 (シミュレータのキャリブレーション)
* 三角図・四面体図など多元系可視化の追加
* 焼結温度・時間・圧力など粒度以外の条件を説明変数に含めた回帰モデル
* SEM 画像からの粒径分布自動抽出との連携
