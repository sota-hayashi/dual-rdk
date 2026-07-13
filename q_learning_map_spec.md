# Q学習モデル パラメータ推定 仕様書

## 概要

各参加者の行動データからQ学習モデルのパラメータ $(\alpha, \beta)$ をMAP推定により推定する。

---

## 入力

```python
concat_list: List[Tuple[str, pd.DataFrame]]
```

各タプルは `(subj_id, df)` の形式。DataFrameの使用カラムは以下の通り：

| カラム名 | 型 | 内容 |
|---|---|---|
| `rt` | float | 反応時間．NaNの試行は除外 |
| `chosen_item` | int | 1=ターゲット選択，0=ディストラクター選択 |
| `reward_points` | int | 0〜10の報酬値．**1以上を1，0を0に二値化** |

---

## 前処理

1. `rt` が NaN の試行を drop
2. `reward_points` を二値化：$r_t = \mathbb{1}[\text{reward\_points} \geq 1]$

---

## モデル定義

### Q値の初期化

$$Q_{\text{target}}(0) = Q_{\text{dist}}(0) = 0.5$$

### Q値の更新（選択した行動のみ更新）

$$Q_{a_t}(t+1) = Q_{a_t}(t) + \alpha \left[ r_t - Q_{a_t}(t) \right]$$

### 選択確率（softmax）

$$P(\text{target} \mid t) = \frac{e^{\beta Q_{\text{target}}(t)}}{e^{\beta Q_{\text{target}}(t)} + e^{\beta Q_{\text{dist}}(t)}}$$

---

## MAP推定

### 事前分布

| パラメータ | 事前分布 | 理由 |
|---|---|---|
| $\alpha$ | $\text{Beta}(2, 2)$ | $[0,1]$ に制約，中央付近に緩やかな山 |
| $\beta$ | $\text{Gamma}(2, 3)$ | 正値制約，過度に大きい値にペナルティ |

### 最適化

対数事後確率を最大化（＝負の対数事後確率を最小化）：

$$\hat{\alpha}, \hat{\beta} = \arg\max_{\alpha, \beta} \left[ \sum_t \log P(a_t \mid Q_t) + \log P(\alpha) + \log P(\beta) \right]$$

- 最適化アルゴリズム：`scipy.optimize.minimize`，メソッド `L-BFGS-B`
- 複数初期値（グリッドサーチ）で局所解を回避
- $\alpha \in (0, 1)$，$\beta \in (0, \infty)$ の制約を適用

---

## 出力

```python
results: pd.DataFrame
```

| カラム名 | 内容 |
|---|---|
| `subj_id` | 参加者ID |
| `alpha` | 推定された学習率 |
| `beta` | 推定された逆温度 |
| `log_posterior` | 最大化された対数事後確率 |
| `log_likelihood` | 対数尤度（事前分布なし） |
| `n_trials` | 使用試行数（NaN除外後） |
