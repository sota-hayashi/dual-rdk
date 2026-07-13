# Q学習モデル パラメータ推定 仕様書（ベイズ推定）

## 概要

各参加者の行動データからQ学習モデルのパラメータ $(\alpha, \beta)$ をベイズ推定（MCMC）により推定し，事後分布を得る。

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

## ベイズ推定（MCMC）

### 事前分布

| パラメータ | 事前分布 | 理由 |
|---|---|---|
| $\alpha$ | $\text{Beta}(2, 2)$ | $[0,1]$ に制約，中央付近に緩やかな山 |
| $\beta$ | $\text{Gamma}(2, 3)$ | 正値制約，過度に大きい値にペナルティ |

### サンプリング

事後分布からMCMCによりサンプルを生成：

$$P(\alpha, \beta \mid \{a_t, r_t\}) \propto \prod_t P(a_t \mid Q_t; \beta) \cdot P(\alpha) \cdot P(\beta)$$

- サンプリングアルゴリズム：`emcee`（アンサンブルサンプラー）または `PyMC`（NUTS）
- チェイン数：4
- バーンイン：1000 サンプル
- サンプリング：2000 サンプル（バーンイン後）
- $\alpha \in (0, 1)$，$\beta \in (0, \infty)$ の制約を事前分布により暗黙的に適用
- 収束診断：$\hat{R} < 1.05$，有効サンプルサイズ（ESS）の確認

---

## 出力

```python
results: pd.DataFrame
```

| カラム名 | 内容 |
|---|---|
| `subj_id` | 参加者ID |
| `alpha_mean` | $\alpha$ の事後平均 |
| `alpha_median` | $\alpha$ の事後中央値 |
| `alpha_hdi_low` | $\alpha$ の95% HDI 下限 |
| `alpha_hdi_high` | $\alpha$ の95% HDI 上限 |
| `beta_mean` | $\beta$ の事後平均 |
| `beta_median` | $\beta$ の事後中央値 |
| `beta_hdi_low` | $\beta$ の95% HDI 下限 |
| `beta_hdi_high` | $\beta$ の95% HDI 上限 |
| `r_hat_alpha` | $\alpha$ の $\hat{R}$ 統計量 |
| `r_hat_beta` | $\beta$ の $\hat{R}$ 統計量 |
| `log_likelihood` | 事後平均パラメータでの対数尤度 |
| `n_trials` | 使用試行数（NaN除外後） |

```python
traces: dict[str, np.ndarray]
```

| キー | 内容 |
|---|---|
| `subj_id` | 参加者IDのリスト |
| `alpha_samples` | 各参加者の $\alpha$ 事後サンプル（shape: n_subjects × n_samples） |
| `beta_samples` | 各参加者の $\beta$ 事後サンプル（shape: n_subjects × n_samples） |
