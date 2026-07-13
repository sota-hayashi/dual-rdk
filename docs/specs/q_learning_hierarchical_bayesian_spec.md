# Q学習モデル パラメータ推定 仕様書（階層ベイズ推定）

## 概要

全参加者の行動データを用いて，Q学習モデルのパラメータ $(\alpha_i, \beta_i)$ を階層ベイズモデルにより同時推定する。個人レベルのパラメータは集団レベルのハイパーパラメータから生成されるという構造を仮定し，少数試行でも安定した推定（shrinkage）を実現する。

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

$$Q_{\text{target},i}(0) = Q_{\text{dist},i}(0) = 0.5$$

### Q値の更新（選択した行動のみ更新）

$$Q_{a_t,i}(t+1) = Q_{a_t,i}(t) + \alpha_i \left[ r_t - Q_{a_t,i}(t) \right]$$

### 選択確率（softmax）

$$P(\text{target} \mid t, i) = \frac{e^{\beta_i Q_{\text{target},i}(t)}}{e^{\beta_i Q_{\text{target},i}(t)} + e^{\beta_i Q_{\text{dist},i}(t)}}$$

---

## 階層構造

### レベル1：個人パラメータ

パラメータの定義域制約を自然に扱うため，非制約空間（unbounded space）で階層構造を定義する。

$$\mu_i^{(\alpha)} = \text{logit}(\alpha_i), \quad \mu_i^{(\beta)} = \log(\beta_i)$$

$$\mu_i^{(\alpha)} \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)$$

$$\mu_i^{(\beta)} \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)$$

逆変換により個人パラメータを復元：

$$\alpha_i = \text{logistic}(\mu_i^{(\alpha)}), \quad \beta_i = \exp(\mu_i^{(\beta)})$$

### レベル2：集団ハイパーパラメータ

| ハイパーパラメータ | 事前分布 | 理由 |
|---|---|---|
| $\mu_\alpha$ | $\mathcal{N}(0, 1.5^2)$ | logit空間で広い範囲をカバー（$\alpha \approx 0.05$--$0.95$） |
| $\sigma_\alpha$ | $\text{Half-Cauchy}(0, 1)$ | 個人差の大きさに弱い情報を与える |
| $\mu_\beta$ | $\mathcal{N}(0.5, 1.5^2)$ | log空間で $\beta \approx 0.2$--$15$ をカバー |
| $\sigma_\beta$ | $\text{Half-Cauchy}(0, 1)$ | 個人差の大きさに弱い情報を与える |

### グラフィカルモデル

```
μ_α, σ_α    μ_β, σ_β        ← レベル2（集団）
    |            |
    v            v
  μ_i^(α)     μ_i^(β)        ← 非制約空間の個人パラメータ
    |            |
    v            v
   α_i          β_i           ← レベル1（個人，変換後）
    \           /
     \         /
      v       v
    Q_target_i(t), Q_dist_i(t)  ← Q値の時系列
          |
          v
    P(target | t, i)           ← 選択確率
          |
          v
        a_t,i                  ← 観測データ（選択行動）
```

---

## 推定方法

### サンプリング

事後分布全体からMCMCによりサンプルを生成：

$$P(\{\mu_i^{(\alpha)}, \mu_i^{(\beta)}\}, \mu_\alpha, \sigma_\alpha, \mu_\beta, \sigma_\beta \mid \{a_{t,i}, r_{t,i}\})$$

- 推奨ライブラリ：`PyMC`（NUTS）または `emcee` + カスタム対数事後確率関数
- チェイン数：4
- バーンイン（tune）：2000 サンプル
- サンプリング：4000 サンプル（バーンイン後）
- 収束診断：$\hat{R} < 1.05$，有効サンプルサイズ（ESS）> 400

### 実装上の注意

- Q値更新ループはTheano/PyTensorのscanと相性が悪い場合がある。その場合は `emcee` でカスタム対数尤度関数を書き，NumPyベースでQ値更新を計算するアプローチが安定する
- Non-centered parameterization（$\mu_i^{(\alpha)} = \mu_\alpha + \sigma_\alpha \cdot z_i^{(\alpha)},\ z_i^{(\alpha)} \sim \mathcal{N}(0,1)$）を用いることで，funnel問題を回避し，サンプリング効率を改善できる
- 参加者数が少ない場合（$N < 20$ 程度），$\sigma_\alpha, \sigma_\beta$ の推定が不安定になりやすいため，事前分布の選択に注意する

---

## 出力

### 集団レベル

```python
group_results: pd.DataFrame  # 1行
```

| カラム名 | 内容 |
|---|---|
| `mu_alpha_mean` | $\mu_\alpha$ の事後平均 |
| `mu_alpha_hdi_low` | $\mu_\alpha$ の95% HDI 下限 |
| `mu_alpha_hdi_high` | $\mu_\alpha$ の95% HDI 上限 |
| `sigma_alpha_mean` | $\sigma_\alpha$ の事後平均 |
| `sigma_alpha_hdi_low` | $\sigma_\alpha$ の95% HDI 下限 |
| `sigma_alpha_hdi_high` | $\sigma_\alpha$ の95% HDI 上限 |
| `mu_beta_mean` | $\mu_\beta$ の事後平均 |
| `mu_beta_hdi_low` | $\mu_\beta$ の95% HDI 下限 |
| `mu_beta_hdi_high` | $\mu_\beta$ の95% HDI 上限 |
| `sigma_beta_mean` | $\sigma_\beta$ の事後平均 |
| `sigma_beta_hdi_low` | $\sigma_\beta$ の95% HDI 下限 |
| `sigma_beta_hdi_high` | $\sigma_\beta$ の95% HDI 上限 |

### 個人レベル

```python
individual_results: pd.DataFrame  # N行（参加者数）
```

| カラム名 | 内容 |
|---|---|
| `subj_id` | 参加者ID |
| `alpha_mean` | $\alpha_i$ の事後平均 |
| `alpha_median` | $\alpha_i$ の事後中央値 |
| `alpha_hdi_low` | $\alpha_i$ の95% HDI 下限 |
| `alpha_hdi_high` | $\alpha_i$ の95% HDI 上限 |
| `beta_mean` | $\beta_i$ の事後平均 |
| `beta_median` | $\beta_i$ の事後中央値 |
| `beta_hdi_low` | $\beta_i$ の95% HDI 下限 |
| `beta_hdi_high` | $\beta_i$ の95% HDI 上限 |
| `r_hat_alpha` | $\alpha_i$ の $\hat{R}$ 統計量 |
| `r_hat_beta` | $\beta_i$ の $\hat{R}$ 統計量 |
| `shrinkage_alpha` | $\alpha_i$ の収縮度（個人推定からの移動量） |
| `shrinkage_beta` | $\beta_i$ の収縮度（個人推定からの移動量） |
| `n_trials` | 使用試行数（NaN除外後） |

### トレース

```python
traces: dict
```

| キー | 内容 |
|---|---|
| `subj_ids` | 参加者IDのリスト |
| `alpha_samples` | 各参加者の $\alpha_i$ 事後サンプル（shape: n_subjects × n_samples） |
| `beta_samples` | 各参加者の $\beta_i$ 事後サンプル（shape: n_subjects × n_samples） |
| `mu_alpha_samples` | $\mu_\alpha$ の事後サンプル（shape: n_samples） |
| `sigma_alpha_samples` | $\sigma_\alpha$ の事後サンプル（shape: n_samples） |
| `mu_beta_samples` | $\mu_\beta$ の事後サンプル（shape: n_samples） |
| `sigma_beta_samples` | $\sigma_\beta$ の事後サンプル（shape: n_samples） |
