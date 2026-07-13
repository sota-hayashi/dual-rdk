# Rescorla-Wagnerモデル パラメータ推定 仕様書（階層ベイズ推定）

## 概要

全参加者の行動データを用いて，刺激ベースのRescorla-Wagner（RW）モデルのパラメータ $(\alpha_i, \beta_i)$ を階層ベイズモデルにより同時推定する。Q学習モデルが「ターゲット vs ディストラクター」という行動に価値を割り当てるのに対し，RWモデルは「白色RDK vs 黒色RDK」という知覚的に利用可能な刺激に連合強度を割り当てる。

---

## 入力

```python
concat_list: List[Tuple[str, pd.DataFrame]]
```

各タプルは `(subj_id, df)` の形式。DataFrameの使用カラムは以下の通り：

| カラム名 | 型 | 内容 |
|---|---|---|
| `rt` | float | 反応時間．NaNの試行は除外 |
| `chosen_color` | str | `"white"` または `"black"`：参加者が報告した方向に近い色 |
| `reward_points` | int | 0〜10の報酬値．**1以上を1，0を0に二値化** |

### `chosen_item` との対応

元データの `chosen_item`（1=ターゲット，0=ディストラクター）から `chosen_color` への変換は，試行ごとの刺激パターンに依存する：

| パターン | ターゲット（chosen_item=1） | ディストラクター（chosen_item=0） |
|---|---|---|
| W_H & B_L | white | black |
| B_H & W_L | black | white |

---

## 前処理

1. `rt` が NaN の試行を drop
2. `reward_points` を二値化：$r_t = \mathbb{1}[\text{reward\_points} \geq 1]$
3. `chosen_item` と試行パターンから `chosen_color` を導出（上表に基づく）

---

## モデル定義

### 連合強度の初期化

$$V_{\text{white},i}(0) = V_{\text{black},i}(0) = 0.5$$

### 連合強度の更新（選択した刺激のみ更新）

参加者 $i$ が試行 $t$ で色 $c_t \in \{\text{white}, \text{black}\}$ を選択し，報酬 $r_t$ を受け取ったとき：

$$V_{c_t, i}(t+1) = V_{c_t, i}(t) + \alpha_i \left[ r_t - V_{c_t, i}(t) \right]$$

選択されなかった色の連合強度は変化しない：

$$V_{\bar{c}_t, i}(t+1) = V_{\bar{c}_t, i}(t)$$

### 選択確率（softmax）

$$P(\text{white} \mid t, i) = \frac{e^{\beta_i V_{\text{white},i}(t)}}{e^{\beta_i V_{\text{white},i}(t)} + e^{\beta_i V_{\text{black},i}(t)}}$$

---

## パラメータの意味

| パラメータ | 意味 | Q学習との対比 |
|---|---|---|
| $\alpha_i$ | 連合強度の学習率：報酬予測誤差に対する更新の速さ | Q学習と同一の役割 |
| $\beta_i$ | 逆温度：連合強度の差が選択にどれだけ反映されるか | Q学習と同一の役割 |
| $V_{\text{white}}$ | 白色RDKと報酬の連合強度 | $Q_{\text{target/dist}}$ に対応するが帰属先が刺激 |
| $V_{\text{black}}$ | 黒色RDKと報酬の連合強度 | 同上 |

---

## 階層構造

### レベル1：個人パラメータ

$$\mu_i^{(\alpha)} = \text{logit}(\alpha_i), \quad \mu_i^{(\beta)} = \log(\beta_i)$$

$$\mu_i^{(\alpha)} \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^2)$$

$$\mu_i^{(\beta)} \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)$$

逆変換：

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
μ_α, σ_α    μ_β, σ_β           ← レベル2（集団）
    |            |
    v            v
  μ_i^(α)     μ_i^(β)           ← 非制約空間の個人パラメータ
    |            |
    v            v
   α_i          β_i              ← レベル1（個人，変換後）
    \           /
     \         /
      v       v
  V_white_i(t), V_black_i(t)    ← 連合強度の時系列
          |
          v
    P(white | t, i)              ← 選択確率
          |
          v
        c_t,i                    ← 観測データ（色の選択）
```

---

## 推定方法

### サンプリング

$$P(\{\mu_i^{(\alpha)}, \mu_i^{(\beta)}\}, \mu_\alpha, \sigma_\alpha, \mu_\beta, \sigma_\beta \mid \{c_{t,i}, r_{t,i}\})$$

- 推奨ライブラリ：`emcee`（アンサンブルサンプラー）+ カスタム対数事後確率関数
- チェイン数：4
- バーンイン（tune）：2000 サンプル
- サンプリング：4000 サンプル（バーンイン後）
- 収束診断：$\hat{R} < 1.05$，有効サンプルサイズ（ESS）> 400
- Non-centered parameterization を推奨

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
| `shrinkage_alpha` | $\alpha_i$ の収縮度 |
| `shrinkage_beta` | $\beta_i$ の収縮度 |
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

---

## Q学習モデルとの比較

### モデル比較指標

両モデルの適合度を以下の指標で比較する：

| 指標 | 計算方法 | 判断基準 |
|---|---|---|
| WAIC | 事後サンプルから試行ごとの対数尤度を計算 | 低い方が良い |
| LOO-CV | Pareto smoothed importance sampling による近似 | 低い方が良い |

### 比較の解釈

- RWモデルの $\beta$ がQ学習より大きい → 刺激ベースの帰属の方が行動を説明できている
- 両モデルとも $\beta \approx 0$ → 試行ごとの更新メカニズム自体が不適切（WSLSやマッチングモデルを検討）
- RWモデルの $\alpha$ のHDIがQ学習より狭い → 刺激ベースの方がパラメータを同定できている
