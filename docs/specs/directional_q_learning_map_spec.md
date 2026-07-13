# 色別方向価値Q学習モデル（OOZ状態依存学習率）パラメータ推定 仕様書（MAP推定）

## 概要

各参加者の行動データから，**色ごとに定義した連続方向空間上の価値関数**の学習パラメータ $(\alpha_0, \alpha_1, \beta, c_{\text{black}})$ をMAP推定する．

学習率は各試行の **OOZ ラベル** $z_t \in \{0, 1\}$ に応じて切り替える（`q_learning_ooz_map` と同じ構成）：

- $z_t = 0$（非OOZ）：学習率 $\alpha_0$ で更新
- $z_t = 1$（OOZ）：学習率 $\alpha_1$ で更新

OOZ ラベルは `features.behavior.label_if_ooz` で事前付与しておく必要がある．

各試行で参加者は2つのドット群（白・黒）を見る．それぞれが運動方向 $\theta_{\text{white}}, \theta_{\text{black}}$ を持つ．参加者はどちらかの色を選び，その方向を報告する．報酬は **選んだ色（ターゲットか否か）と報告方向の正確さ（angular error）の両方**で決まり，`reward_points` にその両方が畳み込まれている．

本モデルは「参加者は報酬構造（どの色がターゲットになるかのパターン）に気づいていない」という研究仮定と整合するよう，**抽象的な報酬パターンのラベル（白高/黒高）をモデルに一切与えない**．モデルが受け取るのは知覚できる入力（2色とその運動方向）と報酬フィードバックのみであり，「どの色のどの方向が得か」を経験から創発的に学習する．

---

## 既存モデルとの関係

| | tabular Q学習（`q_learning_map`） | von Mises方向価値（`von_mises_value_learning_map`） | 本モデル（色別方向価値） |
|---|---|---|---|
| Q値の表現 | 表の4マス `Q[s,a]` | 方向の関数 `V(θ)`（色を無視） | **色ごとの方向の関数** `V_c(θ)` |
| 状態 | 真の報酬パターン（白高/黒高） | 運動方向のみ | 運動方向 ＋ 色 index |
| 般化 | なし（マス独立） | 方向に般化 | 方向に般化（色別に独立） |
| 色情報 | あり（ただし無自覚仮定と矛盾） | なし | **あり（パターンラベルは未使用）** |
| 無自覚仮定 | 矛盾 | 整合 | 整合 |

アルゴリズムは3モデルとも同じQ学習（TD誤差更新 ＋ softmax 選択）であり，違いは **Q値をどう表現するか** のみ．本モデルは「線形関数近似によるQ学習」（Sutton & Barto, Ch. 9–11）の色×方向特徴版に相当する．

---

## 入力

```python
concat_list: List[Tuple[str, pd.DataFrame]]
```

各タプルは `(subj_id, df)`．使用カラム：

| カラム名 | 型 | 内容 |
|---|---|---|
| `rt` | float | 反応時間．NaNの試行は除外 |
| `chosen_color` | str | 参加者が選んだ色 `"white"`/`"black"` |
| `response_angle_css` | float | 参加者が報告した方向（ラジアン $[0,2\pi)$） |
| `target_direction` | float | ターゲット（高報酬側）ドット群の運動方向（ラジアン） |
| `distractor_direction` | float | ディストラクタ（低報酬側）ドット群の運動方向（ラジアン） |
| `target_group` | str | ターゲットの色 `"white"`/`"black"`（**色別方向の復元にのみ使用**） |
| `reward_points` | int | 0〜10の報酬値（色×正確さが畳み込み済み）．$[0,1]$ にスケーリング |
| `ooz` | int | OOZ ラベル $z_t \in \{0,1\}$．`label_if_ooz` で事前付与．学習率の切り替えに使用 |

### 色別運動方向の復元

モデルの選択計算には各試行の **白の運動方向 $\theta_{\text{white}}$** と **黒の運動方向 $\theta_{\text{black}}$** が必要．これを `target_direction`/`distractor_direction` と `target_group` から復元する：

```
target_group == "white":  θ_white = target_direction,     θ_black = distractor_direction
target_group == "black":  θ_white = distractor_direction, θ_black = target_direction
```

> 注意：`target_group` は **色別方向の物理的な対応づけ**にのみ使う．これは知覚できる情報（白がどちらに動いていたか）であり，「どちらが高報酬か」という報酬パターンの情報はモデルの学習・選択には渡らない（報酬は `reward_points` というスカラーとしてのみ与えられる）．

---

## 前処理

1. `rt`, `chosen_color`, `ooz` のいずれかが NaN の試行を drop
2. `target_group` から `θ_white`, `θ_black` を上記ルールで復元
3. `reward_points` を $[0,1]$ にスケーリング
4. `ooz` を整数 $\{0,1\}$ に変換

---

## モデル定義

### von Mises基底関数（円環性の処理）

$N$ 個の基底を円環上に等間隔配置する：

$$\mu_j = \frac{2\pi j}{N}, \quad j = 0, 1, \ldots, N-1$$

$$\phi_j(\theta) = \exp\left[\kappa \cos(\theta - \mu_j)\right]$$

基底ベクトル $\boldsymbol{\phi}(\theta) = [\phi_0(\theta), \ldots, \phi_{N-1}(\theta)]^\top$．

> **円環性はここで解決される．** 角度 $\theta$ を生のラジアンのまま線形項に入れるのではなく，必ず周期 $2\pi$ の基底 $\boldsymbol{\phi}(\theta)$ に通す．これにより $0$ と $2\pi$ が同一に扱われ，近い方向ほど近い特徴ベクトルになる．「2角度を入れる」こと自体は円環性とは無関係で，円環性を解くのはこの基底符号化である．

推奨設定：$N = 4$（中心 0°, 90°, 180°, 270°），$\kappa = 2.0$．

### 色別価値関数

色 $c \in \{\text{white}, \text{black}\}$ ごとに重みベクトル $\mathbf{W}_c \in \mathbb{R}^N$ を持つ：

$$V_c(\theta) = \mathbf{W}_c \cdot \boldsymbol{\phi}(\theta)$$

### 重みの初期化

$$\mathbf{W}_{\text{white}}(0) = \mathbf{W}_{\text{black}}(0) = \mathbf{0}$$

全方向・全色に対して $V_c(\theta) = 0$ からスタート．

### 選択確率

各試行で，白の方向価値 $V_{\text{white}}(\theta_{\text{white}})$ と黒の方向価値 $V_{\text{black}}(\theta_{\text{black}})$ を比較する．色バイアス $c_{\text{black}}$ を加えて：

$$dq = \beta\left[V_{\text{black}}(\theta_{\text{black}}) - V_{\text{white}}(\theta_{\text{white}})\right] + c_{\text{black}}$$

$$P(\text{black} \mid t) = \sigma(dq) = \frac{1}{1 + e^{-dq}}, \qquad P(\text{white} \mid t) = 1 - P(\text{black} \mid t)$$

$c_{\text{black}} > 0$ は黒選好（既存 `q_learning_map` と同じ役割・符号規約）．

### 重みの更新

参加者が色 $c^\star$（`chosen_color`）を選び，方向 $\theta_{\text{reported}}$（`response_angle_css`）を報告し，報酬 $r_t$ を得たとき，**選んだ色の価値関数のみ**を報告方向で更新する．学習率は試行の OOZ ラベル $z_t$ で切り替える：

$$\alpha_t = \begin{cases} \alpha_0 & (z_t = 0,\ \text{非OOZ}) \\ \alpha_1 & (z_t = 1,\ \text{OOZ}) \end{cases}$$

$$\delta_t = r_t - V_{c^\star}(\theta_{\text{reported}})$$

$$\mathbf{W}_{c^\star} \leftarrow \mathbf{W}_{c^\star} + \alpha_t \cdot \delta_t \cdot \boldsymbol{\phi}(\theta_{\text{reported}})$$

選ばなかった色の重みは更新しない．$\boldsymbol{\phi}(\theta_{\text{reported}})$ が掛かることで，報告方向に近い基底ほど強く更新され，価値が方向空間に局所的に般化する．

> **報酬の二要因（色×正確さ）の扱い**：`reward_points` は「ターゲット色を選べたか」と「報告方向がどれだけ正確だったか」を畳み込んだ値なので，$V_{c}(\theta)$ は「その色をその方向で選んだときに**期待される報酬（典型的な正確さ込み）**」を学習することになる．角度誤差を別変数として明示的にモデル化する必要はない（スカラー報酬として吸収される）．

---

## パラメータ

### 推定するパラメータ

| パラメータ | 範囲 | 意味 |
|---|---|---|
| $\alpha_0$ | $(0, 1)$ | 非OOZ試行（$z_t=0$）の学習率 |
| $\alpha_1$ | $(0, 1)$ | OOZ試行（$z_t=1$）の学習率 |
| $\beta$ | $(0, \infty)$ | 逆温度（価値差の選択への反映感度） |
| $c_{\text{black}}$ | $(-\infty, \infty)$ | 色バイアス（黒選好の切片） |

パラメータ数は4で，既存 `q_learning_ooz_map` と同一．

### 固定するパラメータ

| パラメータ | 推奨値 | 意味 |
|---|---|---|
| $N$ | 4 | 基底関数の数 |
| $\kappa$ | 2.0 | 基底の集中度（般化の幅） |

48試行での推定可能性を優先し，$N, \kappa$ は固定．

---

## MAP推定

### 事前分布

| パラメータ | 事前分布 |
|---|---|
| $\alpha_0$ | $\text{Beta}(2, 2)$ |
| $\alpha_1$ | $\text{Beta}(2, 2)$ |
| $\beta$ | $\text{Gamma}(\text{shape}=2, \text{scale}=3)$ |
| $c_{\text{black}}$ | $\text{Normal}(0, 2)$ |

（既存 `q_learning_ooz_map` と同一．）

### 対数事後確率

$$\log P(\alpha_0, \alpha_1, \beta, c_{\text{black}} \mid \text{data}) \propto \sum_t \log P(c^\star_t \mid \theta_{\text{white},t}, \theta_{\text{black},t}, V_t) + \log P(\alpha_0) + \log P(\alpha_1) + \log P(\beta) + \log P(c_{\text{black}})$$

ここで尤度は `chosen_color` を直接用いる：$c^\star_t = \text{black}$ なら $P(\text{black})$，$c^\star_t = \text{white}$ なら $1 - P(\text{black})$．

### 手順

1. $(\alpha_0, \alpha_1, \beta, c_{\text{black}})$ の初期値をグリッドで生成
2. 各初期値について：
   a. $\mathbf{W}_{\text{white}}, \mathbf{W}_{\text{black}}$ を 0 に初期化
   b. 各試行 $t$：
      - $V_{\text{white}}(\theta_{\text{white},t})$, $V_{\text{black}}(\theta_{\text{black},t})$ を計算
      - $dq$ と $P(\text{black})$ を計算し，観測された `chosen_color` の対数確率を尤度に加算
      - $\delta_t = r_t - V_{c^\star}(\theta_{\text{reported},t})$ を計算
      - OOZ ラベル $z_t$ に応じて $\alpha_t \in \{\alpha_0, \alpha_1\}$ を選び，選んだ色の重みを更新
   c. 対数事後確率を計算
3. 負の対数事後を `scipy.optimize.minimize`（L-BFGS-B）で最小化
4. 複数初期値のうち最小を最適推定値とする

### 最適化の制約

- $\alpha_0 \in (10^{-6}, 1 - 10^{-6})$
- $\alpha_1 \in (10^{-6}, 1 - 10^{-6})$
- $\beta \in (10^{-6}, \infty)$
- $c_{\text{black}} \in (-\infty, \infty)$

---

## 出力

```python
results: pd.DataFrame
```

| カラム名 | 内容 |
|---|---|
| `subject` | 参加者ID |
| `alpha_0` | 非OOZ試行の推定学習率 |
| `alpha_1` | OOZ試行の推定学習率 |
| `beta` | 推定逆温度 |
| `c_black` | 推定色バイアス |
| `log_posterior` | 最大化された対数事後確率 |
| `log_likelihood` | 対数尤度（事前なし） |
| `n_trials` | 使用試行数 |
| `n_ooz` | OOZ試行数（$z_t=1$） |
| `n_non_ooz` | 非OOZ試行数（$z_t=0$） |
| `N` | 基底関数の数 |
| `kappa` | 集中度 |

---

## 事後分析：ターゲット選択確率の試行別復元

### 目的

推定パラメータで重み更新を再実行し，各試行のターゲット色選択確率を復元する．モデルはターゲット/パターン情報を学習に使っていないので，ターゲット選択率の上昇が**モデルの帰結として創発するか**を検証する．

### 手順

1. $\hat\alpha_0, \hat\alpha_1, \hat\beta, \hat{c}_{\text{black}}$ で各参加者の試行系列に沿って重みを再更新（更新時の学習率は各試行の $z_t$ で切り替え）
2. 各試行で $V_{\text{white}}(\theta_{\text{white}})$, $V_{\text{black}}(\theta_{\text{black}})$ を記録
3. `target_group` を使い，ターゲット色の選択確率を算出：
   - `target_group == "black"`：$P(\text{target}) = \sigma(dq)$
   - `target_group == "white"`：$P(\text{target}) = 1 - \sigma(dq)$
4. 全参加者で平均し，試行を通じた $P(\text{target})$ の推移を可視化
5. 実データのターゲット選択率の前後半差分とモデル復元値を比較

### 出力

| カラム名 | 内容 |
|---|---|
| `subject`, `trial` | 識別子 |
| `state` | `white_high`/`black_high`（target_group 由来，参考表示） |
| `chosen_color` | 実際の選択 |
| `ooz` | OOZ ラベル $z_t$ |
| `V_white`, `V_black` | 各色の現在の方向価値 |
| `p_black` | $P(\text{black})$ |
| `p_target` | ターゲット選択確率 |

---

## モデル評価

### 1. パラメータリカバリ

48試行・$N=4$ で $(\alpha_0, \alpha_1, \beta, c_{\text{black}})$ を同定できるか検証（OOZ試行数が少ない参加者では $\alpha_1$ の同定が難しい点に注意）．
- 真値を複数水準で設定し，実刺激系列で人工データを生成 → MAP推定 → 真値と推定値を比較
- 指標：Pearson相関，MAE，バイアス．相関 ≥ 0.7 かつ散布図が対角線に沿えば実用可

### 2. 事後予測チェック

$\hat\alpha, \hat\beta, \hat{c}_{\text{black}}$ と実刺激系列で多数回シミュレーション → ターゲット選択率の試行推移を実データと比較（95%区間に実データが収まるか）．

### 3. モデル比較

| 比較対象 | パラメータ数 | 指標 |
|---|---|---|
| tabular Q（`q_learning_map`） | 3 | BIC, AIC, WAIC |
| tabular Q OOZ（`q_learning_ooz_map`） | 4 | 同上 |
| von Mises 方向価値 | 2 | 同上 |
| 本モデル（色別方向価値・OOZ） | 4 | 同上 |

$\text{BIC} = -2\mathcal{L} + k\log T$．本モデルが「無自覚仮定と整合」かつ「tabular Q と同等以上の適合」を示せれば，矛盾を解消しつつ説明力を保てたと結論できる．

---

## 前提と未解決の論点（要確認）

本モデルが意味を持つかは「報酬パターン（どの色がターゲットになるか）が何によって決まるか」に依存する：

1. **ターゲット色が運動方向と結びついている場合**（例：特定方向に動く群が高報酬）
   → 方向を状態に入れる本モデルが、報酬構造を**方向価値として暗黙に発見**できる．最も適した状況．

2. **ターゲット色がブロック内で固定／方向と無関係の場合**
   → 方向価値はほぼ一様に上下し、色全体のレベル差（$\mathbf{W}_c$ の平均）と $c_{\text{black}}$ が色選好を担う．方向の般化はあまり効かず、tabular Q に対する優位は小さくなる可能性がある．

3. **ターゲット色が方向以外の知覚特徴（coherence 等）で決まる場合**
   → その特徴を基底に追加する拡張が必要（本モデルの方向基底だけでは不足）．

→ **確認したいこと：各試行でどちらの色がターゲット（高報酬）になるかは、何によって決まっているか？ 特に運動方向と関係しているか？**
