# 連続ジョイント方向価値Q学習モデル（OOZ状態依存学習率）パラメータ推定 仕様書（MAP推定）

## 概要

各参加者の行動データから，**白・黒ドット群の運動方向を1つの連続状態ベクトルに統合（ジョイント）した行動価値関数**の学習パラメータ $(\alpha_0, \alpha_1, \beta, c_{\text{black}})$ をMAP推定する．

既存の色別方向価値モデル（`directional_q_learning_map`）が「白の価値は $\theta_{\text{white}}$ のみ，黒の価値は $\theta_{\text{black}}$ のみ」に依存する **分離（factorized）** 構造だったのに対し，本モデルは **1つの共有された状態 $s=(\theta_{\text{white}}, \theta_{\text{black}})$ を両行動が参照する** 構造を取る．これにより，white を選ぶ価値が $\theta_{\text{black}}$ にも依存しうる（＝白と黒の方向が相互作用しうる）．

学習率は各試行の **OOZ ラベル** $z_t \in \{0,1\}$ で切り替える（`q_learning_ooz_map`，`directional_q_learning_map` と同一構成）：

- $z_t = 0$（非OOZ）：学習率 $\alpha_0$
- $z_t = 1$（OOZ）：学習率 $\alpha_1$

本モデルも分離モデルと同様に，**抽象的な報酬パターンのラベル（白高/黒高）をモデルに一切与えない**（無自覚仮定と整合）．モデルが受け取るのは知覚できる入力（2色の運動方向）と報酬フィードバックのみ．

---

## 設計の動機：分離モデルとの差分は「ジョイント構造のみ」

本モデルの存在意義は，**分離モデルを厳密に内包（nested）する上位モデル**として作り，両者の汎化性能差から「白と黒の方向が相互作用しているか」を検定することにある．

| | ワンホット（`q_learning_ooz_map`） | 分離方向価値（`directional_q_learning_map`） | **本モデル（連続ジョイント）** |
|---|---|---|---|
| 状態 | 共有 $s\in\{0,1\}$（オラクル1ビット） | 行動ごとに分割（$\theta_w$ / $\theta_b$） | **共有 $s=(\theta_w,\theta_b)$（連続）** |
| ジョイント性 | ジョイント（退化） | 非ジョイント（分離） | **ジョイント（豊か）** |
| 状態の出所 | 与えられる（無自覚仮定と矛盾） | 観測から符号化 | 観測から符号化 |
| 方向の相互作用 | — | 表現不可 | **表現可能** |
| 潜在重み数 | $2\times2$ | $N\times2$ | $N\times N\times 2$ |
| 自由パラメータ | 4 | 4 | 4 |

> **核心**：分離モデルは「ジョイント重み行列の非対角（交差項）をゼロに固定した特殊ケース」である．したがって本モデルと分離モデルの当てはまり差は，**追加された交差項＝方向間相互作用**のみに由来する．この差を比較することで，「白と黒の方向が影響し合うか」を定量的に問える（→ モデル評価セクション）．

---

## 入力

```python
concat_list: List[Tuple[str, pd.DataFrame]]
```

各タプルは `(subj_id, df)`．使用カラムは `directional_q_learning_map` と完全に同一：

| カラム名 | 型 | 内容 |
|---|---|---|
| `rt` | float | 反応時間．NaNの試行は除外 |
| `chosen_color` | str | 参加者が選んだ色 `"white"`/`"black"` |
| `response_angle_css` | float | 参加者が報告した方向（度数法 $[0,360)$，内部でラジアン変換） |
| `target_direction` | float | ターゲット（高報酬側）ドット群の運動方向 |
| `distractor_direction` | float | ディストラクタ（低報酬側）ドット群の運動方向 |
| `target_group` | str | ターゲットの色 `"white"`/`"black"`（**色別方向の復元にのみ使用**） |
| `reward_points` | int | 0〜10の報酬値（色×正確さが畳み込み済み）．$[0,1]$ にスケーリング |
| `ooz` | int | OOZ ラベル $z_t\in\{0,1\}$．`label_if_ooz` で事前付与 |

### 色別運動方向の復元

```
target_group == "white":  θ_white = target_direction,     θ_black = distractor_direction
target_group == "black":  θ_white = distractor_direction, θ_black = target_direction
```

> `target_group` は知覚できる物理的対応づけ（白がどちらに動いていたか）にのみ使用し，報酬パターン情報は学習・選択に渡さない．

---

## 前処理

1. `rt`, `chosen_color`, `ooz` のいずれかが NaN の試行を drop
2. `target_group` から $\theta_{\text{white}}, \theta_{\text{black}}$ を復元
3. 角度を度数法 → ラジアンに変換
4. `reward_points` を $[0,1]$ にスケーリング
5. `ooz` を整数 $\{0,1\}$ に変換

---

## モデル定義

### von Mises基底関数

分離モデルと同一の基底を用いる．$N$ 個の基底を円環上に等間隔配置：

$$\mu_j = \frac{2\pi j}{N}, \quad j = 0, \ldots, N-1, \qquad \phi_j(\theta) = \exp[\kappa\cos(\theta - \mu_j)]$$

基底ベクトル $\boldsymbol{\phi}(\theta) = [\phi_0(\theta), \ldots, \phi_{N-1}(\theta)]^\top \in \mathbb{R}^N$．

推奨設定：$N=4$，$\kappa=2.0$．

### ジョイント状態特徴（外積基底）

白・黒の方向を**1本の状態特徴ベクトル**に統合する．von Mises 基底の**外積（クロネッカー積）**を用いる：

$$\boldsymbol{\Phi}(\theta_{\text{white}}, \theta_{\text{black}}) = \boldsymbol{\phi}(\theta_{\text{white}}) \otimes \boldsymbol{\phi}(\theta_{\text{black}}) \in \mathbb{R}^{N^2}$$

成分で書くと $\Phi_{jk} = \phi_j(\theta_{\text{white}})\,\phi_k(\theta_{\text{black}})$．これは2方向の同時生起（white が方向 $\mu_j$ 付近 **かつ** black が方向 $\mu_k$ 付近）を表すタイルであり，方向間の相互作用を符号化できる．

### ジョイント行動価値関数

行動 $a \in \{\text{white}, \text{black}\}$ ごとに重み行列 $\mathbf{W}_a \in \mathbb{R}^{N\times N}$（展開して $\mathbb{R}^{N^2}$）を持つ：

$$Q_a(\theta_{\text{white}}, \theta_{\text{black}}) = \mathbf{W}_a \cdot \boldsymbol{\Phi}(\theta_{\text{white}}, \theta_{\text{black}})$$

両行動の価値が **同一のジョイント状態 $\boldsymbol{\Phi}$** を参照する点が，分離モデルとの構造的差分である．

### 重みの初期化

$$\mathbf{W}_{\text{white}}(0) = \mathbf{W}_{\text{black}}(0) = \mathbf{0}$$

### 選択確率

$$dq = \beta\left[Q_{\text{black}}(\theta_w,\theta_b) - Q_{\text{white}}(\theta_w,\theta_b)\right] + c_{\text{black}}$$

$$P(\text{black}\mid t) = \sigma(dq) = \frac{1}{1+e^{-dq}}, \qquad P(\text{white}\mid t) = 1 - P(\text{black}\mid t)$$

符号規約は分離モデル・`q_learning_map` と同一（$c_{\text{black}}>0$ は黒選好）．

### 重みの更新

参加者が色 $a^\star$（`chosen_color`）を選び方向 $\theta_{\text{reported}}$ を報告し報酬 $r_t$ を得たとき，**選んだ行動の価値のみ**を更新する．学習率は OOZ ラベルで切り替える：

$$\alpha_t = \begin{cases}\alpha_0 & (z_t=0)\\ \alpha_1 & (z_t=1)\end{cases}$$

更新に用いる状態特徴は，**選択時に呈示されたジョイント状態 $\boldsymbol{\Phi}(\theta_w, \theta_b)$** を用いる（分離モデルとの厳密なネストを保つための設計，後述）：

$$\delta_t = r_t - Q_{a^\star}(\theta_w, \theta_b)$$

$$\mathbf{W}_{a^\star} \leftarrow \mathbf{W}_{a^\star} + \alpha_t\,\delta_t\,\boldsymbol{\Phi}(\theta_w, \theta_b)$$

選ばなかった行動の重みは更新しない．

> **報告方向 $\theta_{\text{reported}}$ の扱い（要設計判断）**：分離モデルは更新を $\boldsymbol{\phi}(\theta_{\text{reported}})$ で行っていた（報告方向の正確さを価値に取り込むため）．ジョイント版でこれと整合させる方法は2案ある：
> - **(A) 呈示状態で更新**（上式）：更新も評価も同じ $\boldsymbol{\Phi}(\theta_w,\theta_b)$ を使う．標準的な Q学習に最も近く，ネスト関係が最もクリーン．**デフォルト推奨**．
> - **(B) 報告方向を白側に差し込む**：$a^\star=\text{white}$ なら $\boldsymbol{\Phi}(\theta_{\text{reported}}, \theta_b)$，$a^\star=\text{black}$ なら $\boldsymbol{\Phi}(\theta_w, \theta_{\text{reported}})$ で更新．分離モデルの「報告方向で般化」の挙動をジョイントに持ち上げた版．分離モデルとのネストを厳密に保ちたい場合はこちら．
>
> **モデル比較の妥当性は，分離モデルと本モデルで更新則を一致させること（A同士／B同士で比べる）に依存する．** 詳細はモデル評価セクション参照．

---

## 分離モデルとのネスト関係（理論的確認）

ジョイント重み $\mathbf{W}_a \in \mathbb{R}^{N\times N}$ を，「$\theta_b$ 方向に定数（$k$ について一様）」となるよう制約すると：

$$\mathbf{W}_a = \mathbf{w}_a \, \mathbf{1}^\top \quad\Rightarrow\quad Q_a = (\mathbf{w}_a\cdot\boldsymbol{\phi}(\theta_w))\,\underbrace{(\mathbf{1}\cdot\boldsymbol{\phi}(\theta_b))}_{\text{$\theta_b$ に依存する定数}}$$

として white の価値が実質 $\theta_w$ のみに依存する形に縮約される（black も対称）．すなわち **分離モデルはジョイント重み行列の交差成分をゼロ化した部分空間に対応する nested な特殊ケース**．逆に，本モデルが分離モデルを有意に上回るなら，その説明力は交差成分＝方向間相互作用に帰属する．

---

## パラメータ

### 推定するパラメータ（分離モデル・OOZ版と同一）

| パラメータ | 範囲 | 意味 |
|---|---|---|
| $\alpha_0$ | $(0,1)$ | 非OOZ試行の学習率 |
| $\alpha_1$ | $(0,1)$ | OOZ試行の学習率 |
| $\beta$ | $(0,\infty)$ | 逆温度 |
| $c_{\text{black}}$ | $(-\infty,\infty)$ | 色バイアス |

自由パラメータ数は4で，比較対象の各モデルと同一．

### 固定するパラメータ

| パラメータ | 推奨値 | 意味 |
|---|---|---|
| $N$ | 4 | 基底関数の数（ジョイント特徴は $N^2=16$ 次元） |
| $\kappa$ | 2.0 | 基底の集中度 |

> **注意（複雑さの非対称性）**：本モデルの潜在重みは $N^2\times2=32$ 個で，分離モデルの $N\times2=8$ 個より多い．自由パラメータ（最適化対象）は両者とも4で同じだが，**潜在重みの実効的柔軟性は本モデルの方が大きい**．このため in-sample 尤度・AIC・BIC では本モデルが構造的に有利になり，公正な比較にならない（→ モデル評価で交差検証を用いる）．

---

## MAP推定

### 事前分布（既存モデルと同一）

| パラメータ | 事前分布 |
|---|---|
| $\alpha_0$ | $\text{Beta}(2,2)$ |
| $\alpha_1$ | $\text{Beta}(2,2)$ |
| $\beta$ | $\text{Gamma}(\text{shape}=2,\text{scale}=3)$ |
| $c_{\text{black}}$ | $\text{Normal}(0,2)$ |

### 対数事後確率

$$\log P(\boldsymbol{\theta}\mid\text{data}) \propto \sum_t \log P(a^\star_t \mid \theta_{w,t},\theta_{b,t}, \mathbf{W}_t) + \sum \log P(\text{prior})$$

### 手順

1. $(\alpha_0,\alpha_1,\beta,c_{\text{black}})$ の初期値をグリッドで生成
2. 各初期値について：
   a. $\mathbf{W}_{\text{white}}, \mathbf{W}_{\text{black}}$ を $\mathbf{0}$ に初期化
   b. 各試行 $t$：
      - $\boldsymbol{\Phi}_t = \boldsymbol{\phi}(\theta_{w,t})\otimes\boldsymbol{\phi}(\theta_{b,t})$ を計算
      - $Q_{\text{white}}, Q_{\text{black}}$ を計算し $dq$，$P(\text{black})$ を求め，観測 `chosen_color` の対数確率を尤度に加算
      - $\delta_t = r_t - Q_{a^\star}$ を計算
      - OOZ ラベル $z_t$ に応じた $\alpha_t$ で選んだ行動の重みを更新
   c. 対数事後を計算
3. 負の対数事後を `scipy.optimize.minimize`（L-BFGS-B）で最小化
4. 複数初期値のうち最小を最適推定値とする

### 最適化の制約

- $\alpha_0, \alpha_1 \in (10^{-6}, 1-10^{-6})$
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
| `alpha_0`, `alpha_1` | 推定学習率（非OOZ / OOZ） |
| `beta` | 推定逆温度 |
| `c_black` | 推定色バイアス |
| `log_posterior` | 最大化された対数事後 |
| `log_likelihood` | 対数尤度（事前なし，in-sample） |
| `cv_log_likelihood` | **交差検証 out-of-sample 対数尤度**（モデル比較の主指標） |
| `n_trials` | 使用試行数 |
| `n_ooz`, `n_non_ooz` | OOZ / 非OOZ 試行数 |
| `N`, `kappa` | 基底設定 |

---

## モデル評価

### 1. パラメータリカバリ

48試行・$N=4$（潜在 $N^2$ 次元）で $(\alpha_0,\alpha_1,\beta,c_{\text{black}})$ を同定できるか検証．分離モデルより潜在自由度が大きいため，**少試行で交差項がノイズに当てはまるリスク**を特に確認する．
- 真値を複数水準で設定 → 実刺激系列で人工データ生成 → MAP推定 → 真値と比較
- 指標：Pearson相関，MAE，バイアス（相関 ≥ 0.7 を実用基準）

### 2. モデル比較：相互作用の検定（本仕様書の主目的）

**比較対象は分離モデル（`directional_q_learning_map`）と本モデルのみ**（他は副次的）．両者は分離 ⊂ ジョイントの nested 関係で，差分は交差項＝方向間相互作用に対応する．

| 比較対象 | 構造 | 自由パラメータ | 潜在重み |
|---|---|---|---|
| 分離方向価値（OOZ） | 非ジョイント | 4 | $N\times2$ |
| 本モデル（連続ジョイント・OOZ） | ジョイント | 4 | $N^2\times2$ |

> **裁定指標は AIC/BIC ではなく交差検証（CV）**：潜在重みが自由パラメータ数に現れないため，AIC/BIC は本モデルの実効的柔軟さを過小評価する．代わりに **試行を分割した out-of-sample 予測対数尤度**で比較する．

**手順（参加者ごと・leave-trials-out CV）**
1. 各参加者の試行を時系列を保ったまま $K$ 分割（例：前半学習→後半予測，または blocked $K$-fold）
2. 学習 fold で MAP 推定，保留 fold で予測対数尤度（と選択正解率）を算出
3. 両モデルの $K$ 平均 CV 対数尤度を参加者ごとに比較
4. **更新則を両モデルで一致させる**（本仕様 (A) なら分離側も呈示状態更新，(B) なら分離側も報告方向更新）ことで，差を相互作用成分に限定する
5. 参加者単位のペア差（本モデル − 分離モデル）の符号と分布を集計（符号検定 / paired Wilcoxon）

**解釈**
- 本モデルが CV で有意に上回る → 白と黒の方向は**相互作用している**（片方の方向が他方の価値に影響）
- 互角／分離が上回る → **分離仮定で十分**．少なくともこのデータでは相互作用を支持しない
- 検出力に注意：48試行で $N^2$ の交差項は拘束が弱く，差が出ない場合は「相互作用なし」ではなく「**この試行数では検出できず**」が正しい結論

### 3. 事後予測チェック

推定パラメータと実刺激系列で多数回シミュレーションし，ターゲット選択率の試行推移を実データと比較（95%区間に実データが収まるか）．`predict_target_choice_probs` を本モデル用に実装し，`target_group` から $P(\text{target})$ を再構成する（分離モデルと同形式）．

---

## 前提と未解決の論点（要確認）

1. **更新則 (A)/(B) の選択**：報告方向の正確さを価値に取り込むか（B）／取り込まないか（A）．モデル比較では分離モデルと揃えることが必須．まず (A) で実装・比較し，必要に応じて (B) を追加検討する．
2. **$N$ の選択とデータ量**：$N=4$ で潜在16次元/行動．$N$ を上げると表現力は増すが，48試行では過学習が支配的になりうる．CV で $N$ も選ぶ余地がある（ただし固定基底の拡大は DQN の特徴学習とは別軸；`continuous_joint` は浅い固定基底のまま）．
3. **相互作用の心理的中身**：交差項が捉えるのは「相対角 $\theta_w-\theta_b$ が効く」「片方が他方を抑制する」等．有意なら，どの $(\mu_j,\mu_k)$ 成分が効いているかを可視化して解釈する．
4. **報酬パターンの決まり方**（分離モデルと共通の前提）：ターゲット色が運動方向と結びついているかに，方向ベース価値学習の妥当性が依存する．
