# tabular OOZ Q学習 と 色別方向価値Q学習 の更新式の違い

`stats/q_learning_ooz_map.py`（tabular）と `stats/directional_q_learning_map.py`（色別方向価値）の
価値関数更新が、数学的・役割的にどう異なるかを整理する。

**結論：両者は数学的にも役割的にも同等の処理ではない。** フィット結果で α が大幅に異なるのは
当然の帰結であり、バグでも矛盾でもない。以下にその理由を更新式から示す。

---

## 1. 2つの更新式を並べる

### tabular OOZ（`q_learning_ooz_map`）

$$Q[s,a] \leftarrow Q[s,a] + \alpha\,\underbrace{(r - Q[s,a])}_{\delta}$$

状態 $s$（白高/黒高パターン）× 行動 $a$（白/黒）の **2×2 の表の1マス**だけを更新する。
これは特徴ベクトルが完全な one-hot（選んだマスだけ 1、他 0）の特殊ケースに相当する。

### 色別方向価値（`directional_q_learning_map`）

$$\mathbf{W}_{c} \leftarrow \mathbf{W}_{c} + \alpha\,\delta\,\boldsymbol{\phi}(\theta_{\text{reported}}),
\qquad \delta = r - V_c(\theta_{\text{reported}}),
\qquad V_c(\theta) = \mathbf{W}_c \cdot \boldsymbol{\phi}(\theta)$$

重み**ベクトル** $\mathbf{W}_c$ を、密な基底特徴 $\boldsymbol{\phi}(\theta)$（$\phi_j(\theta)=\exp[\kappa\cos(\theta-\mu_j)]$）の
方向へ動かす。

---

## 2. 核心：α が「価値そのものに効く率」になっていない

報告方向における価値変化を計算すると：

$$V_c^{\text{new}}(\theta_r)
= (\mathbf{W}_c + \alpha\delta\boldsymbol{\phi}(\theta_r))\cdot\boldsymbol{\phi}(\theta_r)
= V_c^{\text{old}}(\theta_r) + \alpha\delta\,\underbrace{\|\boldsymbol{\phi}(\theta_r)\|^2}_{\text{倍率}}$$

つまり報告方向における**実効学習率は $\alpha$ ではなく $\alpha\cdot\|\boldsymbol{\phi}(\theta_r)\|^2$** である。

- tabular：one-hot なので $\|\boldsymbol{\phi}\|^2 = 1$ → 実効率はちょうど $\alpha$
- 方向モデル（$N=4,\ \kappa=2$）：$\|\boldsymbol{\phi}(\theta)\|^2 \approx 34\text{–}57$（円環平均 **約 45**）

| $\theta$ | $\|\boldsymbol{\phi}(\theta)\|^2$ |
|---|---|
| 0° / 90°（基底中心） | 56.6 |
| 22° / 112° / 202° / 292° | 45.5 |
| 45° / 135°（基底の中間） | 34.0 |
| 円環全体 | min 34.0 / max 56.6 / mean 45.2 |

→ **同じ価値変化を生むには、方向モデルの $\alpha$ は tabular の約 1/45 でなければならない。**
「α が大幅に異なる」のはまさにこれで、両者の $\alpha$ は**そもそも別スケールの量**である。
実際のフィット値も方向モデルは 0.02–0.1 程度（tabular の典型 0.3–0.7 の約 1/40）になり整合する。

---

## 3. 役割的な違い（数学以外でも別物）

| | tabular OOZ | 色別方向価値 |
|---|---|---|
| 更新が及ぶ範囲 | **1マスのみ**（般化なし） | $\boldsymbol{\phi}(\theta_r)\!\cdot\!\boldsymbol{\phi}(\theta)$ に比例し**近い方向へ波及** |
| α の役割 | 更新速度のみ | 更新速度 ＋ **般化の強さ**が絡む（$\|\phi\|^2$ 経由） |
| δ の基準点 | $Q[s,a]$＝更新する当のマス | $V_c(\theta_{\text{reported}})$。一方**選択の評価は** $V_c(\theta_{\text{stimulus}})$＝別の点 |
| 初期値 | $Q=0.5$ → 1試行目 $\delta = r-0.5$ | $\mathbf{W}=0,\ V=0$ → 1試行目 $\delta = r$（系統的に大きい） |
| 状態の分割 | 色価値を**パターン別に2分割**して別々に保持 | パターンは存在せず、$V_c(\theta)$ は実験全体で**1本の連続関数** |

特に重要な2点：

- **δ 系列が違う。** 初期値（0.5 vs 0）と状態分割（パターン別2行 vs 方向の連続関数）が違うため、
  各モデルが経験する報酬予測誤差の系列そのものが別物。α は「その系列に最もよく合う値」なので、
  当然違う値に落ちる。
- **β との結合が違う。** tabular の $Q$ は概ね $[0,1]$ に収まるが、方向モデルの
  $V_c(\theta)=\mathbf{W}\cdot\boldsymbol{\phi}$ は基底値が大きいぶん別スケール。
  $dq=\beta(V_{\text{black}}-V_{\text{white}})$ なので $\beta$ がその差を吸収し、
  $\alpha$ と $\beta$ のトレードオフ点も移動する。→ **β も両モデルで直接比較できない。**

---

## 4. では何が言えるか

- **当初の問題（パターンを状態として渡す＝気づき前提の矛盾）は解決できている。**
  方向モデルは学習・選択でパターンラベルを一切使っていない。
- ただし「矛盾を解消した」ことと「tabular と数学的に等価」は別問題で、
  **方向モデルは設計上まったく別のダイナミクス**である。だから α が違うのは想定どおりで、
  **2モデルの生の α・β を直接比べてはいけない。**
- 比較すべきは**モデル適合度**（BIC / AIC / WAIC、pseudo-$R^2$、choice accuracy）であって、
  生パラメータではない。

---

## 5. 補足：$\log(\alpha_1/\alpha_0)$ という検定統計量は頑健

OOZ 効果の検定に使う $\log(\alpha_1/\alpha_0)$ は、上の $\|\boldsymbol{\phi}\|^2$ 倍率がおおむね**比で相殺**される
（倍率は OOZ/非OOZ で共通の $\theta$ 分布に依存し、$\alpha$ の絶対スケールではなく比に効く）：

$$\log\frac{\alpha_1}{\alpha_0}
= \log\frac{\alpha_1\|\phi\|^2}{\alpha_0\|\phi\|^2}\quad(\text{実効率の比})$$

したがって、**生の α が別スケールでも「OOZ のとき学習が速い／遅い」という
within-model のコントラストは両モデルで意味が保たれる。**

ただし完全な相殺ではない点に注意：

- (a) $\theta$ と OOZ が相関すると倍率がずれる
- (b) $\delta$ が履歴依存で非線形なので、これは一次近似での議論

可能なら「$\theta$ 分布が OOZ/非OOZ で偏っていないか」を一度確認しておくと安全。

---

## まとめ

同じ「Q学習（TD ＋ softmax）」という枠ではあるが、**特徴表現・初期値・状態分割・実効学習率スケールが
すべて異なる別モデル**である。α の絶対値の不一致はその必然的な帰結であり、

- モデルの妥当性は **適合度** で評価する
- OOZ 効果は **$\log(\alpha_1/\alpha_0)$** で評価する（生 α の直接比較は不可）

という整理になる。
