# 状態を観測から推定するQ学習モデル 仕様書（MAP推定）

本仕様書は，[`state_estimation_primer.md`](state_estimation_primer.md) の概念整理を受けて，**隠れ報酬パターンをオラクルとして渡さず，観測（各刺激の運動方向・画像）から状態を扱う**Q学習モデルを具体的な数式・疑似コードで定義する．

`q_learning_ooz_map.py` が抱える矛盾 ―― 状態 `target_item`（真の白高／黒高パターン）を正解として与えている ―― を，状態推定の枠組みで解消することを目的とする．

3つの道を提示する：

| ルート | 状態の扱い | 円環性の解決 | index 問題の解決 | 既存実装との関係 |
|---|---|---|---|---|
| **A. 観測を状態とする関数近似** | 観測角度 $\theta$ をそのまま状態入力に | 周期基底 $\phi(\theta)$ | 表を引かず $Q(\theta,a)=w_a^\top\phi(\theta)$ | **既に `directional_q_learning_map.py`** |
| **B. 潜在レジームの状態推定** | 隠れパターン $z$ をベイズフィルタで推定 | （角度は観測尤度側で処理） | index は離散 $z$，$Q(b,a)=\sum_z b(z)q(z,a)$ | 新規（`gaussian_hmm.py` の思想を行動モデルへ） |
| **C. 知覚的状態推定（補）** | ノイズあるドットから真方向を推定 | 円環カルマン／von Mises 測定モデル | ― | 拡張オプション |

本研究の中心的主張（暗黙学習）に最も忠実なのは **Route B**．Route A は既に実装済みで識別性も高い実務的第一歩．以下，各ルートの数式を与える．

---

## 記号と共通設定

各試行 $t$ で：

- $o_t = (\theta_{w,t}, \theta_{b,t})$：白・黒ドット群の運動方向（ラジアン，$[0,2\pi)$）．観測．
- $a_t \in \{\text{white}, \text{black}\}$：参加者が選んだ色（`chosen_color`）．
- $\theta_{\text{rep},t}$：報告方向（`response_angle_css`，ラジアン）．
- $r_t \in [0,1]$：スカラー報酬（`reward_points`/10．色の正誤と角度精度が畳み込み済み）．
- $z_t \in \{W, B\}$：**隠れ報酬パターン**（白高＝W／黒高＝B）．**モデルには渡さない**（＝推定対象）．
- `target_group` は事後分析でのみ使用（学習・選択には渡さない）．

前処理は既存モデルと共通（`rt`/`chosen_color` の NaN を drop，度→ラジアン，報酬を $[0,1]$ に）．

---

# Route A：観測角度を状態とする関数近似Q学習

## A.1 位置づけ

**これは既存の `directional_q_learning_map.py` がすでに実装している道**である．本節はそれを「状態推定」の言葉で読み直し，円環性と index 問題がどこで解けているかを明示する．新規実装は不要（既存を使えばよい）．

## A.2 状態表現：周期基底による符号化

生の角度を離散キーにする代わりに，$N$ 個の von Mises 基底で符号化する：

$$\mu_j = \frac{2\pi j}{N},\quad \phi_j(\theta)=\exp[\kappa\cos(\theta-\mu_j)],\quad \boldsymbol{\phi}(\theta)=[\phi_0,\ldots,\phi_{N-1}]^\top$$

$\phi(\theta+2\pi)=\phi(\theta)$ なので **円環性はここで解決**（primer §5-1）．

## A.3 価値・選択・更新

色ごとの重み $\mathbf{W}_c\in\mathbb{R}^N$ で価値を表現（**index は連続角度でなく重みベクトル**なので primer §5-2 の index 問題も解決）：

$$V_c(\theta) = \mathbf{W}_c\cdot\boldsymbol{\phi}(\theta)$$

$$dq = \beta\big[V_{\text{black}}(\theta_b) - V_{\text{white}}(\theta_w)\big] + c_{\text{black}},\qquad P(\text{black}) = \sigma(dq)$$

$$\delta_t = r_t - V_{a_t}(\theta_{\text{rep},t}),\qquad \mathbf{W}_{a_t} \leftarrow \mathbf{W}_{a_t} + \alpha_t\,\delta_t\,\boldsymbol{\phi}(\theta_{\text{rep},t})$$

学習率は OOZ ラベルで切り替え（$\alpha_t=\alpha_0$ if $z^{\text{ooz}}_t=0$ else $\alpha_1$）．パラメータ $(\alpha_0,\alpha_1,\beta,c_{\text{black}})$，事前分布・推定手続きは既存 spec と同一．

> **要点**：Route A では「状態推定」は明示的には行わない．観測をそのまま特徴に通して価値関数を学ぶだけである．暗黙学習の *矛盾（オラクル問題）* は解けているが，**「隠れレジームを推定する」という現象そのものはモデル化していない**．それを正面から扱うのが Route B．

---

# Route B：潜在レジームをベイズフィルタで推定するQ学習（本仕様の主眼）

## B.1 生成モデルの設定

環境は2つの潜在レジーム $z\in\{W,B\}$ を，ブロック境界で確率的に切り替えながら遷移する（reversal/volatile 構造）．エージェントは $z$ を **観測できず**，報酬フィードバックから推定する．

**状態遷移（ハザード率 $h$）**：

$$P(z_t=k\mid z_{t-1}=j) = (1-h)\,\mathbb{1}[k=j] + h\,\mathbb{1}[k\neq j]$$

$h$ は「レジームがどれだけ切り替わりやすいか」＝ボラティリティ（Behrens et al., 2007）．実験のブロック長が既知なら $h\approx 1/(\text{平均ブロック長})$ に固定してよい（推定パラメータを増やさない）．

**観測（報酬）尤度**：レジーム $k$ にいるとき，色 $a_t$ を選んだ場合に期待される報酬を $q(k,a_t)$ とし，観測報酬をその周りのノイズとみなす．$r_t\in[0,1]$ なのでベータ or トランケート正規で書けるが，実装簡便のため正規近似：

$$P(r_t\mid z_t=k, a_t) = \mathcal{N}\!\big(r_t;\ q(k,a_t),\ \sigma_r^2\big)$$

ここで $q(k,a)$ は **2×2 の小さな価値表**で，これも学習される（後述）．$\sigma_r$ は報酬観測ノイズ幅（固定 or 推定）．

> **なぜこれで潜在が識別できるか**：2つの潜在は「黒を選ぶと報酬が高い regime」と「白を選ぶと報酬が高い regime」として *自己組織化* する．どちらが今アクティブかは，直近の報酬系列が $q(W,\cdot)$ と $q(B,\cdot)$ のどちらの予測とよく合うかで決まる．これは **latent-cause 推論**（Gershman & Niv, 2010）そのものである．

## B.2 信念更新（forward ベイズフィルタ）

**予測（時間更新）**：

$$b_t^-(k) = \sum_{j\in\{W,B\}} P(z_t=k\mid z_{t-1}=j)\,b_{t-1}(j)$$

**補正（観測更新，行動と報酬を見た後）**：

$$b_t(k) = \frac{P(r_t\mid z_t=k,a_t)\,b_t^-(k)}{\sum_{k'} P(r_t\mid z_t=k',a_t)\,b_t^-(k')}$$

初期信念 $b_0 = (0.5, 0.5)$（無情報）．これは `gaussian_hmm.py` の forward 再帰と同型で，違いは **観測尤度が「行動で条件づけた報酬」である**点だけ．

## B.3 価値・選択

行動価値は **信念で周辺化** する（belief-MDP の $Q(b,a)$）：

$$Q(b_t, a) = \sum_{k\in\{W,B\}} b_t(k)\,q(k,a)$$

**選択則**（符号規約は既存モデルと統一，$c_{\text{black}}>0$ が黒選好）：

$$dq = \beta\big[Q(b_t,\text{black}) - Q(b_t,\text{white})\big] + c_{\text{black}},\qquad P(\text{black}\mid t)=\sigma(dq)$$

> **選択に使う信念は「補正前」$b_t^-$**（＝行動・報酬を見る前の予測信念）にする．そうしないと，まだ観測していない今試行の報酬で今試行の選択を説明してしまい因果が壊れる．すなわち：選択は $Q(b_t^-,a)$ で計算 → 報酬を観測 → $b_{t-1}\!\to b_t$ に補正 → 次試行へ．

## B.4 価値表の学習（信念で重みづけた TD）

$q(k,a)$ も固定でなく学習する．**責任（responsibility）$b_t(k)$ で重みづけた Rescorla-Wagner 更新**（Starkweather et al., 2017；Gershman & Niv, 2010）：

$$q(k, a_t) \leftarrow q(k, a_t) + \alpha_t\, b_t(k)\,\big(r_t - q(k, a_t)\big)\qquad \forall k$$

学習率は OOZ ラベルで切り替え（$\alpha_t\in\{\alpha_0,\alpha_1\}$），選んだ行動 $a_t$ の列のみ，全レジーム $k$ について責任比で更新する．**「今どのレジームにいると思うか」に比例して各レジームの価値を割り当てる**のが状態推定つき学習の核心である．

## B.5 角度（観測 $\theta$）はどこで効くか ― 2つの設計

報酬パターンが **運動方向と結びついているか** で扱いが変わる（directional spec の「未解決の論点」と同じ分岐）：

- **設計 B-i（パターンは色レジーム，方向は無関係）**：上式のまま．$\theta$ は選択・信念更新には入らず，報告方向の知覚成分は $r_t$ に畳み込まれているとみなす．最小構成でまず実装するならこれ．
- **設計 B-ii（パターンが方向で定義される）**：潜在を「高報酬 *方向* $\psi\in$ 円環」とし，レジームを離散2値でなく方向連続にする．観測尤度を von Mises に：

$$P(r_t\mid \psi, a_t, o_t) \propto \exp\!\big[\lambda\, r_t \cos(\theta_{a_t} - \psi)\big]$$

信念 $b_t(\psi)$ は円環上の分布（von Mises 混合 or グリッド近似）で更新する．これは Route C の測定モデルと連続する．まずは B-i で識別性を確認し，必要なら B-ii へ拡張するのが現実的．

## B.6 パラメータと事前分布

| パラメータ | 範囲 | 意味 | 事前分布 |
|---|---|---|---|
| $\alpha_0$ | $(0,1)$ | 非OOZ 学習率 | $\text{Beta}(2,2)$ |
| $\alpha_1$ | $(0,1)$ | OOZ 学習率 | $\text{Beta}(2,2)$ |
| $\beta$ | $(0,\infty)$ | 逆温度 | $\text{Gamma}(2,3)$ |
| $c_{\text{black}}$ | $(-\infty,\infty)$ | 色バイアス | $\text{Normal}(0,2)$ |

**固定推奨**（48試行での識別性のため）：ハザード率 $h$（ブロック構造から算出），報酬ノイズ $\sigma_r$（例 0.3）．これらを推定に含めると自由度が増え不安定になりやすい．まず固定 → 必要なら $h$ か $\sigma_r$ を1つだけ緩める．

> **自由パラメータ数は4で `q_learning_ooz_map` と同一**．したがって AIC/BIC の直接比較が公平に行える（潜在重みが Route A の von Mises 版より少なく，過学習リスクも低い）．

## B.7 対数事後と推定手続き

$$\log P(\boldsymbol{\theta}\mid\text{data}) \propto \sum_t \log P(a_t\mid b_t^-, q_t) + \sum\log P(\text{prior})$$

疑似コード（1参加者）：

```text
入力: choices a[t], rewards r[t], ooz z_ooz[t]  （states は使わない！）
固定: h, sigma_r, 事前分布ハイパー
パラメータ: alpha0, alpha1, beta, c_black

q = [[0.5, 0.5],      # q[W, white], q[W, black]
     [0.5, 0.5]]      # q[B, white], q[B, black]
b = [0.5, 0.5]        # 信念 b[W], b[B]
loglik = 0

for t in range(T):
    # --- 予測（時間更新） ---
    b_pred[W] = (1-h)*b[W] + h*b[B]
    b_pred[B] = (1-h)*b[B] + h*b[W]

    # --- 選択（補正前信念で） ---
    Q_white = b_pred[W]*q[W,white] + b_pred[B]*q[B,white]
    Q_black = b_pred[W]*q[W,black] + b_pred[B]*q[B,black]
    dq = clip(beta*(Q_black - Q_white) + c_black, -500, 500)
    p_black = sigmoid(dq)
    p_chosen = p_black if a[t]==black else 1-p_black
    loglik += log(p_chosen + 1e-300)

    # --- 補正（報酬観測後の信念更新） ---
    lik_W = normal_pdf(r[t], q[W, a[t]], sigma_r)
    lik_B = normal_pdf(r[t], q[B, a[t]], sigma_r)
    b[W] = lik_W*b_pred[W] / (lik_W*b_pred[W] + lik_B*b_pred[B])
    b[B] = 1 - b[W]

    # --- 価値表の学習（責任重みづけ TD） ---
    alpha = alpha0 if z_ooz[t]==0 else alpha1
    q[W, a[t]] += alpha * b[W] * (r[t] - q[W, a[t]])
    q[B, a[t]] += alpha * b[B] * (r[t] - q[B, a[t]])

return loglik + log_prior(alpha0, alpha1, beta, c_black)
```

MAP 推定は既存モデルと同じく `scipy.optimize.minimize`（L-BFGS-B）＋グリッド初期値で負の対数事後を最小化する．

## B.8 出力

| カラム | 内容 |
|---|---|
| `subject`, `alpha_0`, `alpha_1`, `beta`, `c_black` | 推定値（既存と同形式） |
| `log_posterior`, `log_likelihood`, `n_trials`, `n_ooz`, `n_non_ooz` | 既存と同形式 |
| `h`, `sigma_r` | 使用した固定値 |

## B.9 事後分析：信念軌跡と「暗黙学習」の定量化

Route B の最大の利点は，**信念軌跡 $b_t(W)$ を試行ごとに復元できる**こと．これで暗黙学習を直接可視化・定量化できる：

- 各試行の $b_t(\text{正解レジーム})$（`target_group` を正解として突き合わせ）を全参加者平均し，試行を通じて 0.5 → 1 に上がるか．
- **気づきの指標**：信念エントロピー $H(b_t)=-\sum_k b_t(k)\log b_t(k)$ の低下速度，あるいはブロック切替後に信念が反転するまでの試行数．
- ターゲット選択確率は Route A と同形式で $P(\text{target})$ を再構成し，実データの前後半差分と照合．

---

# Route C（補）：知覚的状態推定 ― ノイズあるドットから真方向を推定

「刺激の画像から状態を推定する」を最も素直に定式化する道．RDK ではコヒーレンスが低いほど運動方向の知覚が不確実になる．真方向 $\theta^\star$ を潜在，知覚 $\hat\theta$ を観測とした **円環上のベイズ推定（von Mises 測定モデル）**：

$$P(\hat\theta\mid\theta^\star) = \frac{1}{2\pi I_0(\kappa_{\text{obs}})}\exp[\kappa_{\text{obs}}\cos(\hat\theta-\theta^\star)],\qquad \kappa_{\text{obs}} = g\cdot(\text{coherence})$$

事前 $P(\theta^\star)$（学習した方向価値を事前に流用も可）と掛けて事後 $P(\theta^\star\mid\hat\theta)$ を得る．これは **円環版カルマンフィルタ**に相当し，Route A の価値計算に「知覚の不確かさ」を注入する拡張になる．

> **深層 encoder（画像→状態）が必要になるのはここだけ**：RDK 刺激は（方向, コヒーレンス, 色）で完全に記述できるため，生画像から CNN で学習しても復元されるのは結局この方向とコヒーレンスである．したがって **手作りの $\phi(\theta)$ ＋ コヒーレンス依存の $\kappa_{\text{obs}}$ が，学習 encoder と実質同等かつ識別性で優る**．画像を直接入れる意義は，コヒーレンス以外の未知の知覚手がかりを疑う場合に限られる．

---

## モデル評価（3ルート共通）

1. **パラメータリカバリ**：48試行・実刺激系列で人工データ生成 → MAP 推定 → 真値と比較（Pearson $\ge 0.7$）．Route B は $\alpha_1$（OOZ 試行が少ない参加者）と，$h$/$\sigma_r$ を緩めた場合の識別性に特に注意．
2. **モデル比較**：`q_learning_ooz_map`（オラクル）／Route A（directional）／Route B（状態推定）を **同一の自由パラメータ数4** で BIC・AIC・交差検証対数尤度で比較．
   - **Route B が Route A / オラクル版と同等以上に当てはまれば**，「隠れレジームを推定しながら暗黙に学習する」という記述が，矛盾なく（オラクルなしで）データを説明できたと言える ―― これが本研究の主張の定量的裏づけになる．
3. **事後予測チェック**：推定パラメータ＋実刺激系列で多数回シミュレーション → ターゲット選択率の試行推移が実データを95%区間に含むか．Route B は加えて信念軌跡の妥当性（切替後の反転タイミング）も確認．

---

## どのルートをいつ使うか（意思決定）

- **すぐ矛盾を消したいだけ** → Route A（既存 `directional_q_learning_map.py`）で十分．追加実装ゼロ．
- **「暗黙に隠れ構造を推定する」現象を正面からモデル化したい**（本研究の主張の核） → **Route B**．新規実装だが自由度4で識別性も良好．**推奨**．
- **知覚ノイズ（コヒーレンス）を明示的に扱いたい** → Route C を Route A/B に足す．

---

## 前提の確認事項（実装前に決める）

1. **報酬パターンは何で決まるか**：色レジーム（B-i）か方向依存（B-ii）か．これで Route B の観測尤度の形が変わる．directional spec の同名の論点と一致．
2. **ブロック構造とハザード率**：レジーム切替の頻度が既知なら $h$ を固定できる．未知なら推定 or ボラティリティ学習（Behrens 2007）に拡張．
3. **報酬ノイズ $\sigma_r$**：報酬分布の分散から経験的に設定．
4. **選択に使う信念のタイミング**：必ず補正前 $b_t^-$（§B.3 の因果注意）．

---

## 実装の入口

新規実装は `stats/state_estimation_q_learning_map.py` として，既存 `directional_q_learning_map.py` の骨格（`_prepare_subject` / `_run_trial_loop` / `_neg_log_posterior` / `fit_*` / `predict_*`）をほぼ流用できる．違いは `_run_trial_loop` の中身を §B.7 の疑似コードに差し替え，`states`（オラクル）への依存を除くこと，出力に信念軌跡 `belief_correct` を足すことのみ．必要なら実装を進める．
