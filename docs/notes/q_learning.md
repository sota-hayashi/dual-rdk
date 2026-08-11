# 連続値（方向角）を状態とするQ学習 ― なぜ関数近似であって状態推定ではないのか

本ノートは次の問いに答える．

> レジーム（環境の状態）を白・黒ドット群の**方向角度**として Q 学習モデルに組み込みたい．しかし状態が 0/1 のワンホットではなく連続値になるため，一般的な表計算の Q 学習では記述できない．こういう連続入力を扱える Q 学習は **state estimation の分野**にあるのではないか？

**結論：ある．ただし探していた場所とは違う棚に．**

実務上の主力は **function approximation（関数近似）**であり，`directional_q_learning_map.py` はその標準解そのものである（2〜4節）．一方で，**価値関数の重み $\mathbf{W}$ 自体を隠れ状態とみなしてフィルタで追跡する** state estimation 版も実在する ―― **Kalman TD** である．そして **Rescorla–Wagner / TD 更新は，その Kalman TD の点推定極限**にあたる（5節）．

つまり関数近似と状態推定は二者択一ではなく，**同じものの両端**である．「連続入力を状態推定の枠組みで書けるはずだ」という直感は正しい．ただし追うべき隠れ状態は**離散レジームではなく連続な重み $\mathbf{W}$** であり，ここを取り違えると Route B のように退化する（4.1節）．

関連ノート：ベイズフィルタの概念は [`bayes_filter_primer.md`](bayes_filter_primer.md)，隠れ状態推論の一次文献は [`key_papers_state_estimation.md`](key_papers_state_estimation.md)．

---

## 1. 2つの問題は直交している ―― ただし「何が隠れているか」が鍵

「連続」と「隠れている」は**別々の軸**である．混同すると，永遠に間違った棚を探すことになる．

|  | **状態が観測できる** | **状態が隠れている** |
|---|---|---|
| **状態が離散** | 表計算 Q 学習 | ベイズフィルタ / POMDP / HMM |
| **状態が連続** | **関数近似** ← 実務上の主力 | Kalman フィルタ / 粒子フィルタ |

それぞれの分野が答えている問いは違う．

- **state estimation（状態推定）**が答える問い：「状態が**見えない**とき，観測からどう推し量るか」
- **function approximation（関数近似）**が答える問い：「状態が**見えている**が連続値のとき，価値関数をどう表現するか」

### 1.1 何が観測できて，何が隠れているのか

本研究について，この表のどこに入るかを決めるには「**何を状態と呼ぶか**」を先に確定させる必要がある．候補は3つあり，それぞれ答えが違う．

| 「状態」の候補 | 観測できるか | 時変か | 適切な道具 |
|---|---|---|---|
| **方向角 $\theta$** | **できる**（coherence $\approx 1.0$） | 試行ごとに変わる | 推定不要．そのまま入力に使う |
| **報酬レジーム**（白高/黒高） | **できる**（$\theta$ から一意に決まる） | 試行ごとにランダム | 推定不要．→ Route B が退化した理由（4.1節） |
| **方向 $\to$ 報酬の写像 $\mathbf{W}$** | **できない** ← **これが隠れている** | ほぼ一定 | 関数近似（2節）または Kalman TD（5節） |

つまり **$\theta$ もレジームも隠れていない．隠れているのは写像 $\mathbf{W}$ だけ**である．「連続入力を state estimation で扱いたい」という発想が正しく実を結ぶのは，$\theta$ ではなく **$\mathbf{W}$ を隠れ状態に据えたとき**である（＝ Kalman TD，5節）．

まず 2〜4 節で関数近似（$\mathbf{W}$ を点推定で学ぶ）を，5 節でその状態推定版（$\mathbf{W}$ を分布で追う）を扱う．

キーワードは *function approximation* / *basis function* / *radial basis function (RBF)* / *tile coding*．教科書は Sutton & Barto (2018) の 9〜10 章（On-policy Prediction with Approximation，coarse coding・RBF・tile coding）．

---

## 2. 「表計算では記述できない」は逆で，表計算のほうが特殊ケース

ここが腑に落ちる鍵である．

### 2.1 表引きは，実はワンホットとの内積

表計算 Q 学習の「表を引く」という操作は，数式で書くと**ワンホットベクトルとの内積**にほかならない．

$$
Q(s,a) \;=\; w_a[s] \;=\; \mathbf{w}_a \cdot \mathbf{e}_s
$$

ここで $\mathbf{e}_s$ は状態 $s$ の位置だけが 1 の**ワンホットベクトル**：

$$
\mathbf{e}_s = (0, \dots, 0, \underset{s\text{番目}}{1}, 0, \dots, 0)^{\top}
$$

更新式も同じことが言える．

$$
w_a[s] \;\leftarrow\; w_a[s] + \alpha\,\delta
\qquad\Longleftrightarrow\qquad
\mathbf{w}_a \;\leftarrow\; \mathbf{w}_a + \alpha\,\delta\,\mathbf{e}_s
$$

$\mathbf{e}_s$ は $s$ 番目以外がゼロなので，ベクトルで足しても結局 $s$ 番目のマスしか動かない．**つまり表計算 Q 学習は，すでに「重みベクトル $\times$ 特徴ベクトル」の形をしている．** 特徴ベクトルがたまたまワンホットなだけである．

### 2.2 連続値化＝ワンホットを「ぼやけたワンホット」に差し替えるだけ

やることはたった一つ．$\mathbf{e}_s$ を，$\theta$ を中心に**なだらかに盛り上がる特徴ベクトル** $\boldsymbol{\phi}(\theta)$ に置き換える．

$$
\boldsymbol{\phi}(\theta) \;=\; \big[\,\phi_1(\theta),\ \phi_2(\theta),\ \dots,\ \phi_N(\theta)\,\big]^{\top},
\qquad
\phi_j(\theta) = \exp\!\big[\kappa \cos(\theta - \mu_j)\big]
$$

$\mu_j = 2\pi j / N$ は円環上に等間隔に並べた基底の中心である．価値関数はこの内積で表す：

$$
V_c(\theta) \;=\; \mathbf{W}_c \cdot \boldsymbol{\phi}(\theta) \;=\; \sum_{j=1}^{N} W_{c,j}\,\phi_j(\theta)
$$

更新式は：

$$
\mathbf{W}_c \;\leftarrow\; \mathbf{W}_c + \alpha\,\delta\,\boldsymbol{\phi}(\theta),
\qquad \delta = r - V_c(\theta)
$$

**2.1 の式と見比べてほしい．形が完全に同じである．** $\mathbf{e}_s$ が $\boldsymbol{\phi}(\theta)$ になっただけで，Rescorla–Wagner / TD の骨格は一文字も変わっていない．

### 2.3 表計算は「粗い binning」の極限

$\kappa$ を大きくすると $\phi_j(\theta)$ は鋭く尖り，$\theta$ に最も近い基底だけが実質的に効くようになる．すなわち**重なりのないハード分割（最近傍ビン）**に近づく．これが表計算 Q 学習にあたる．

> **したがって「連続値だから表計算 Q 学習では記述できない」ではない．正しくは「表計算 Q 学習は，重なりのない粗い binning という特殊ケース」である．**

一般化した側が新しい道具なのではなく，**表のほうが制限版**だった，という関係になっている．

**ただし「$\kappa\to\infty$ で表計算を厳密に復元する」は言い過ぎなので，正確に区別しておく：**

- 状態が**離散**で，その位置が基底中心 $\mu_j$ に一致している場合 → $\boldsymbol{\phi}$ は本物のワンホットになり，2.1 の表計算に**厳密に一致する**．
- $\theta$ が**真に連続**の場合 → そもそも連続空間には無限個の状態があるので「表」は原理的に作れず，必ず何らかの**集約（binning）**になる．$\kappa\to\infty$ が与えるのは Voronoi 的なハード分割であって，ワンホット表そのものではない．
- 加えて実装は $\boldsymbol{\phi}$ を**正規化していない**（`np.exp(kappa * np.cos(theta - mus))`）ため，$\kappa$ を上げると基底のピーク値 $e^{\kappa}$ 自体が発散する．極限を実際に取る操作としては筋が悪い．

要するに $\kappa=\infty$ は**厳密な等価性ではなく直観の橋渡し**として読むのが正しい．骨格（表は連続版の制限ケース）は保たれるが，「厳密に一致」ではない．なお実装の既定値は $N=4$, $\kappa=2.0$ で，基底は**広く重なっている**（極限とは正反対の領域で動かしている）．

### 2.4 連続にして得られるもの＝汎化

では $\kappa$ を有限にして何が嬉しいのか．**汎化（generalization）**である．

- **表（$\kappa=\infty$）**：$10°$ で報酬を得ても，動くのは $10°$ のマスだけ．$15°$ の価値はゼロのまま．
- **連続（$\kappa$ 有限）**：$\boldsymbol{\phi}(10°)$ は $15°$ の基底にも値を持つので，$15°$ の価値も一緒に上がる．

参加者は「近い方向は似た報酬をもたらすはずだ」と当然のように汎化する．そして本実験は **48 試行しかない**．表で全方向を独立に埋めるのは不可能で，汎化が入って初めて学習が成立する．ここは決定的である．

### 2.5 なぜ von Mises なのか

理由は単純で，**方向空間が円環（$0° = 360°$）だから**である．ガウス RBF を使うと巻き戻り地点（$359° \to 1°$）で「遠い」と誤判定して破綻する．von Mises 分布 $\exp[\kappa\cos(\theta-\mu)]$ は**円環上のガウス**にあたり，$\kappa$ が幅（大きいほど鋭い）を表す．

つまり von Mises は深い理論的コミットメントではなく，「空間が円だから円用のガウスを使う」というだけの選択である．実装（`von_mises_value_learning_map.py`）の既定値は $N = 4$，$\kappa = 2.0$ の固定．

---

## 3. 「2次元ベクトル」問題は，この実験では実質1次元

本課題では毎試行 $(\theta_{white},\, \theta_{black})$ の **2 つの角度**が呈示される．これを真面目に 2 次元状態として扱うなら，基底も 2 次元（2 つの基底の外積）にする必要がある：

$$
\boldsymbol{\phi}(\theta_{white}, \theta_{black}) = \boldsymbol{\phi}(\theta_{white}) \otimes \boldsymbol{\phi}(\theta_{black})
\quad\Rightarrow\quad N^2 \text{ 個の重み}
$$

48 試行で $N^2$ 個の重みを推定するのは絶望的である．**しかし実データを確認したところ，その心配は不要だった：**

```
θ_black − θ_white (mod 360°),  n = 480 試行（10名）
  min 227.0   max 270.0   mean 256.3   std 10.2
  pct 5/50/95: 236.0  259.0  269.0
```

$\theta_{black}$ は $\theta_{white}$ から **ほぼ決まる**（$43°$ 幅，すなわち $\pm 21°$ の範囲）．これは `experiment/index.html` の設計上の帰結である（`W_H`=[0,45]° は必ず `B_L`=[270,315]° と，`W_L`=[180,225]° は必ず `B_H`=[90,135]° と対にされる）．

> **2 次元に見えて，実際は 1 次元＋わずかなゆらぎ．**

したがって `directional_q_learning_map.py` のように色ごとに独立な価値関数を持ち，それぞれ自分の角度で評価する分解 ―― 

$$
\Delta q = \beta\big(V_{black}(\theta_{black}) - V_{white}(\theta_{white})\big) + c_{black},
\qquad P(\text{black}) = \sigma(\Delta q)
$$

―― は，情報をほとんど失わない．**この分解は手抜きではなく，データの構造に照らして正当である．**

---

## 4. やりたいことは，実は既にできている

> やりたいことは，レジームあるいは環境の状態を白と黒のドット群の方向角度として Q 学習モデルに組み込みたい

これは $V_c(\theta) = \mathbf{W}_c \cdot \boldsymbol{\phi}(\theta)$ そのものである．そして重要なのは：

> **隠れた報酬構造は，信念 $b$ ではなく重み $\mathbf{W}_c$ として学習される．**

### 4.1 この課題で本当に latent なのは「状態」ではなく「写像」

- 「今どの状態にいるか」（試行ごとに切り替わるもの）→ **実は隠れていない．** 方向を見れば一意に決まる．
- 「方向 $\to$ 報酬」という**写像そのもの** → **これが隠れている．** そしてこれはセッションを通じて**一定**である．

したがって推定の対象は，レジーム上の信念 $b_t$ ではなく重み $\mathbf{W}$ である．点推定で学べば `directional_q_learning`（2〜3節），分布として追えば Kalman TD（5.A節）になる ―― **どちらも対象は $\mathbf{W}$ で共通**であり，違いは不確実性を持つか捨てるかだけである．

**Route B が不活性化した理由（正確に）．** 「一定の写像にフィルタをかけても無意味だから」ではない ―― 一定値を雑音観測から推定するのはフィルタの得意技であり，実際 5.A の Kalman TD はそれをやる．真の敗因は2つである：

1. **レジームは隠れていない．** $\theta$ を見れば白高/黒高は一意に決まる．推定する必要のないものを推定対象に据えた．
2. **その $\theta$ をフィルタに与えなかった．** 報酬だけを尤度に使ったが，報酬はデザイン上 色に関して対称（24/24）なので $\text{lik}_W = \text{lik}_B$ が保たれ，信念は対称固定点 $b=(0.5,0.5)$ から動けない（実測で `b_post_W` の std $= 0$）．

つまり **state estimation という道具が悪かったのではなく，隠れ状態の選び方を間違えた**．同じフィルタを正しい対象（$\mathbf{W}$）に向ければ機能する（5.A.2 の対比表）．詳細は memory `route-b-latent-filter-degenerate` を参照．

### 4.2 $\mathbf{W}_c$ の収束過程が「暗黙的学習」の定量化になる

参加者が報酬帯を言語化できなくても，$\mathbf{W}_{white}$ が $[0,45]°$ 付近で盛り上がっていけば，**それが暗黙的にレジームを学んだ証拠**になる．気づき（awareness）の有無で $\mathbf{W}$ の立ち上がりの速さ・鋭さを比較する，という分析が自然に書ける．

Wilson et al. (2014) の「OFC ＝課題状態空間の認知地図」も，まさにこの $\mathbf{W}$（方向空間上に張られた価値の地図）に対応づけられる．

---

## 5. state estimation が正当に戻ってくる場所

2つある．**（A）が本命**で，冒頭の問い「連続入力を state estimation で扱えないか」への直接の答えである．

### 5.A 【本命】Kalman TD ― 重み $\mathbf{W}$ を隠れ状態としてフィルタで追う

1.1 節で確定した通り，この課題で隠れているのは **写像 $\mathbf{W}$** である．ならば **$\mathbf{W}$ そのものを隠れ状態に据えて，ベイズフィルタで追跡すればよい．** これが **Kalman TD** である．

**状態空間モデルとしての定式化：**

$$
\underbrace{\mathbf{w}_t = \mathbf{w}_{t-1} + \boldsymbol{\epsilon}_t}_{\text{状態遷移：緩やかなドリフト}},
\qquad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0},\, \tau^2 I)
$$

$$
\underbrace{r_t = \boldsymbol{\phi}(\theta_t)\cdot\mathbf{w}_t + \eta_t}_{\text{観測モデル：報酬が観測}},
\qquad \eta_t \sim \mathcal{N}(0,\, \sigma^2)
$$

隠れ状態 $\mathbf{w}_t$ は連続ベクトル，観測は報酬 $r_t$，そして**入力 $\theta_t$ は $\boldsymbol{\phi}(\theta_t)$ を通じて観測モデルに入る**．線形ガウスなのでカルマンフィルタが厳密に適用でき，事後分布 $p(\mathbf{w}_t \mid r_{1:t}, \theta_{1:t}) = \mathcal{N}(\hat{\mathbf{w}}_t, \Sigma_t)$ が閉形式で得られる：

$$
\underbrace{\hat{\mathbf{w}}_t}_{\text{事後平均＝価値の点推定}}, \qquad \underbrace{\Sigma_t}_{\text{事後共分散＝不確実性}}
$$

**更新式（補正ステップ）：**

$$
\hat{\mathbf{w}}_t \;\leftarrow\; \hat{\mathbf{w}}_t + \underbrace{\mathbf{k}_t}_{\text{カルマンゲイン}}\,\underbrace{\big(r_t - \boldsymbol{\phi}(\theta_t)\cdot\hat{\mathbf{w}}_t\big)}_{\delta_t\ =\ \text{報酬予測誤差}},
\qquad
\mathbf{k}_t = \frac{\Sigma_t\,\boldsymbol{\phi}(\theta_t)}{\boldsymbol{\phi}(\theta_t)^{\top}\Sigma_t\,\boldsymbol{\phi}(\theta_t) + \sigma^2}
$$

### 5.A.1 RW / TD は Kalman TD の点推定極限

上のカルマンゲイン $\mathbf{k}_t$ と，2.2 節の RW 更新を見比べてほしい：

| | 更新式 | ゲイン |
|---|---|---|
| **Kalman TD** | $\hat{\mathbf{w}} \leftarrow \hat{\mathbf{w}} + \mathbf{k}_t\,\delta_t$ | $\mathbf{k}_t$：**共分散 $\Sigma_t$ から毎試行計算される適応的ゲイン** |
| **RW / TD**（＝`directional_q_learning`） | $\mathbf{W}_c \leftarrow \mathbf{W}_c + \alpha\,\delta_t\,\boldsymbol{\phi}(\theta)$ | $\alpha\,\boldsymbol{\phi}(\theta)$：**固定スカラー $\times$ 基底** |

> **RW / TD は，Kalman TD から共分散 $\Sigma_t$ を捨て，ゲインを固定スカラー $\alpha$ に置き換えた特殊ケース**である（Dayan & Kakade 2000; Gershman 2015）．

2.3 節で「表計算は関数近似の制限版」と述べたのと**同じ構図がもう一段上でも成り立っている**：

$$
\text{表計算 Q 学習} \;\subset\; \underbrace{\text{関数近似 RW/TD}}_{\text{`directional\_q\_learning`}} \;\subset\; \underbrace{\text{Kalman TD}}_{\text{state estimation}}
$$

**したがって「関数近似 か 状態推定 か」は二者択一ではない．** 連続入力（$\boldsymbol{\phi}$ による表現）と状態推定（$\mathbf{W}$ 上のフィルタ）は同時に成立し，Kalman TD がその合流点である．「連続入力を state estimation の枠組みで書けるはずだ」という直感は**正しかった**．

### 5.A.2 Route B との決定的な違い ―― 何を隠れ状態に据えるか

同じ「state estimation」でも，**追う対象が違えば成否が分かれる**．

| | **Route B**（退化した） | **Kalman TD**（健全） |
|---|---|---|
| 隠れ状態 | 離散レジーム $z_t \in \{W, B\}$ | 連続な重み $\mathbf{w}_t$ |
| 時変性 | 毎試行ランダムに切替（ハザード $h \approx 0.5$） | ほぼ一定（緩やかにドリフト） |
| 問題の性質 | 高速に暴れる標的の追跡 | **一定値を雑音観測から推定** |
| 観測からの識別 | $\theta$ から一意に決まる＝**推定不要** | 報酬からしか分からない＝**推定必要** |
| 帰結 | 対称固定点で信念が凍結（`b` の std $= 0$） | 通常のフィルタとして機能 |

Route B が壊れたのは「state estimation がこの課題に不適合だから」ではなく，**隠れていないもの（レジーム）を隠れ状態に据えてしまったから**である．同じ道具を正しい対象（$\mathbf{W}$）に向ければ機能する．詳細は memory `route-b-latent-filter-degenerate`．

### 5.A.3 実務上の但し書き（48試行の現実）

概念的な正しさとは別に，本実験でフルの Kalman TD を回す実益は限定的である：

- **雑音パラメータの識別が困難**：ドリフト $\tau^2$ と観測雑音 $\sigma^2$ の比が学習率を決めるが，48 試行でこの2つを分離推定するのは苦しい．事実上 $\alpha$ 1個を推定するのと変わらなくなる恐れがある．
- **写像がほぼ一定**なので，ドリフト $\tau^2 \to 0$ が真値に近い．そのとき Kalman TD はゲインが単調減衰する再帰最小二乗に漸近し，**固定 $\alpha$ の RW との差が小さくなる**．
- 逆に **$\Sigma_t$ を活かせる場面**（不確実性駆動の探索，学習初期の大きなゲイン，OOZ 試行での不確実性上昇）を仮説として立てるなら，Kalman TD でしか書けない予測が作れる．

> **落としどころ：** 実際にフィットする主力は点推定版 `directional_q_learning`（4パラメータ）．Kalman TD は **(a) あなたの「状態推定で書けるはず」という直感を理論的に正当化する枠組み**として，かつ **(b) 不確実性に関する仮説を持つ場合の拡張**として位置づけるのが誠実である．論文では「本モデルの RW 更新は Kalman TD の点推定極限にあたる（Gershman 2015）」と一文書けば，関数近似と状態推定の関係を正しく示せる．

### 5.B 【付随】coherence を下げて $\theta$ を曖昧にする場合（Route C）

もう一つは，**coherence を下げて $\theta$ の知覚を曖昧にした場合**である．

そのとき初めて $\theta$ **も**隠れ状態になり，観測ドット $o$ から $p(\theta \mid o)$ を推定する必要が生まれる．価値は信念で周辺化した期待値になる：

$$
\bar{V}_c \;=\; \int p(\theta \mid o)\; V_c(\theta) \; d\theta
$$

これは**知覚レベルの state estimation** である．ただし現行実験は coherence $= 1.0$ なので，その余地はない．知覚的不確実性を操作する実験を新たに組む場合にのみ意味を持つ．

なお 5.A と 5.B は**直交する**：5.A は「報酬構造の不確実性」，5.B は「知覚の不確実性」を扱う．両方を入れれば $\mathbf{w}$ と $\theta$ の二重の不確実性を持つモデルになるが，本データでは 5.B は定数に潰れる．

---

## 6. まとめ

| 問い | 答え |
|---|---|
| 連続入力を扱う Q 学習は state estimation にある？ | **ある．Kalman TD．** ただし隠れ状態に据えるのは $\theta$ でもレジームでもなく**重み $\mathbf{W}$** |
| 関数近似と状態推定は別物？ | **違う．連続体．** 表計算 $\subset$ 関数近似 RW/TD $\subset$ Kalman TD |
| RW / TD と Kalman TD の関係は？ | **RW は Kalman TD の点推定極限**（共分散 $\Sigma_t$ を捨て，ゲインを固定 $\alpha$ にした版） |
| 表計算では記述できない？ | **逆．** 表計算は「重なりのない粗い binning」という制限版 |
| 連続化とは何をすること？ | ワンホット $\mathbf{e}_s$ を，ぼやけた $\boldsymbol{\phi}(\theta)$ に差し替えるだけ．更新式の形は不変 |
| von Mises を使う理由は？ | 方向空間が円環だから．円環上のガウス．それだけ |
| 2次元 $(\theta_w, \theta_b)$ はどうする？ | 実データ上ほぼ 1 次元（差の std = 10.2°）．色ごとの分解で十分 |
| 隠れた報酬構造はどこにある？ | 信念 $b$（レジーム上）ではなく **重み $\mathbf{W}_c$**．その収束過程＝暗黙的学習 |
| なぜ Route B は壊れた？ | state estimation が悪いのではなく，**隠れていないもの（レジーム）を隠れ状態に据えた**から |
| 結局どれをフィットする？ | 主力は `directional_q_learning`（4パラメータ）．Kalman TD は理論的位置づけ＋不確実性仮説がある場合の拡張 |

**要するに `directional_q_learning_map.py` は「解決策のつもり」ではなく，この問題に対する標準解そのものである．** しっくり来なかったとすれば，それは von Mises 基底という見慣れない道具に見えて，実体が「表引きをぼかしただけの Q 学習」だと伝わっていなかったからだと思われる．

そして「連続入力を状態推定の枠組みで扱えるはずだ」という直感は**正しい**．その答えが Kalman TD であり，`directional_q_learning` はすでにその**点推定版**として，同じ家系の中にいる．

---

## 7. 参考文献

**関数近似（2〜4節）**

- **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. ―― 9章（On-policy Prediction with Approximation：線形関数近似，coarse coding，RBF，tile coding），10章（制御への拡張）．2節の主張の教科書的裏付け．

**Kalman TD ／ 状態推定としての価値学習（5.A節）**

- **Dayan, P., & Kakade, S. (2000).** Explaining away in weight space. *Advances in Neural Information Processing Systems (NIPS)*, 13. ―― 連合学習の重みをカルマンフィルタで追う定式化．RW を点推定極限として位置づける原典のひとつ．
- **Gershman, S. J. (2015).** A unifying probabilistic view of associative learning. *PLoS Computational Biology*, 11(11), e1004567. ―― RW・カルマンフィルタ・潜在原因推論を一つの確率的枠組みに統合．「RW は Kalman TD の特殊ケース」を引くならここ．
- **Geist, M., & Pietquin, O. (2010).** Kalman Temporal Differences. *Journal of Artificial Intelligence Research*, 39, 483–532. ―― Kalman TD の定式化を扱う機械学習側の一次文献．

**状態表現の神経基盤（4.2節）**

- **Wilson, R. C., Takahashi, Y. K., Schoenbaum, G., & Niv, Y. (2014).** Orbitofrontal cortex as a cognitive map of task space. *Neuron*, 81(2), 267–279. ―― 状態表現としての $\mathbf{W}$ の神経基盤（要約は [`key_papers_state_estimation.md`](key_papers_state_estimation.md)）．

## 8. 関連実装

| ファイル | 役割 |
|---|---|
| `src/dualrdk/models/von_mises_value_learning_map.py` | 基底ユーティリティ（`_make_mus`, `_basis_vec`, `_value`）＋単一価値関数版 |
| `src/dualrdk/models/directional_q_learning_map.py` | **本命．** 色別 $\mathbf{W}_{white}, \mathbf{W}_{black}$ ＋ OOZ 依存学習率．パラメータ 4 個（`alpha_0`, `alpha_1`, `beta`, `c_black`），基底 $N=4$・$\kappa=2.0$ は固定 |
| `src/dualrdk/models/q_learning_ooz_map.py` | オラクル版（状態を正解として与える）．比較対象 |
| `src/dualrdk/models/state_estimation_q_learning_map.py` | Route B（潜在レジーム＋ベイズフィルタ）．本デザインでは不活性化することが判明済み |
