# 状態推定（State Estimation）入門資料 ― 本研究の暗黙学習モデルのために

本資料は，強化学習（RL）における **状態推定（state estimation）** の概要を，本研究（dual-RDK 課題での暗黙的な報酬構造学習）に接続する形で整理したものである．目的は次の2つ：

1. 「状態推定」という概念が RL・ロボティクス・神経科学でそれぞれ何を指すかを整理する
2. 現行の「状態ありQ学習」がなぜ暗黙学習の前提と矛盾するのかを，状態推定の言葉で言語化する

具体的な数式・実装は姉妹資料 [`state_estimation_q_learning_map_spec.md`](state_estimation_q_learning_map_spec.md) に置く．本資料はその前提となる概念整理に徹する．

---

## 0. 一行で言うと

> **現行モデルの問題は「観測（observation）」と「状態（state）」を混同していること**である．エージェントには本来 *観測できないはずの隠れ報酬パターン*（白高／黒高）を状態として直接与えてしまっている．暗黙学習を正しく記述するには，エージェントが *観測できるもの*（各刺激の運動方向や画像）だけから，行動価値の計算に必要な状態を **推定** しなければならない．この「観測から状態を復元する処理」こそが state estimation である．

---

## 1. MDP における「状態」とは何か ― マルコフ性

強化学習の標準的枠組みであるマルコフ決定過程（MDP）では，状態 $s_t$ は次の **マルコフ性** を満たす量と定義される：

$$P(s_{t+1}, r_t \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1}, r_t \mid s_t, a_t)$$

すなわち **「$s_t$ さえ分かれば，過去の履歴は将来の予測に何も足さない」** という十分統計量である．Q学習 $Q(s,a)$ の理論的保証（収束・最適性）は，この $s$ が手元にあることを前提にしている．

問題は，**現実の課題では $s$ を直接観測できないことがほとんど**という点にある．ロボットはカメラ画素やセンサ値 $o_t$ しか得られず，「自分の位置・速度」という状態はそこから **推定** するしかない．本実験の参加者も同様に，画面上のドット運動（観測）しか得られず，「今どちらの色が高報酬か」という報酬構造（状態）は直接には見えない．

---

## 2. 観測 vs 状態，そして POMDP

用語を厳密に分ける：

| 記号 | 名称 | 本実験での対応 |
|---|---|---|
| $s_t$ | 状態（state）：マルコフ性をもつ意思決定の十分統計量 | **隠れ報酬パターン** $z_t\in\{$白高, 黒高$\}$（＝どの色がターゲットか） |
| $o_t$ | 観測（observation）：センサから得られる生の情報 | 各試行の刺激：白の運動方向 $\theta_w$，黒の運動方向 $\theta_b$（あるいは生の画像） |
| $b_t$ | 信念（belief）：観測履歴から計算した状態の確率分布 | $P(z_t \mid o_{1:t}, a_{1:t}, r_{1:t})$ |

観測が状態を一意に決めないとき，問題は **MDP ではなく POMDP（部分観測マルコフ決定過程）** になる．POMDP では，行動を決めるのに十分な統計量は「今の観測 $o_t$」ではなく「観測履歴 $o_{1:t}$ 全体」，あるいはそれを圧縮した **信念状態 $b_t$** である（Kaelbling, Littman & Cassandra, 1998）．

> **本実験は本質的に POMDP である．** 報酬構造（$z$）はブロックごとに切り替わる隠れ変数で，1試行の刺激を見ただけでは分からない．過去の報酬フィードバックを積み重ねて初めて「今は黒が得らしい」と推定できる．これはまさに信念更新の問題であり，「暗黙的学習」の実体はこの **信念状態の緩やかな更新** だと解釈できる．

---

## 3. 「状態推定」の三つの系譜

同じ "state estimation" でも，分野によって力点が違う．本研究に効くのは主に (A) と (C)．

### (A) ロボティクス／制御：ベイズフィルタと信念状態

古典的な状態推定は **ベイズフィルタ（Bayes filter）** ―― その線形ガウス版が **カルマンフィルタ**，離散版が **HMM の forward 再帰** ―― である（Thrun, Burgard & Fox, *Probabilistic Robotics*, 2005）．観測モデル $P(o_t\mid s_t)$ と状態遷移モデル $P(s_t\mid s_{t-1},a_{t-1})$ を使い，信念を再帰的に更新する：

$$\underbrace{b_t(s)}_{\text{事後}} \propto \underbrace{P(o_t\mid s)}_{\text{観測尤度}} \underbrace{\sum_{s'} P(s\mid s',a_{t-1})\,b_{t-1}(s')}_{\text{予測}}$$

RL と接続すると，「信念 $b_t$ を新しい状態とみなした MDP」＝ **belief-MDP** に帰着し，価値は信念上で $Q(b_t,a)$ と定義される．**ロボット工学で「state estimation が大事」と言うときは，通常この (A) を指す．** 本研究の隠れパターン推定は，このフィルタの2状態版として書ける（→ spec の Route B）．

> リポジトリに既にある `stats/gaussian_hmm.py` は，まさにこの forward-backward による潜在状態推定である．state estimation を「行動データ生成モデルに組み込む」のが次の一歩になる．[[gaussian-hmm]]

### (B) 深層RL／表現学習（SRL）：観測から低次元マルコフ表現を学習する

生の画像のように高次元な観測から，行動選択に必要な情報だけを残した低次元表現 $s = f_\psi(o)$ を **学習** する枠組み（Lesort et al., 2018, *Neural Networks*；survey: arXiv:2506.17518）．手段は AE/VAE による再構成，逆モデル・順モデル，時間的一貫性などの補助損失，あるいは **世界モデル**（Ha & Schmidhuber, 2018；Hafner et al., Dreamer）．部分観測に対しては **リカレント状態**（DRQN: Hausknecht & Stone, 2015；R2D2: Kapturowski et al., 2019）が履歴を要約して状態の代わりを務める．

> 「刺激の画像から状態を推定する」というアイデアはここに属する．ただし RDK 刺激は本質的に **（運動方向, コヒーレンス, 色）で完全に要約できる**ので，学習した encoder が復元するのは結局「運動方向」であり，**手作りの方向特徴 $\phi(\theta)$ がすでに最良の状態表現**になっている（→ spec 補節）．深層 encoder が意味を持つのは，知覚ノイズ（低コヒーレンス）を明示的にモデル化したい場合に限られ，そのときは (C) に合流する．

### (C) 神経科学：隠れ状態推論（hidden-state inference）と信念状態

動物・ヒトの学習を，「観測が曖昧なとき脳は隠れ状態の確率分布（信念状態）を計算し，それに基づいて価値と報酬予測誤差を計算する」と捉える一連の研究：

- **Starkweather, Babayan, Uchida & Gershman (2017, Nature Neuroscience)**：ドパミン RPE が単なる時刻ではなく **隠れ状態推論** を反映する．
- **Babayan, Uchida & Gershman (2018, Nature Communications)**：ドパミン系が **信念状態** を表現する．
- **Gershman, Blei & Niv (2010, Psych Review) / Gershman & Niv (2010)**：**潜在原因（latent cause）** 理論．環境を有限個の潜在レジームの混合とみなし，各試行がどのレジームに属すかを推論しながら学習する．
- **Wilson, Takahashi, Schoenbaum & Niv (2014, Neuron)**：眼窩前頭皮質を **課題状態空間の認知地図** とみなす．
- **Behrens, Woolrich, Walton & Rushworth (2007, Nature Neuroscience)**：環境の **ボラティリティ（切り替わりやすさ＝ハザード率）** 自体を学習する．

> **本研究の「暗黙的学習」は (C) の枠組みにほぼそのまま乗る．** 参加者は「白高／黒高」という潜在レジームを *明示的にラベルとして与えられていない* が，報酬経験から **どちらのレジームにいるかの信念を暗黙に更新** していく．これは latent-cause 推論そのものであり，「気づきの有無」を信念の鋭さ（エントロピー）や更新速度として定量化できる可能性がある．

---

## 4. 現行モデルの何が「矛盾」なのか（状態推定の言葉で）

現行の `q_learning_ooz_map.py` / `q_learning_map.py` は，状態 $s=$ `target_item`（白高／黒高の真のパターンラベル）を **試行ごとに正解として直接与えて** いる：

```python
s = int(states[t])          # ← 真の報酬パターン（オラクル）
q[s, a] += alpha * (r - q[s, a])
```

これは状態推定の観点から言えば：

> **POMDP を，あたかも完全観測 MDP であるかのように偽っている．** エージェント（＝参加者のモデル）に，本来は観測から推定すべき隠れ状態 $z_t$ を *神の視点（oracle）* で手渡している．つまりモデル内の学習者は「初めから今どちらが高報酬か知っている」ことになり，「気づかないまま報酬構造を学ぶ」という研究の中心的主張と真っ向から矛盾する．

修正の方向は2つ（詳細と数式は spec）：

- **Route A（観測を状態とする関数近似）**：隠れパターンラベルを捨て，**観測できる運動方向 $\theta$ を状態入力**として $Q(\theta,a)=w_a^\top\phi(\theta)$ を学習する．円環性は特徴符号化 $\phi$ で解決．**これは既存の `directional_q_learning_map.py` / `von_mises_value_learning_map.py` がすでに実装している道**であり，オラクル問題を解消済みである．[[directional-q-learning]]
- **Route B（状態推定そのもの）**：隠れパターン $z$ が潜在であることを認め，**ベイズフィルタで信念 $b_t$ を更新**し，$Q(b_t,a)$ で行動する．ロボティクス／神経科学の state estimation を忠実に持ち込んだ版で，暗黙学習の記述としては最も筋が良い．

---

## 5. ユーザーが感じた2つの引っかかりへの回答（要約）

姉妹 spec で詳述するが，結論だけ先に：

### 引っかかり1：円環（角度が循環する）

生の角度 $\theta$ をスカラーのまま線形項に入れると，$0$ と $2\pi$ が「同じ点なのに数値的に最遠」となり破綻する．**解決は，生の角度を決して直接使わず，周期的な特徴写像に通すこと**：

$$\text{von Mises RBF: } \phi_j(\theta)=\exp[\kappa\cos(\theta-\mu_j)] \qquad\text{または}\qquad \text{Fourier: } \phi(\theta)=[\cos\theta,\sin\theta,\cos2\theta,\sin2\theta,\ldots]$$

どちらも $\phi(\theta+2\pi)=\phi(\theta)$ を満たし，近い方向が近い特徴ベクトルに写る．**円環性を解くのは「2角度を状態に入れること」ではなく，この基底符号化**である（既存モデルは von Mises で解決済み）．

### 引っかかり2：「$s=$ 角度だと $Q(s,a)$ の index が無意味になる」

これは **完全に正しい直観**で，まさに *連続かつ循環する量を離散的な表の見出し（index）に使うのが誤り* だと言っている．解決は2通り：

1. **関数近似にする（Route A）**：表を引くのをやめ，$Q(\theta,a)=w_a^\top\phi(\theta)$ と **角度の滑らかな関数**にする．離散 index が消えるので「意味のない index」問題も消える．
2. **離散なのは潜在状態のほう（Route B）**：index には角度ではなく **離散の潜在レジーム $z\in\{$白高,黒高$\}$（有限・カテゴリ的）** を使い，$Q(b,a)=\sum_z b(z)\,q(z,a)$ とする．角度は index ではなく **信念を更新する証拠（観測）** としてのみ使う．これなら表 $q(z,a)$ は意味を取り戻す．

> 一言でいえば：**「観測できる連続・循環量（角度）を，そのまま離散の状態キーにしようとした」のが誤りの本体**である．角度は (1) 関数の引数にするか，(2) 潜在状態を推定するための観測にするか，のどちらかで使う．

---

## 6. 読むべき文献（優先度つき）

**まず読む（本研究に直結）**
- Starkweather, Babayan, Uchida & Gershman (2017). *Dopamine reward prediction errors reflect hidden-state inference across time.* Nature Neuroscience. ― 隠れ状態推論と RPE の核心．
- Gershman, Blei & Niv (2010). *Context, learning, and extinction.* Psychological Review. ― 潜在原因（latent cause）理論．
- Wilson, Takahashi, Schoenbaum & Niv (2014). *Orbitofrontal cortex as a cognitive map of task space.* Neuron. ― 課題状態空間＝信念．

**フィルタ／POMDP の基礎**
- Thrun, Burgard & Fox (2005). *Probabilistic Robotics.* ― ベイズ/カルマンフィルタ，状態推定の教科書．
- Kaelbling, Littman & Cassandra (1998). *Planning and acting in partially observable stochastic domains.* Artificial Intelligence. ― POMDP と belief-MDP．
- Behrens, Woolrich, Walton & Rushworth (2007). *Learning the value of information in an uncertain world.* Nature Neuroscience. ― ボラティリティ（ハザード率）学習．

**関数近似・表現学習（Route A / 画像）**
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed., Ch. 9–11（線形関数近似・RBF・Fourier 基底），17.3（POMDP）．
- Lesort, Díaz-Rodríguez, Goudou & Filliat (2018). *State representation learning for control: An overview.* Neural Networks.
- *A Survey of State Representation Learning for Deep RL* (2025). arXiv:2506.17518.

**円統計**
- Mardia & Jupp (1999). *Directional Statistics.* ― von Mises 分布，円環上の確率と推定．

---

## 7. 本資料のまとめ

- **状態 ≠ 観測**．本実験は隠れ報酬パターンを推定する **POMDP**．
- 現行モデルは隠れ状態をオラクルで手渡しており，暗黙学習の前提と矛盾する．
- 直し方は2系統：**(A) 観測（角度）を関数近似の入力にする**（既存 directional モデル）／**(B) 隠れパターンをベイズフィルタで推定する**（真の state estimation）．
- 円環性は **周期的特徴符号化** で，index 問題は **関数近似 or 離散潜在の信念** で解ける．どちらも「生の角度を離散キーにしない」ことが本質．

→ 具体的な数式・疑似コード・パラメータ・評価は [`state_estimation_q_learning_map_spec.md`](state_estimation_q_learning_map_spec.md) へ．
