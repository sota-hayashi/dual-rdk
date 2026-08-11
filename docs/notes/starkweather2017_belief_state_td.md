# Starkweather et al. (2017) のモデル解説 ― 完全観測 (Task 1) と部分観測 (Task 2) の定式化

対象論文：Clara Kwon Starkweather, Benedicte M Babayan, Naoshige Uchida, Samuel J Gershman,
**"Dopamine reward prediction errors reflect hidden-state inference across time"**, *Nature Neuroscience* **20**(4), 581–589 (2017). [doi:10.1038/nn.4520](https://doi.org/10.1038/nn.4520)
（PDF: [`../papers/nn.4520.pdf`](../papers/nn.4520.pdf)）

一行でまとめると：

> **同じ TD 学習でも，状態が完全観測なら価値表現は「時計」（CSC）で足りるが，報酬が省略されうる課題では状態が隠れるため，価値は「今どの状態にいるかの確率分布（信念状態）」の上で計算される必要がある．ドーパミン RPE の時間変調の符号が2課題で逆転することが，その証拠になっている．**

---

## 0. 冒頭：用語の整理

本論文を読む上で最低限必要な語を先に定義しておく．

### 0.1 試行の時間構造：ITI と ISI

古典的条件づけ（パブロフ型）の1試行は，おおむね次の時間構造を持つ．

```
… ITI …… [ CS: odor 1 s ] …… ISI …… [ US: reward ] …… ITI …… [ CS ] …
                ↑ cue onset                ↑ reward
```

| 用語 | 正式名 | 意味 | 本論文での値 |
|---|---|---|---|
| **ISI** | Inter-**Stimulus** Interval | **試行内**の間隔．CS（手がかり刺激）と US（報酬）の間の待ち時間．「報酬をいつ受け取れるか」の時間的期待を規定する | odor A: 1.2–2.8 s を9点に離散化した Gaussian（平均2.0 s，s.d. 0.5 s）／odor B: 1.2 s 固定／odor C: 2.8 s 固定 |
| **ITI** | Inter-**Trial** Interval | **試行間**の間隔．前の試行が終わってから次の CS が出るまでの待ち時間．「次の試行がいつ始まるか」を規定する | 指数分布（平均 12–14 s）。指数分布にするのは **ハザードを一定（フラット）にする**ため＝いつ次の試行が来るか予測できないようにするため |

区別のポイントは **「S = Stimulus 間（試行の中）」vs「T = Trial 間（試行の外）」**．
本論文の設計上の肝は，**ISI をばらつかせた（1.2–2.8 s）** ことで「報酬がいつ来るか」の不確実性を作り，かつ **Task 2 では 10% の確率で報酬を出さない** ことで「そもそも報酬が来るのか」の不確実性を追加した点にある．

### 0.2 CS / US / RPE

- **CS (conditioned stimulus)**：条件刺激．ここでは匂い（odor A–D）．
- **US (unconditioned stimulus)**：無条件刺激．ここでは水報酬（3 µL）．
- **RPE (reward prediction error)**：報酬予測誤差 $\delta(t)$．「実際に得たもの」−「予測していたもの」．中脳ドーパミン細胞の位相性発火がこれを符号化する，というのが標準的仮説．
- **pre-reward firing / post-reward firing**：報酬到来直前（−400〜0 ms）の発火と，報酬直後（+50〜300 ms）の発火．論文はこの2つを別々に定量している．

### 0.3 CSC（complete serial compound）

**CSC = complete serial compound representation**，日本語では「完全直列複合表現」．

TD 学習でドーパミンをモデル化した古典的研究（Schultz–Dayan–Montague 1997, Sutton–Barto 1990）が用いた **時間の表現方法**である．CS 呈示後の経過時間を，

$$
x(t) = (x_1(t), x_2(t), \dots, x_N(t)), \qquad
x_i(t) = \begin{cases} 1 & (i = t - t_{\text{onset}}) \\ 0 & (\text{otherwise}) \end{cases}
$$

という **one-hot ベクトルの系列**（各時刻にちょうど1つのサブ状態が点灯するデジタル時計）で表す．CS onset をトリガーとして $x_1 \to x_2 \to \dots$ と弾道的（ballistic）に進むだけで，途中で観測に応じて分岐したり戻ったりしない．

- 長所：単純．価値を $\hat V(t) = \sum_i w_i x_i(t)$ と線形に書けて，時刻ごとの重み $w_i$ が「その時点での報酬期待」に対応する．
- 短所：**状態が観測から一意に決まる（完全観測）ことを暗黙に前提**している．報酬が来なかったときに「まだ ISI 中なのか，もう試行が終わったのか」を表現できない．

論文の主張は「CSC では Task 2 のデータが説明できない．CSC を **信念状態 $b(t)$** に置き換えれば説明できる」というもの．

### 0.4 belief state（信念状態）とサブ状態

**belief state** $b(t)$ ＝ 「現時点までの観測履歴を条件とした，隠れ状態上の事後分布」．POMDP の標準的な十分統計量．
本論文では ISI 状態・ITI 状態をさらに細かい **sub-state（サブ状態，$i_1 \dots i_{15}$）** に分割し，$b_i(t) = P(\text{サブ状態 } i \mid o_{1:t})$ を保持する．CSC は「$b$ が常に one-hot になる特殊ケース」に他ならない．

### 0.5 hazard function と subjective hazard

- **ハザード関数** $h(t) = \dfrac{f(t)}{1 - F(t)}$：「まだ起きていない」条件のもとで，まさに今イベントが起きる瞬間確率．「まだ報酬が来ない → そろそろ来るはず」という時間的期待の標準的定式化．
- **subjective hazard（主観的ハザード）**：生体の時間計測はスカラー的ノイズ（Weber 則）を持つため，確率密度を経過時間に比例する s.d. のガウシアンでぼかしてから計算するもの（Weber fraction $\phi = 0.25$）：

$$
\tilde f(t) = \frac{1}{\phi t \sqrt{2\pi}} \int_{-\infty}^{\infty} f(\tau)\, e^{-(\tau - t)^2 / (2\phi^2 t^2)} \, d\tau
$$

論文の結論は，**Task 2 のドーパミン応答は（主観的）ハザードでは説明できず，信念状態 TD が必要**というもの（Fig. 7）．

### 0.6 semi-Markov / dwell time

サブ状態の系列は，「ISI 状態にどれだけ滞在するか（dwell time）」の分布を，通常のマルコフ連鎖で近似的に表現する仕掛けである．連続時間の **semi-Markov 過程**（滞在時間分布を直接扱う）と数学的に等価であり，論文はサブ状態版を採用している（CSC との対応が見やすいため）．

---

## 1. 課題デザインの違い

| | **Task 1** | **Task 2** |
|---|---|---|
| odor A–C 後の報酬確率 | **100%** | **90%**（10% は省略 = omission） |
| odor A の ISI | 1.2–2.8 s の離散化 Gaussian（9点） | 同左 |
| odor B / C の ISI | 1.2 s / 2.8 s 固定 | 同左 |
| odor D | 報酬なし | 報酬なし |
| 状態の可観測性 | **fully observable** | **partially hidden** |
| 記録細胞数 | 30 | 43 |

**なぜ Task 1 は完全観測なのか**：cue onset は必ず ITI→ISI 遷移を意味し，reward onset は必ず ISI→ITI 遷移を意味する．報酬は 100% 与えられるので，**「イベントなしに状態が変わる」ことが起こらない**．したがって観測列から現在状態が一意に定まる．残る計算問題は「報酬の予測」だけである．

**なぜ Task 2 は部分観測なのか**：10% の省略試行では，cue が出たのに ISI に入らない（＝ITI が自己遷移する）ことが起きる．つまり

$$
\text{ITI} \to \text{ISI} \quad\text{と}\quad \text{ITI} \to \text{ITI} \quad\text{が\textbf{同じ観測（cue）}を生成する}
$$

ため，cue を見ても真の状態が判別できない．**Task 2 は追加の計算問題「隠れ状態推論」を課している**．これが2課題の唯一の構造的差分であり，実験デザインとして非常にクリーンな対比になっている．

### 1.1 実験結果（モデルが説明すべきターゲット）

| | pre-reward firing の ISI 依存 | **post-reward firing の ISI 依存** |
|---|---|---|
| Task 1 | 時間とともに **減少** | 時間とともに **減少**（最短 ISI で最大） |
| Task 2 | 時間とともに **減少** | 時間とともに **増加**（最長 ISI で最大） |

**post-reward RPE の符号が逆転する**のが核心。個別ニューロンでも Task 1 は 23/30 が負の傾き，Task 2 は 33/43 が正の傾き（Fig. 3）。

---

## 2. 共通の TD 学習の枠組み

両課題に共通する骨格は同じで，**状態表現だけが違う**．

価値の定義（式1）：

$$
V(t) = \mathbb{E}\left[ \sum_{\tau = t}^{\infty} \gamma^{\tau - t} r(\tau) \right]
$$

特徴ベクトル $\phi(t)$ による線形近似：

$$
\hat V(t) = \sum_i w_i \, \phi_i(t)
$$

TD 誤差（式4）と重み更新（式3）：

$$
\delta(t) = r(t) + \gamma \hat V(t+1) - \hat V(t), \qquad
\Delta w_i = \alpha \, \phi_i(t) \, \delta(t)
$$

シミュレーションのパラメータ：**学習率 $\alpha = 0.1$，割引率 $\gamma = 0.98$**，5,000 試行学習（約1,000試行で漸近，本文の図は2,000–5,000試行から取得）．時間の離散単位は 200 ms．

**この $\delta(t)$ がドーパミン発火に対応する**という仮定のもとで，$\phi(t)$ に何を入れるかが Task 1 と Task 2 で分かれる．

---

## 3. Task 1：完全観測の場合（CSC = 縮退した信念状態）

### 3.1 状態表現

完全観測なので，観測履歴から現在のサブ状態が一意に定まる．特徴は CSC の one-hot ベクトル：

$$
\phi_i(t) = x_i(t) = \mathbb{1}[\,i = t - t_{\text{onset}}\,]
$$

$$
\hat V(t) = \sum_i w_i x_i(t) \tag{式2}
$$

すなわち **時刻 $t$ において点灯しているサブ状態の重み $w_i$ がそのまま価値**になる．
信念状態の言葉でいえば

$$
b_i(t) = \delta_{i,\,t - t_{\text{onset}}} \quad (\text{one-hot})
$$

であり，**Task 1 では belief state TD と CSC TD が完全に一致する**（論文 Fig. 5c）。手がかり onset の瞬間に「ISI サブ状態のどれかにいる確率 100%，ITI にいる確率 0%」と確定し，あとは時間経過とともに 100% の確率が $i_1 \to i_2 \to \dots$ と順送りされるだけ．

### 3.2 価値関数の形と RPE の予測

報酬が Gaussian ISI 分布に従うので，**遅い ISI サブ状態ほど「そこに到達すれば報酬が近い」ため大きな重みを獲得する**（$w_i$ が $i$ とともに増加）。したがって

$$
\hat V(t) \nearrow \quad (\text{ISI 中，時間とともにランプ状に上昇})
$$

$\hat V$ が上昇するほど報酬は「予測済み」になるので，報酬時の RPE

$$
\delta(t_{\text{reward}}) = r + \gamma \hat V(t+1) - \hat V(t)
$$

は **遅い ISI ほど強く抑制される → post-reward RPE は時間の減少関数**．これは Task 1 のデータ（負の傾き）と一致する．

### 3.3 うまくいかない対抗モデル（参考）

| モデル | Task 1 | Task 2 | 結論 |
|---|---|---|---|
| **素の CSC TD** | 反転したガウス分布状の RPE（分布の中心で最も抑制） | 同左（両課題で同一） | 両課題を区別できず ✗ |
| **CSC + reset**（報酬後に RPE を 0 にリセット） | 減少 ✓ | **減少**（データは増加） | Task 2 で ✗ |
| **hazard / subjective hazard** | 減少 ✓ | 最短 ISI で最小（データは最大） | Task 2 で ✗ |
| **belief state TD** | 減少 ✓ | 増加 ✓ | ✓ |

CSC の「reset」は，本質的には**報酬観測によって状態が変わったという推論を手で埋め込んだもの**であり，信念状態モデルの粗い近似と見なせる（Daw et al. 2006 の指摘）．

---

## 4. Task 2：部分観測の場合（belief state TD）

### 4.1 生成モデル（POMDP）

隠れ状態は **15個のサブ状態** $i_1, \dots, i_{15}$：

| サブ状態 | 対応 |
|---|---|
| $i_1$–$i_5$ | odor 呈示中の 1 s（5 × 200 ms） |
| $i_6$–$i_{14}$ | 報酬が来うる 9 つのタイミング（1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8 s） |
| $i_{15}$ | **ITI** |

遷移行列 $T(j, i) = p(i \mid j)$，観測行列 $O(j, i, k) = p(o = k \mid j \to i)$．観測は3種：$o \in \{\text{null}, \text{cue}, \text{reward}\}$（$k = 1,2,3$）．

**ISI 内部の遷移**は ISI 分布のハザードで与える．例えば 1.2 s に報酬を受け取る確率（＝ハザード）は $T(6,15)$，そのまま次のサブ状態へ進む確率は $T(6,7) = 1 - T(6,15)$，という具合に $i_6 \dots i_{14}$ を設定する．

**ITI の滞在時間**はハザード一定の指数分布：`ITI_hazard` $= 1/65$（平均滞在 65 時間単位；実験の平均 ITI に比例させてある）．

**2課題の差はここだけ**：

$$
\textbf{Task 1:}\quad T(15,15) = 1 - \texttt{ITI\_hazard}, \qquad T(15,1) = \texttt{ITI\_hazard}
$$

$$
\textbf{Task 2:}\quad T(15,15) = 1 - (\texttt{ITI\_hazard} \times 0.9), \qquad T(15,1) = \texttt{ITI\_hazard} \times 0.9
$$

> 注：論文 Online Methods の該当箇所は Task 1 の自己遷移が `T(15,1)` と誤記されている（同じ添字が2行続く）。文脈上，1行目は自己遷移 $T(15,15)$ が正しい．

観測モデルは，「ITI→ISI 遷移には必ず cue 観測が伴う」「ISI 途中→ITI 遷移には必ず reward 観測が伴う」ことを表す：

$$
O(15, 1, 2) = 1 \quad (\text{ITI} \to \text{ISI 開始には cue が必要}), \qquad
O(10, 15, 3) = 1 \quad (\text{ISI 途中} \to \text{ITI には reward が必要})
$$

そして **本質的な差分**：

$$
\textbf{Task 1:}\quad O(15,15,1) = 1 \quad (\text{ITI 自己遷移は必ず無観測})
$$

$$
\textbf{Task 2:}\quad O(15,15,1) = 1 - (\texttt{ITI\_hazard} \times 0.1), \qquad
O(15,15,2) = \texttt{ITI\_hazard} \times 0.1
$$

つまり Task 2 では **「ITI に留まったまま cue だけが観測される」経路が存在する**．これが省略試行であり，これゆえに cue を見ても ISI に入ったかどうか確信できない＝**部分観測**になる．

### 4.2 信念状態の更新（ベイズフィルタ）

論文の式5：

$$
b_i(t) \;\propto\; p\big(o(t) \mid i\big) \sum_j p(i \mid j)\, b_j(t-1)
\tag{5}
$$

右辺の $\sum_j p(i\mid j) b_j(t-1)$ が**予測ステップ**（遷移で1ステップ進める），$p(o(t)\mid i)$ の掛け算が**更新ステップ**（観測による尤度重み付け），最後に総和1へ正規化．これは標準的な HMM の forward 再帰＝ベイズフィルタそのものである（[`bayes_filter_primer.md`](bayes_filter_primer.md) 参照）．

### 4.3 価値と RPE

CSC ベクトル $x(t)$ を信念ベクトル $b(t)$ に **差し替えるだけ**：

$$
\boxed{\;\hat V(t) = \sum_i w_i \, b_i(t)\;}
$$

$$
\delta(t) = r(t) + \gamma \hat V(t+1) - \hat V(t), \qquad
\Delta w_i = \alpha \, b_i(t)\, \delta(t)
$$

**価値は「重みベクトルと信念ベクトルの内積」**になる．学習則も同様に，信念に比例して credit が各サブ状態に分配される（one-hot なら1つに全振り，分散していれば按分）．

### 4.4 なぜ post-reward RPE が「増加」に転じるのか

Task 2 の信念のダイナミクスは次の通り：

1. cue 観測直後：$b$ は ISI サブ状態側に大きな確率（≈90%），ITI に 10% 程度を残す．
2. 時間が経っても報酬が来ない → 「もし本当に ISI にいたなら，そろそろ報酬が来ているはず」→ **ISI 仮説の尤度が下がり，確率質量が $i_{15}$（ITI）へ流れ込む**．
3. したがって遅い ISI サブ状態 $i_{11}, \dots, i_{14}$ には，そもそも**信念がほとんど乗らない**．

学習則 $\Delta w_i \propto b_i(t) \delta(t)$ より，信念が乗らないサブ状態は重みを積めるが（報酬が来た時は $b$ が反応するので $w_i$ 自体は大きくなる），**価値 $\hat V(t) = \sum_i w_i b_i(t)$ は「大きな重み × 小さな信念」となり，時間とともに減少する**（Fig. 6c,d）．

$$
\textbf{Task 1:}\ \hat V(t)\ \nearrow \ \Rightarrow\ \delta(t_{\text{rew}})\ \searrow
\qquad
\textbf{Task 2:}\ \hat V(t)\ \searrow \ \Rightarrow\ \delta(t_{\text{rew}})\ \nearrow
$$

**遅い ISI で報酬が来ると「もう省略だと思っていたのに来た」＝驚きが大きい**．これが post-reward RPE の正の時間変調の直感的な説明であり，データと一致する．

一方 **pre-reward RPE は両課題とも減少**する．これも信念状態モデルが同時に再現している．

---

## 5. 2つの定式化の対応表

| | **Task 1（完全観測）** | **Task 2（部分観測）** |
|---|---|---|
| 状態表現 | $x(t)$：one-hot（CSC） | $b(t)$：$\Delta^{14}$ 上の確率分布 |
| 状態の同定 | 観測から一意 | 事後分布のみ（ベイズフィルタ式5） |
| 価値 | $\hat V = \sum_i w_i x_i(t)$ | $\hat V = \sum_i w_i b_i(t)$ |
| 更新 | $\Delta w_i = \alpha x_i \delta$ | $\Delta w_i = \alpha b_i \delta$ |
| $\hat V(t)$ の時間変化 | 単調増加（ランプ） | 途中でピークを打ち **減少**（Fig. 6c,d） |
| post-reward RPE | ISI とともに減少 | ISI とともに **増加** |
| モデル間の関係 | belief state モデルの $b$ が one-hot に縮退した特殊ケース | 一般形 |

**要するに，2つのモデルは別物ではなく，$b(t)$ が退化するか否かの1点で繋がっている．** 論文が「同じ TD 学習則」を保ったまま観測構造だけを変えて2課題を説明できたのが，主張の強さの源になっている．

---

## 6. 補足：論文が明示的に排除したこと・残した限界

- **hazard / subjective hazard による説明の棄却**（Fig. 7）：ハザード説は「経過時間とともに期待が高まる」ため最短 ISI で RPE が最大になることを予測できない．Task 2 データは最短 ISI で最も抑制されていた（＝期待が最大）．
- **一様 ISI 分布で学習させても結論は不変**（Suppl. Fig. 10）：動物が Gaussian 分布を完璧に学習していたという仮定への頑健性チェック．
- **Task 1 の post-reward RPE の絶対値**はモデル予測より大きい．著者らは匂い検知タイミングの試行間ジッタ（モデルに未実装）と訓練期間の短さ（1–2週間）を理由に挙げている．
- **スカラー時間ノイズは基本モデルでは省略**．信念分布を経過時間に比例する s.d. のガウシアンでぼかす拡張（blurred belief state）でも結論は保たれる（Suppl. Fig. 12）．
- **microstimulus モデルとの関係**：microstimulus は固定形状の時間受容野を持つ特徴だが，belief state は課題構造（省略確率）に応じて形が変わる．著者らは microstimulus を belief state の神経実装と見なせると論じている．

コード：<https://github.com/cstarkweather>（論文記載）

---

## 7. 本リポジトリ（dual-RDK）への含意

- 本論文の Task 1 / Task 2 の対比は，**「オラクル状態を与える設計」と「観測から隠れ状態を推論させる設計」の対比そのもの**である．RDK 課題で隠れ報酬パターンを推論させる方向性（[[pomdp-state-estimation-direction]]）と直接対応する．
- 実装上の教訓：**学習則は変えず，特徴ベクトルを one-hot → 信念ベクトルに差し替えるだけ**で POMDP 化できる．価値・更新の式がそのまま流用できるので，既存の Q 学習コード（`src/dualrdk/models/state_estimation_q_learning_map.py`）への接続は素直．
- 設計上の教訓：**信念が動くには「観測が状態を一意に決めない」構造が必要**．Starkweather の場合それは「cue が出ても 10% は ITI のまま」という遷移／観測の作り込みで担保されている．ランダム化しすぎて信念が動かない設計（[[route-b-latent-filter-degenerate]]）を避けるには，このように **観測の曖昧性を明示的に生成モデルへ埋め込む**ことが要る．
