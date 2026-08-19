# dual-RDK：POMDP / Q 学習モデルの実装要件定義書

- 版: 3.0
- 対象: learning ステージ48試行の行動データに対するモデル実装
- 課題仕様（刺激生成・報酬関数・実験手続き）は本書の対象外．`experiment/index.html` および `reports/ieicej3.4a/` を参照．

---

## 0. 前提

### 0.1 モデル化の範囲

- **learning ステージ48試行のみ**をモデル化する．練習16試行と awareness ステージ24試行は尤度に含めない．
- awareness ステージは参加者除外の算出にのみ使用済み．したがって遷移核は恒等写像で完結し，ハザード率は導入しない．
- 参加者除外（4基準・26名）と試行ごとの注意状態ラベルは**前処理済みとして入力に含まれる**．本実装では再計算しない．

### 0.2 入力データ

```python
concat_list: List[Tuple[str, pd.DataFrame]]
```

各タプルは `(subj_id, df)`．使用カラム：

| 列 | 型 | 単位 | 説明 |
|---|---|---|---|
| `num_trial` | int | | 1〜48（learning ステージ内で連続） |
| `rt` | float | ms | 反応時間．NaNの試行は除外 |
| `session_rotation` | float | deg | セッション回転量($\varphi$)．**任意**．尤度計算には不要で（M2 系は $\theta^W,\theta^B,a,r$ のみ，M3 系も信念を一様から始めるため $\varphi$ を知らない），検証と参加者間の座標回転にのみ使う．欠測時は該当検証をスキップする |
| `response_angle_rdk` | float | deg | 参加者が報告した方向 |
| `target_direction` | float | deg | ターゲット（高報酬側）ドット群の運動方向 |
| `distractor_direction` | float | deg | ディストラクタ（低報酬側）ドット群の運動方向 |
| `target_group` | str | |ターゲットの色 `"white"`/`"black"`（**色別方向の復元にのみ使用**） |
| `reward_points` | int | |0〜10の報酬値（色×正確さが畳み込み済み）．$[0,1]$ にスケーリング |
| `ooz` | int | |OOZ ラベル $z_t \in \{0,1\}$．`label_if_ooz` で事前付与．学習率の切り替えに使用 |

#### 色別運動方向の復元

モデルの選択計算には各試行の **白の運動方向 $\theta_{\text{white}}$** と **黒の運動方向 $\theta_{\text{black}}$** が必要．これを `target_direction`/`distractor_direction` と `target_group` から復元する：

```
target_group == "white":  θ_white = target_direction,     θ_black = distractor_direction
target_group == "black":  θ_white = distractor_direction, θ_black = target_direction
```

### 0.3 実装が依存する課題の定数

| 量 | 値 |
|---|---|
| 潜在報酬軸 | $\psi = \varphi + 67.5°$ |
| 高報酬側の判定 | $\theta^\star = \arg\min_{\theta \in \{\theta^W, \theta^B\}} d(\theta, \psi)$ |
| 報酬関数 | $r = \max\{0,\ \lfloor 10 - \lvert e \rvert/5 \rfloor\}$，$e = \mathrm{wrap}(a - \theta^\star)$ |
| 報酬 > 0 の条件 | $\lvert e \rvert \le 45°$ |
| 2方向の角度差 | $\Delta \in [90°, 134°]$（試行ごとに変動．$90°$ 固定ではない） |

円距離は $d(x,y) = \lvert\operatorname{atan2}(\sin(x-y),\ \cos(x-y))\rvert$．

### 0.4 記号

| 記号 | 意味 |
|---|---|
| $b_t(\psi)$ | $\psi$ についての信念（円周上の確率質量） |
| $q_t$ | 「白が高報酬側」信念確率 |
| $d_t$ | 決定変数（モデルごとに定義） |
| $p^W_t$ | 白を選ぶ確率 |
| $z_t$ | 注意状態 $\in \{\text{in}, \text{out}\}(z_t = 0:\text{in}, z_t = 1:\text{out})$ |

---

## 1. POMDP 定式化

### 1.1 構成要素

| 要素 | 定義 |
|---|---|
| 状態 $\mathcal{S}$ | $\psi \in S^1$（潜在報酬軸） |
| 行動 $\mathcal{A}$ | $a \in S^1$（報告方向） |
| 観測 $\mathcal{O}$ | $o_t = (\theta^W_t, \theta^B_t)$ |
| 遷移核 $T$ | $T(\psi' \mid \psi, a) = \delta(\psi' - \psi)$ |
| 報酬 $R$ | §0.3 |

### 1.2 遷移核が恒等写像であることの実装上の帰結

$\delta$ はデルタ（点質量の密度）．$\psi$ が連続変数なので $T$ は $\psi'$ についての密度でなければならず，「確率1で留まる」を素の数値では書けない．行動 $a$ が右辺に現れないことは「行動は状態を動かせない」を意味する．

篩の性質 $\int \delta(x-c) f(x)\,dx = f(c)$ より，予測ステップは

$$b^-_{t+1}(\psi') = \int \delta(\psi' - \psi)\, b_t(\psi)\, d\psi = b_t(\psi')$$

**FR-1.1**: 予測ステップは恒等作用素であるため実装しない（no-op）．離散化すれば遷移行列は単位行列である旨をコメントに残すこと．

**FR-1.2**: $\psi$ は動かないが $b_t$ は動く．belief-MDP のダイナミクスは「エージェントの知識が動く」ことに由来する．$s_t$ は毎試行更新される可変状態として扱う．

### 1.3 エージェントのマルコフ状態

$$s_t = \big(b_t,\ \theta^W_t,\ \theta^B_t\big)$$

**FR-1.3**: 状態に $\psi$ を用いてはならない．$\psi$ は環境側の隠れ変数で履歴可測でない．エージェントが持てるのは $b_t$ のみ．また刺激 $(\theta^W_t, \theta^B_t)$ を状態に含めないと $p(r \mid s, a)$ が定まらない（報酬十分性が破れる）．

### 1.4 信念のスカラーへの縮約

$$q_t = \int_{\{\psi\,:\, d(\theta^W_t, \psi) < d(\theta^B_t, \psi)\}} b_t(\psi)\, d\psi$$

積分範囲は $\theta^W_t$ と $\theta^B_t$ の垂直二等分線が切る $180°$ の弧．

**FR-1.4**: `belief_to_q(log_b, theta_w, theta_b)` を独立関数として実装する．信念は意思決定に対してはこのスカラー1個に圧縮される．

---

## 2. 信念表現

### 2.1 グリッド

刺激角が整数度なので $1°$ 刻みとする．

```
GRID_N   = 360
psi_grid = np.arange(360)                  # degrees
log_b    = np.full(360, -np.log(360))      # 初期信念：一様
```

**FR-2.1**: 信念は**対数空間の確率質量**として保持する（密度ではない）．`logsumexp(log_b) == 0` を不変条件とする．

### 2.2 尤度

$\psi$ の下での高報酬側と，報酬の符号との整合で定義する（順序尤度）．

$$\theta^\star(\psi) = \begin{cases}\theta^W_t & d(\theta^W_t,\psi) < d(\theta^B_t,\psi)\\ \theta^B_t & \text{otherwise}\end{cases}$$

$$\mathrm{consistent}_t(\psi) \ \equiv\ \Big(\big[\,d(a_t,\ \theta^\star(\psi)) \le 45°\,\big] = \big[\,r_t > 0\,\big]\Big)$$

汚染混合 $L = (1-\epsilon)\,\mathbb{1}[\mathrm{consistent}] + \epsilon$ を代入すると，実装は1行になる．

```python
log_L = np.where(consistent, 0.0, np.log(eps))
```

**FR-2.2**: 報酬が決定論的なため，$\epsilon$ なしでは $\log L = -\infty$ が伝播し，1試行の逸脱で事後が不可逆に潰れる．$\epsilon$ は必須の自由パラメータとする．$\epsilon$ が吸収するのは知覚ノイズとモデル誤特定であり，反応 lapse ではない（尤度は実際の $a_t$ で条件づけているため lapse は尤度を壊さない）．

**FR-2.3**: `logsumexp(log_b + log_L) == -inf` を検知したら例外を送出する．黙って一様分布にリセットしてはならない．

### 2.3 更新

$$\log b_{t+1} = \log b_t + \log L_t - \operatorname{logsumexp}(\log b_t + \log L_t)$$

**FR-2.4**: 正規化定数 $\operatorname{logsumexp}(\log b_t + \log L_t)$ はその試行の予測尤度 $p(r_t \mid \text{履歴})$ である．診断用に毎試行記録する．

### 2.4 閉形式（単体テスト用）

予測ステップが恒等なので，フィルタは矛盾回数の数え上げに退化する：

$$\log b_t(\psi) = \log b_0(\psi) + n_t(\psi)\log\epsilon, \qquad n_t(\psi) = \#\{k \le t : \psi\ \text{が試行}\ k\ \text{と矛盾}\}$$

**FR-2.5**: 逐次フィルタ実装と，この直接カウント実装が一致することを単体テストとする．

---

## 3. Q 学習モデル

### 3.1 方策（全モデル共通）

報酬の山は幅 $\pm 45°$ で，2方向は $\Delta \ge 90°$ 離れているため**互いに素**である．したがって最適行動は必ず $\theta^W_t$ か $\theta^B_t$ のいずれかであり，中間を報告する利得はない．方策は2峰混合とする．

$$p^W_t = \sigma\big(\beta\, d_t - c_{\text{black}}\big), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

$$\pi_t(a) = (1-\lambda)\Big[\,p^W_t\,\mathrm{vM}(a;\theta^W_t,\kappa) + (1-p^W_t)\,\mathrm{vM}(a;\theta^B_t,\kappa)\,\Big] + \frac{\lambda}{2\pi}$$

| パラメータ | 何から推定されるか |
|---|---|
| $\beta$ | 峰の重みが $d_t$ に追随する度合い |
| $c_{\text{black}}$ | 価値と無関係な色の偏り（$>0$ で黒寄り） |
| $\kappa$ | 各峰まわりの広がり |
| $\lambda$ | 一様な底上げ成分 |

**FR-3.1b**: $c_{\text{black}}$ を必ず含める．これが無いと，価値差 $d_t$ で説明できない系統的な色の偏りが $d_t$ に押し付けられる．実データでは観測 $P(\text{白選択}) = 0.466$ に対し $c_{\text{black}}$ 無しのモデル平均 $p^W = 0.492$ で，較正表の $p^W \in (0.5, 0.7]$ 帯（全体の 30%）に約 $5\sigma$ のずれが出ていた．既存の2値実装が持っていた項に対応する．

**FR-3.1**: von Mises 対数密度は指数スケール済みベッセル関数で実装する．

```python
def log_vonmises(x, mu, kappa):
    return kappa * (jnp.cos(x - mu) - 1.0) - jnp.log(2 * jnp.pi) - jnp.log(i0e(kappa))
```

素の `i0` は $\kappa \gtrsim 700$ でオーバーフローする．

**FR-3.2**: 2峰性は仮定であって検証結果ではない．フィット前に $a_t$ の $\theta^W_t / \theta^B_t$ 相対角ヒストグラムを描き，中間に質量がないことを確認する．

### 3.2 報酬の正規化

**FR-3.3**: モデル内部では $\tilde{r}_t = r_t / 10 \in [0,1]$ を用いる．これによりモデル間で $d_t$ のスケールが揃い，$\beta$ の事前分布を共通化できる．

### 3.3 モデル族

決定変数 $d_t$ の生成規則のみが異なる．**M2z が主モデル**である．

| モデル | 観測空間 | $d_t$ | パラメータ |
|---|---|---|---|
| M0 | 角度 | $c$（定数） | $c, \kappa, \lambda$ |
| M1 | **2値** | $Q_t[s_t, W] - Q_t[s_t, B]$ | $\beta, \alpha, c_{\text{black}}$ |
| M1z | **2値** | 同上（$\alpha$ が注意状態依存） | $\beta, \alpha_{\text{in}}, \alpha_{\text{out}}, c_{\text{black}}$ |
| M2 | 角度 | $V_t(\theta^W_t) - V_t(\theta^B_t)$ | $\beta, \alpha, \kappa_{\text{gen}}, \kappa, \lambda, c_{\text{black}}$ |
| **M2z** | 角度 | 同上（$\alpha$ が注意状態依存） | $\beta, \alpha_{\text{in}}, \alpha_{\text{out}}, \kappa_{\text{gen}}, \kappa, \lambda, c_{\text{black}}$ |
| M3 | 角度 | $2q_t - 1$ | $\beta, \epsilon, \kappa, \lambda, c_{\text{black}}$ |
| M3z | 角度 | 同上（$\epsilon$ が注意状態依存） | $\beta, \epsilon_{\text{in}}, \epsilon_{\text{out}}, \kappa, \lambda, c_{\text{black}}$ |

M0 では $\beta$ と $c$ が縮退するため $\beta \equiv 1$ に固定する．

**FR-3.6**: 観測空間が2種類あることに注意．M1系は2値選択（確率質量），それ以外は円環上の応答角（確率密度）を予測する．**この2つの対数尤度・ELPD を同じ表に並べてはならない**（単位が違う）．族をまたぐ比較は §6.5 の共通指標のみで行う．

#### M1 / M1z：2値状態・2値行動の表形式 Q 学習

状態 $s_t \in \{0,1\}$（どちらの雲が高報酬側か）をデータから直接与え，$Q \in \mathbb{R}^{2\times2}$ を更新する．

$$Q_{t+1}[s_t, a_t] = Q_t[s_t, a_t] + \alpha_t\big(\tilde{r}_t - Q_t[s_t, a_t]\big), \qquad Q_0 \equiv 0.5$$

$$d_t = Q_t[s_t, W] - Q_t[s_t, B], \qquad \log \pi_t = \log \sigma\big(\pm(\beta d_t - c_{\text{black}})\big)$$

**これは仮定である**．参加者は試行1の時点で刺激が2群に分かれることを知らず，どちらの群が報酬をもたらすかも知らない．M1系は**カテゴリ化のコストをゼロと置き**，正しい2分割を最初から誤差なく持っている参加者をモデル化している．M2 / M3 系はこの仮定を置かず，$V$ や $b_t$ から自力で導く．**M1 と M2 の比較はこの仮定の当否を測るものである**．

行動も2値なので，書き込んだセルは次に同じ状態が来たとき必ず読まれる．したがって汎化カーネルを必要としない．応答角の情報は捨てるので $\kappa, \lambda$ も持たない．

#### M2 / M2z：方向ベース Q 学習

絶対方向上の価値関数を von Mises 汎化カーネルで更新する．

$$V_{t+1}(\theta) = V_t(\theta) + \alpha_t\, K(\theta - a_t)\,\big(\tilde{r}_t - V_t(a_t)\big), \qquad K(x) = \exp\big(\kappa_{\text{gen}}(\cos x - 1)\big)$$

$$V_0 \equiv 0, \qquad \alpha_t = \begin{cases}\alpha_{\text{in}} & z_t = \text{in}\\ \alpha_{\text{out}} & z_t = \text{out}\end{cases} \quad \text{(M2z)}$$

$V$ は $360$ 次元のグリッドベクトル．刺激角も報告角も整数度なので**補間は不要**でインデックス参照でよい．M2 では $\alpha_{\text{in}} = \alpha_{\text{out}} = \alpha$．

#### M3 / M3z：信念ベース

$q_t$ は §1.4，信念更新は §2．M3z では $\epsilon$ を注意状態依存にする．

$$\log L_t = \begin{cases}0 & \mathrm{consistent}\\ \log \epsilon_{z_t} & \text{otherwise}\end{cases}$$

$\epsilon$ は事実上の学習速度パラメータである（小さいほど1試行の証拠を強く取り込む）．M2 の $\alpha$ との違いは速度そのものではなく**汎化構造**にある：M2 は $\kappa_{\text{gen}}$ 幅の局所更新，M3 は半円制約により円周の半分が一度に更新される．

**FR-3.4**: 全モデルを共通インターフェースで実装する．

```python
class Model(Protocol):
    def init_state(self, params) -> State: ...
    def decision_variable(self, state, obs) -> float: ...            # d_t
    def update(self, state, obs, action, reward, zone, params) -> State: ...
```

### 3.4 割引率

**FR-3.5**: $\gamma = 0$（myopic）とする．遷移核が恒等写像であるため行動は将来の環境状態を変えられず，環境由来の時間的クレジット割当が存在しない．

---

## 4. 参加者は報酬関数を知らない

### 4.1 環境の $R$ とエージェントの $\hat{R}$ の分離

| | 環境の $R$ | エージェントの $\hat{R}$ |
|---|---|---|
| 何か | 実験プログラムに実装された事実 | 参加者が思っている報酬の決まり方 |
| 精密に定義するか | する（§0.3） | パラメトリックにして自由度を残す |

$R$ はデータを生成した客観的事実であり，参加者の知識状態とは無関係に存在する．参加者が知らないという事実が制約するのは $\hat{R}$ の方である．

$\hat{R}$ が入る箇所は2つある．

```
(a) 尤度 L_t(psi)  : 「r_t が出た．psi はどれくらい尤もらしいか」  -> M3 / M3z のみ
(b) 決定変数 d_t   : 「a と報告したら何点もらえるか」               -> どのモデルも使わない
```

**M2 / M2z は $\hat{R}$ を一切必要としない**（観測した $\tilde{r}_t$ をそのまま予測誤差に使う）．(b) については，§3.1 の方策が報酬の**ピーク位置**（選んだクラウドの真の運動方向）しか使わず，傾き・打ち切り・量子化・最大値を使わない．

### 4.2 教示による正当化

教示文は「報酬は報告角度の正確さに応じて与えられる」と明示しており，参加者は $\hat{R}$ の**形**を知らされている．一方 $\psi$ の存在も方向随伴性も知らされていない．これは POMDP の構造に対応する．

```
R_hat の形（角度誤差の減少関数）  = 既知（教示による）
psi（どちらを報告すべきか）        = 未知．推論対象
```

具体的な数値（最大10点，$45°$ 打ち切り，`floor` 量子化）は知らされていない．

### 4.3 実装要件

**FR-4.1**: $\hat{R}$ を `agent_reward.py` として独立モジュールに置き，環境の $R$ と関数を共用しない．

**FR-4.2**: $\hat{R}$ は**順序尤度**（§2.2）とする．「点が入ったか否か」の1ビットのみを使うため，$R$ の傾き・量子化・最大値にほぼ非依存であり，参加者の知識状態についての仮定が最も弱い．

**FR-4.3**: 感度分析として `smooth` バリアント $\hat{R}(x) = \max\{0,\ 1 - \lvert x \rvert/w\}$（主観的幅 $w$ を自由パラメータ）を実装し，M3 系のモデル順位が変わらないことを確認する．変わる場合はその旨を報告する．

---

## 5. 尤度とパラメータ推定

### 5.1 対数尤度

$$\log \mathcal{L}(\vartheta \mid \mathcal{D}_i) = \sum_{t=1}^{48} \log \pi_t\big(a_t;\ \vartheta\big)$$

$\pi_t$ は $s_t$ に依存し，$s_t$ は前試行までの $(o, a, r, z)$ から逐次的に構成される．**試行は条件付き独立ではない．**

**FR-5.1**: 尤度は試行を逐次スキャンして計算する（`jax.lax.scan`）．ベクトル化して独立試行として扱ってはならない．

### 5.2 パラメータ変換

**FR-5.2**: 有界パラメータは無制約空間で扱う．

| パラメータ | 範囲 | 変換 |
|---|---|---|
| $\beta,\ \kappa,\ \kappa_{\text{gen}}$ | $(0,\infty)$ | $\log$ |
| $\alpha_{\text{in}},\ \alpha_{\text{out}},\ \epsilon_{\text{in}},\ \epsilon_{\text{out}},\ \lambda$ | $(0,1)$ | $\mathrm{logit}$ |

### 5.3 事前分布

$$\log\beta \sim \mathcal{N}(\log 5,\ 1.0), \qquad \log\kappa \sim \mathcal{N}(\log 15,\ 0.7), \qquad \log\kappa_{\text{gen}} \sim \mathcal{N}(\log 2,\ 0.8)$$

$$\mathrm{logit}(\alpha) \sim \mathcal{N}(-1,\ 1.2), \qquad \epsilon \sim \mathrm{Beta}(1, 20), \qquad \lambda \sim \mathrm{Beta}(1, 20)$$

$\kappa = 15$ は円周標準偏差 $\approx 15°$ に対応し，既報の $\mathrm{AE}_{\min}$ の水準（10〜30°）と整合する．$\kappa_{\text{gen}} = 2$ は汎化幅 $\approx 45°$ に対応し，報酬の山の幅と整合する．

### 5.4 階層ベイズ

**FR-5.3**: 主推定は階層ベイズで行う．48試行・最大7パラメータでは個人ごとの独立推定は不安定である．

**FR-5.3b**: ただし MAP（参加者ごとに独立な事後モード）も同じインターフェースで選べるようにする（`--method map`）．既存の2値 Q 学習解析が参加者ごとの MAP だったため，**推定法だけを揃えた比較**ができなければ既存結果と接続できないため．事前分布・モデル・評価コードは両者で共通にし，最適化は無制約空間で行う（階層モデルが事前分布を無制約スケールで置いているため，制約空間で最頻値を取るとヤコビアンの分だけ別の量になる）．

**FR-5.3c**: MAP と階層ベイズをインサンプル指標で比較する場合，**必ず `--train-trials` による分割検証を併記する**．MAP は参加者ごとに自由に当てはめるため，当てはまりの良さが過適合か否かをインサンプル指標だけでは判別できない．

### 5.4b 2段階推定（知覚パラメータの事前固定）

**FR-5.10**: $\kappa$（応答角の精度）と $\lambda$（ラプス率）は，応答角と2雲の位置だけで決まり，学習について情報を持たない．リカバリでの復元相関はそれぞれ 0.987 / 0.990 と他のどのパラメータよりも高い．したがって段階1で先に推定して段階2で固定してよい．

段階1のモデル（学習を含まない）：

$$a_t \sim (1-\lambda)\big[\,w\,\mathrm{vM}(a;\theta^W_t,\kappa) + (1-w)\,\mathrm{vM}(a;\theta^B_t,\kappa)\big] + \frac{\lambda}{2\pi}$$

$w$ は白選択の周辺確率で，純粋な nuisance である．固定により M2z の推定対象は $7 \to 5$，階層モデルの可変量は $7\times(2+54) = 392 \to 280$ に減る．

段階1の不確実性を段階2に伝えない点が理論的な弱点である．復元相関 0.99 の量については実害が無いと判断するが，`--fix-perceptual` を外した推定と結果を比べる感度分析を行うこと。

無制約空間で non-centered parameterization を用いる．

$$\tilde\vartheta_{i,k} = \mu_k + \sigma_k\, \eta_{i,k}, \qquad \eta_{i,k} \sim \mathcal{N}(0,1), \qquad \mu_k \sim \mathcal{N}(m_k, s_k^2), \qquad \sigma_k \sim \mathrm{HalfNormal}(1)$$

**FR-5.4（重要）**: `alpha_out - alpha_in` を**個人ごとに推定しようとしない**．out-of-zone 試行は48試行の一部にすぎない．群レベルの平均差 $\mu_{\alpha_{\text{out}}} - \mu_{\alpha_{\text{in}}}$ を部分プーリングで推定し，その事後分布を報告する．

**FR-5.5**: 収束診断が以下を満たさない場合はエラーとする．

| 指標 | 基準 |
|---|---|
| $\hat{R}$ | $< 1.01$ |
| $\mathrm{ESS}_{\text{bulk}}$ | $> 400$ |
| divergent transitions | 全体の $0.1\%$ 未満 |

サンプリング設定：4 chains × 2000 draws（warmup 1000），`target_accept_prob=0.9`．

### 5.5 モデル比較

**FR-5.6**: PSIS-LOO で比較する．Pareto $\hat{k}$ 診断を必ず報告する．

**FR-5.7**: **CV の単位は参加者（leave-one-subject-out）とする．** 試行はパラメータ条件付きでも独立でなく $b_t$ / $V_t$ を通じて逐次依存しているため，leave-one-trial-out は不正である．

### 5.6 パラメータリカバリ

**FR-5.8**: 事前分布からパラメータをサンプル $\to$ 48試行 × 54名のデータを生成 $\to$ 推定 $\to$ 真値と比較．散布図を出力する（相関係数のみで判断しない）．

**FR-5.9**: 最優先の検証項目は群レベルの `alpha_out - alpha_in` の回復である．真値–推定値の相関と，真値が事後95%区間に入る割合（カバレッジ）を報告する．**ここが通らなければ以降の解析は成立しない．**

---

## 6. 実装

### 6.1 依存パッケージ

| パッケージ | バージョン | 用途 |
|---|---|---|
| `python` | $\ge$ 3.11 | |
| `numpy` | $\ge$ 2.0 | 配列 |
| `scipy` | $\ge$ 1.14 | `special.i0e`, `special.logsumexp` |
| `pandas` | $\ge$ 2.2 | tidy データ |
| `jax` | $\ge$ 0.4.35 | `lax.scan`，自動微分 |
| `numpyro` | $\ge$ 0.15 | NUTS，階層モデル |
| `arviz` | $\ge$ 0.20 | 診断，PSIS-LOO |
| `matplotlib` | $\ge$ 3.9 | 図 |
| `h5netcdf`, `h5py` | $\ge$ 1.8 / $\ge$ 3.16 | `trace.nc` の書き出し |
| `pytest` | $\ge$ 8.0 | テスト |

NumPyro を選ぶ理由：逐次スキャン尤度を `lax.scan` で直接書け，Stan より実装が短く，`arviz` と直結する．

**FR-6.0**: パッケージ初期化時に `jax.config.update("jax_enable_x64", True)` を実行すること．信念は 360 点グリッド上で 48 回逐次更新されるため，JAX 既定の float32 では AC-1 / AC-5 の許容誤差（$10^{-10}$ / $10^{-8}$）を満たせない．この設定は最初の JAX 配列が作られる前に行う必要がある．

**FR-6.0b**: 二等分線上の格子点の扱いに注意すること．刺激角・報告角・格子がすべて整数度なので，「どちらの方向が近いか」の比較は**整数度の巡回距離**で行う．浮動小数点の `atan2` で比較するとタイが数値誤差で解け，(a) 一様信念で $q$ が厳密に $0.5$ にならない，(b) 全角度を回転したときに対数尤度が不変にならない（AC-5 が落ちる）．タイの格子点は $q$ の計算で重み $0.5$ を割り当てる．

### 6.2 モジュール構成

```
src/dualrdk/pomdp/
  __init__.py        # jax_enable_x64 の設定（FR-6.0）
  circular.py        # wrap, circ_dist, log_vonmises (i0e ベース)
  belief.py          # グリッド信念, 更新, belief_to_q, カウンタ閉形式
  agent_reward.py    # R_hat（ordinal / smooth）．環境の R と関数を共用しない
  task_reward.py     # 環境の報酬関数 R．検証とシミュレーションにのみ使う
  models.py          # M0, M1, M1z, M2, M2z, M3, M3z の d_t 生成規則
  policy.py          # 2峰混合方策 + lapse + 選択のみ対数尤度
  likelihood.py      # lax.scan による対数尤度
  data.py            # concat_list -> tidy -> モデル入力配列，検証
  simulate.py        # モデルからのデータ生成（刺激と zone は実データから借りる）
  perceptual.py      # 段階1：kappa / lam の事前推定（FR-5.10）
  fit.py             # NumPyro 階層モデル + NUTS / 参加者ごとの MAP
  evaluate.py        # 共通評価指標（一致率・選択のみ対数尤度・較正・学習曲線）
  subgroup.py        # モデル適合による参加者層別（副次解析．FR-7.1）
  recovery.py        # パラメータリカバリ
  compare.py         # PSIS-LOO (leave-one-subject-out), 共通表, 事前チェック

tests/
  conftest.py        # src をパスに追加
  pomdp/
    test_circular.py
    test_belief.py
    test_policy.py
    test_likelihood.py
    test_m1.py       # M1 系, 色バイアス, 2段階推定
    test_evaluate.py # 共通評価指標
```

**FR-6.2b**: PSIS-LOO の観測次元名は変数名（`subject`）と別にすること（例：`dims={"subject": ["subject_id"]}`）．同名にすると arviz が chain / draw まで観測次元として畳み込み，`n_data_points` が `chains * draws * n_subj` になる．

### 6.3 実行手順

すべて **リポジトリルート**（`dual-rdk/`）を作業ディレクトリとして実行する．

```bash
# 0a. 環境構築（src レイアウトなので editable install が必要）
pip install -r requirements.txt
pip install -e .

# 0b. tidy テーブルの生成
#     参加者除外（4基準・26名 -> N=54）と OOZ ラベルは既存ローダが適用済み
python -m dualrdk.pomdp.data --data-dir data/raw/online --out outputs/pomdp/trials.parquet

# 1. テスト（§6.4 の全項目が通ること）
pytest tests -q

# 2. フィット前のモデル非依存チェック（FR-3.2 / AC-2）
python -m dualrdk.pomdp.compare --precheck --input outputs/pomdp/trials.parquet

# 3. パラメータリカバリ（FR-5.9．通らなければ以降に進まない）
python -m dualrdk.pomdp.recovery --model M2z --input outputs/pomdp/trials.parquet --n-sim 100

# 4. 段階1：知覚パラメータの事前推定（FR-5.10）
python -m dualrdk.pomdp.perceptual --input outputs/pomdp/trials.parquet \
    --out outputs/pomdp/perceptual.csv

# 5. 主モデルの推定（kappa / lam は固定して 7 -> 5 パラメータ）
python -m dualrdk.pomdp.fit --model M2z --input outputs/pomdp/trials.parquet \
    --fix-perceptual outputs/pomdp/perceptual.csv \
    --chains 4 --draws 2000 --warmup 1000 --out outputs/pomdp/fit/M2z

# 6. 全モデルの推定と，同じ族の中でのモデル比較
for m in M0 M1 M1z M2 M2z M3 M3z; do
  python -m dualrdk.pomdp.fit --model $m --input outputs/pomdp/trials.parquet \
      --out outputs/pomdp/fit/$m
done
# ELPD は観測空間が同じ族の中だけで比べる
python -m dualrdk.pomdp.compare --models M0,M2,M2z,M3,M3z --out outputs/pomdp/comparison
python -m dualrdk.pomdp.compare --models M1,M1z --out outputs/pomdp/comparison_binary

# 7. 共通指標による評価（推定法・モデル族に非依存）
for d in outputs/pomdp/fit/*; do
  python -m dualrdk.pomdp.evaluate --fit $d --input outputs/pomdp/trials.parquet
done

# 8. MAP との比較（既存の2値 MAP 解析との接続．FR-5.3b）
for m in M1z M2z; do
  python -m dualrdk.pomdp.fit --model $m --method map \
      --input outputs/pomdp/trials.parquet --out outputs/pomdp/fit_map/$m
done

# 9. 分割検証（FR-5.3c．MAP と階層ベイズを比べるには必須）
for m in M1z M2z; do
  python -m dualrdk.pomdp.fit --model $m --method map --train-trials 24 \
      --input outputs/pomdp/trials.parquet --out outputs/pomdp/split_map/$m
  python -m dualrdk.pomdp.fit --model $m --train-trials 24 \
      --input outputs/pomdp/trials.parquet --out outputs/pomdp/split_nuts/$m
done
python -m dualrdk.pomdp.compare --input outputs/pomdp/trials.parquet \
    --out outputs/pomdp/comparison_split \
    --common "map:M1z=outputs/pomdp/split_map/M1z,nuts:M1z=outputs/pomdp/split_nuts/M1z,\
map:M2z=outputs/pomdp/split_map/M2z,nuts:M2z=outputs/pomdp/split_nuts/M2z"

# 10. R_hat 感度分析（FR-4.3 / AC-8）
python -m dualrdk.pomdp.fit --model M3z --agent-reward smooth \
    --input outputs/pomdp/trials.parquet --out outputs/pomdp/fit/M3z_smooth
```

**FR-6.3a**: 中間ファイル（tidy テーブル）は生成物なので `outputs/` に置く．`data/` は読み取り専用の入力データ領域である（README のリポジトリ規約）．`--input` を省いて `--data-dir data/raw/online` を渡せば生データから直接読むこともできる．

**FR-6.3b**: 参加者除外は4基準・計26名で N=54．うち rt_cv 基準（7名）は標本内の相対位置で決まるため `config.EXCLUDED_SUBJECTS` に凍結してある．凍結値と再計算値の一致は `tests/test_exclusions.py` が検証する．

### 6.4 単体テスト

| テスト | 期待 |
|---|---|
| `circ_dist(0, 359)` | $1°$ |
| 一様信念 $\to q$ | $0.5$（誤差 $10^{-10}$） |
| $\psi$ に集中した信念 $\to q$ | `target_group` と整合 |
| 信念更新の正規化 | `logsumexp(log_b) == 0`（誤差 $10^{-10}$） |
| 逐次フィルタ vs カウンタ閉形式 | 全試行で一致（誤差 $10^{-10}$） |
| `log_vonmises` at $\kappa = 1000$ | 有限値を返す |
| 方策の正規化 | $\int \pi_t(a)\,da = 1$（誤差 $10^{-6}$） |
| 回転同変性 | 全角度を $\delta$ 回転しても対数尤度が不変（誤差 $10^{-8}$） |
| $\alpha_{\text{in}} = \alpha_{\text{out}}$ の M2z | M2 と対数尤度が一致 |

### 6.5 出力

```
outputs/pomdp/
  perceptual.csv                       # 段階1の kappa / lam（参加者ごと）
  fit/<model>/trace.nc                 # ArviZ InferenceData（NUTS のみ）
  fit/<model>/diagnostics.json         # R_hat, ESS, divergences（NUTS のみ）
  fit/<model>/params.csv               # 参加者ごとの点推定（MAP のみ）
  fit/<model>/fit_config.json          # 推定法・パラメータ数・train_trials
  fit/<model>/latent.csv               # 参加者 x 試行の d_t, q_t, p_W_t
  fit/<model>/evaluation.json          # 共通指標
  fit/<model>/calibration.csv          # p_W ビン別の予測 vs 実測
  fit/<model>/learning_curve.{csv,png} # ブロック別の観測 vs 予測正解率
  fit/<model>/per_subject.csv          # 参加者ごとの一致率
  recovery/<model>/scatter.png
  recovery/<model>/summary.json        # 相関・カバレッジ
  comparison/loo.csv                   # elpd_loo, se, pareto_k（同一族内のみ）
  comparison/common.csv                # 族・推定法をまたぐ共通指標
  comparison/precheck_bimodality.png
  alpha_contrast.json                  # mu_alpha_out - mu_alpha_in の事後要約
```

**FR-6.1**: `latent.csv`（潜在変数の軌跡）を必ず出力する．個人差解析の入力になり，`evaluate.py` の唯一の入力でもある（だから推定法によらず同じ形で書く）．

**FR-6.2**: `alpha_contrast.json` に群レベル差の事後平均・95%区間・$\Pr(\Delta > 0)$ を出力する．これが本解析の主結果である．

**FR-6.5（共通評価指標）**: モデル族・推定法をまたぐ比較は，次の2つの指標だけで行う．

$$\text{一致率} = \frac{1}{T}\sum_t \mathbb{1}\big[(p^W_t > 0.5) = c_t\big], \qquad \log\mathcal{L}_{\text{choice}} = \sum_t \log P(c_t)$$

$c_t$ は参加者の実際の選択（応答角がどちらの雲に近いか）．$\log\mathcal{L}_{\text{choice}}$ は**確率質量**なので族をまたいで足し合わせられる．

**「ターゲット選択確率の平均」を評価指標にしてはならない．** 参加者が実際にターゲットを選んだ割合（本データで 0.617）を上回っていても，それは過大予測を意味するだけで妥当性の証拠にならない．`evaluation.json` は `predicted_correct_rate` と `participant_correct_rate` を必ず並べて出力し，この2つの乖離を可視化する．

**FR-6.6**: `evaluate.py` は較正表と学習曲線を必ず出力する．一致率という単一のスカラーでは「どの確率帯で・いつずれているか」が見えないため（FR-5.8 と同じ理由）．

---

## 6.6 副次解析：モデル適合による参加者層別

**FR-7.1**: `subgroup.py` に実装する．M0（学習なし）と M2z（学習あり）の参加者ごとの当てはまりを比べ，学習モデルが勝つ群と勝たない群に分けて，各群で $\Delta\alpha$ を推定する．

**FR-7.2（必須の但し書き）**: **これは選択と推定に同じデータを使う手続きである（double dipping）**．層別に使う統計量（M2z $-$ M0 の当てはまり差）は「その参加者が学習したか」をほぼそのまま測っており（実データで正解率との相関 $0.826$），学習量は $\alpha$ と直結する．したがって learner 群の $\alpha$ 分布は上に切り詰められ，$\Delta\alpha$ の推定も偏りうる．**主解析にしてはならない．**

**FR-7.3**: 層別の指標は既定で選択のみ対数尤度とする．一致率（argmax）は予測の確信度を捨てるため，僅差の参加者で不安定になる．

**FR-7.4**: `--random-split SEED` による対照条件を必ず併記する．群サイズを保ったままランダムに分けた場合の群差と比べ，実際の層別による群差がそれを上回るかを見る．1本のランダム分割は検定ではないが，小標本の推定誤差が生む見かけの群差の目安になる．

**FR-7.5**: 出力は既存結果と別のツリー（`outputs/pomdp/subgroup/<tag>/`）に書き，主解析の結果を上書きしない．

```bash
python -m dualrdk.pomdp.subgroup \
    --baseline-fit outputs/pomdp/fit/M0 --learning-fit outputs/pomdp/fit/M2z \
    --input outputs/pomdp/trials.parquet \
    --fix-perceptual outputs/pomdp/perceptual.csv \
    --out outputs/pomdp/subgroup/M2z_vs_M0

# 対照条件（同じ群サイズのランダム分割）
python -m dualrdk.pomdp.subgroup ... --random-split 0 \
    --out outputs/pomdp/subgroup/random_control_0
```

### 6.7 参加者ごと MAP による群コントラストの較正（FR-7.6）

**FR-7.6**: 参加者ごと MAP（部分プーリングをしない推定）で得た $\Delta\alpha$ の群レベル統計量は，**円環置換による帰無分布に照らしてからでなければ報告してはならない**．$t$ 検定・Wilcoxon・符号検定の $p$ 値を，帰無が 0 中心であるとの前提のまま用いてはならない．

根拠は §R-1 の実測である．注意状態ごとの試行数が非対称（$n_{\text{out}}$ 中央値 12 対 $n_{\text{in}}$ 36，54名全員が同じ向き）なため，データが少ない側が事前分布の中心に強く引かれ，**真の差がゼロでも $\Delta\alpha > 0$ が系統的に出る**（帰無平均 $+0.031$，帰無分布の95%が正側）．符号検定の分布非依存性はこの偏りを防がない．

`zone_permutation.py` は有効試行の zone ラベルを円環シフトし，同一の推定手続きで帰無分布を作る．**観測側も同じ `--n-starts` で推定し直すこと**（多点スタート数は推定量の定義の一部であり，観測と帰無で異なる設定を突き合わせると最適化精度の差が効果と誤読される）．$p$ の下限は $1/(B+1)$ なので，裾での判定が必要なら $B \geq 1000$ とする．

```bash
python -m dualrdk.pomdp.zone_permutation --n-perm 3000 --n-jobs 8 \
    --rung R4_pomdp_data --out outputs/pomdp/zone_permutation

python -m dualrdk.pomdp.zone_permutation_stats \
    --input outputs/pomdp/zone_permutation
```

この検定が消せる対抗仮説は「試行数の非対称に由来する推定量の偏り」の1つだけである．棄却できても真の学習率差の証明にはならない（$\beta$ や $\lambda$ のゾーン依存は別に潰す必要がある）．

---

## 7. 受け入れ基準

| # | 基準 |
|---|---|
| AC-1 | §6.4 の全単体テストが通る |
| AC-2 | 事前チェック：$a_t$ の相対角ヒストグラムが2峰で，中間に有意な質量がない |
| AC-3 | リカバリ：群レベル `alpha_out - alpha_in` の真値–推定値相関 $> 0.7$，95%区間カバレッジ $> 0.9$ |
| AC-4 | リカバリ：$\beta, \kappa$ の相関 $> 0.7$，$\lambda, \epsilon$ で $> 0.5$ |
| AC-5 | リカバリ：生成パラメータに無い偽相関が $\lvert r \rvert > 0.4$ を超えない |
| AC-6 | 収束：$\hat{R} < 1.01$，$\mathrm{ESS}_{\text{bulk}} > 400$，divergences $< 0.1\%$ |
| AC-7 | PSIS-LOO：$\hat{k} > 0.7$ の観測が $1\%$ 未満 |
| AC-8 | $\hat{R}$ を `smooth` に変えても M3 系のモデル順位が変わらない |
| AC-9 | 全モデルで一致率が偶然（0.50）と「常に高報酬側」ベースライン（0.617）の両方を上回る |
| AC-10 | `predicted_correct_rate` と `participant_correct_rate` の差が 0.05 未満（過大予測が無い） |
| AC-11 | 較正表のどのビンでも $\lvert z \rvert < 3$ |
| AC-12 | 分割検証（`--train-trials 24`）で，選ばれたモデルがホールドアウトの選択のみ対数尤度でも最良 |
| AC-13 | 参加者ごと MAP の $\Delta\alpha$ を報告する場合，円環置換の帰無分布（$B \geq 1000$）を併記し，帰無平均からの超過で判定している（FR-7.6） |

---

## 8. 既知の制約

| # | 内容 | 対応 |
|---|---|---|
| R-1 | 群レベル `alpha_out - alpha_in` の実測分解能は低い（下記の数値を参照） | 階層ベイズ必須（FR-5.3），群レベル差のみ推定（FR-5.4），AC-3 で事前に検証．結果は有意/非有意ではなく事後平均・95%区間・$P(\Delta>0)$ と検出限界を併記して報告する |
| R-2 | 意識的学習による除外が $\alpha$ の上端を切っている可能性（範囲制限） | 除外前データでも推定し $\alpha$ の頑健性を確認，limitation に記載 |
| R-3 | 除外基準はフィードバック下の反転追随を「意識的」と解釈しており，速い暗黙的反転学習と区別できない | 「暗黙的学習は非柔軟である」という理論的前提を明示して記載 |
| R-4 | M2 と M3 が実験設計上分離しない可能性 | LOO で分離しない場合はその事実を報告し，$\alpha$ の解釈を M2 系に限定 |
| R-5 | 2峰性は方策の仮定であり，ヘッジングを検出できない | AC-2 のモデル非依存チェックで事前に確認 |
| R-6 | $\beta$ と $\lambda$ は行動のばらつきを説明する役割が重複しうる | AC-5 で監視．重複が深刻なら $\lambda$ を固定 |
| R-7 | 報酬が決定論的なため $\epsilon$ なしでフィルタが崩壊する | FR-2.2（汚染混合），FR-2.3（例外送出） |

### R-1 の数値化（M2z, n_sim = 30 の実測）

`outputs/pomdp/recovery/M2z/records.csv`（54名 × 48試行の design-matched シミュレーション，2 chains × draws 1000 / warmup 1000）から測った，群レベル `contrast_alpha` = $\alpha_{\text{out}} - \alpha_{\text{in}}$ の回復性能：

| 量 | 実測値 |
|---|---|
| 真値–推定値の相関 | 0.865（AC-3 の 0.7 を満たす） |
| 95%区間カバレッジ | 0.933（AC-3 の 0.9 を満たす。名目 0.95 と整合） |
| 縮小の傾き（真値に対する推定値の回帰係数） | 0.793 |
| 測定ノイズ SD（回帰残差 SD） | 0.138 |
| 事後 95% 区間の平均幅 | 0.605（＝ 片側 0.303） |

点推定のばらつきだけを見た下限は $1.96 \times 0.138 = 0.27$ である．ただし実際に「95%区間が 0 を除外する」ためには，縮小（傾き 0.793）を受けた推定値が片側幅 0.303 を超える必要がある．$\hat{\Delta} \sim N(0.793\Delta,\ 0.138^2)$ として検出力を計算すると

$$\text{power}(\Delta) = \Phi\!\left(\frac{0.793\,\lvert\Delta\rvert - 0.303}{0.138}\right)$$

| 検出力 | 必要な真の $\lvert\Delta\alpha\rvert$ |
|---|---|
| 50% | 0.38 |
| 80% | 0.53 |

30本の実測でも整合する（$\lvert\Delta\rvert \ge 0.3$ で 7/14 が 0 を除外，$\ge 0.4$ で 3/4）．一方，符号の一致率は $\lvert\Delta\rvert > 0.10$ で 0.92，$\lvert\Delta\rvert \le 0.10$ では 0.67（偶然と大差ない）．

したがって本設計が検出できるのは **$\lvert\Delta\alpha\rvert \gtrsim 0.4$ 程度の大きな差に限られる**．主因はパラメータ数ではなく out-of-zone 試行の少なさである（全試行の約26%，中央値12試行/名）．実測でも $\alpha_{\text{in}}$ の相関 0.941 に対し $\alpha_{\text{out}}$ は 0.777 と劣る．区間が 0 をまたいだ場合に「差が無い」と結論してはならない．

### R-1 に対して試した対策と，その結果

解像度を上げるために次を実施した．**いずれも M2z 系では改善しなかった．**

| 対策 | 根拠 | 結果 |
|---|---|---|
| パラメータ削減（$\kappa,\lambda$ を段階1で固定．$7\to5$） | FR-5.10 | 区間は**広がった**（幅 $0.605 \to 0.810$）．$c_{\text{black}}$ が選択の系統成分を吸収した分，$\alpha$ に残る情報が減ったため |
| 学習が成立した参加者への限定（FR-7.1） | 非学習者が希釈しているとの仮説 | 効果なし（下表） |
| M1 系（2値状態・2値行動）への変更 | 観測が単純で自由度が低い | **区間幅が 1/3 に**（下表） |

#### 層別の実測（`outputs/pomdp/subgroup/`）

M0 と M2z の参加者ごとの当てはまりで層別．選択のみ対数尤度を指標とした．

| 群 | $n$ | 正解率(中央値) | $\Delta\alpha$ | 95%区間 | 幅 |
|---|---|---|---|---|---|
| 全体 | 54 | 0.617 | $-0.103$ | $[-0.548, +0.263]$ | 0.810 |
| learner | 37 | 0.750 | $-0.102$ | $[-0.531, +0.254]$ | 0.785 |
| nonlearner | 17 | 0.354 | $+0.057$ | $[-0.249, +0.445]$ | 0.694 |

群間差 $-0.160$，95%区間 $[-0.704, +0.308]$，$\Pr(\text{diff}>0) = 0.278$．

対照条件（群サイズを保ったランダム分割，`--random-split 0`，FR-7.4）では群間差 $+0.009$，95%区間 $[-0.632, +0.637]$．**観測された群間差 $-0.160$ は対照区間の内側にあり，無作為分割の揺らぎと区別できない．**

区間幅も対照と同程度（learner 37名で 0.785 対 0.675）で，**層別した参加者を集めてもランダムな同数より精度が上がらない**．正解率の分離自体は成功している（0.750 対 0.354，対照では 0.708 対 0.500）ので，**分類は効いているが $\Delta\alpha$ の推定には効いていない**．

非学習者 17名を除いても推定値・区間幅がほぼ動かないのは，彼らの $\alpha$ が平坦な尤度のもとで階層事前に縮んでおり，群平均を押し引きしていなかったためである．「非学習者が希釈している」という仮説は否定された．

注：ランダム分割は1本のみで帰無分布ではない．厳密には20〜50本の並べ替えが必要だが，観測値が対照区間の中央付近にあるため結論は変わらないと判断する．

#### モデル族による違い

| モデル | パラメータ | $\Delta\alpha$ | 95%区間 | 幅 |
|---|---|---|---|---|
| M2z（$\kappa,\lambda$ 固定，$c_{\text{black}}$ あり） | 5 | $-0.103$ | $[-0.548, +0.263]$ | 0.810 |
| **M1z** | 4 | $+0.002$ | $[-0.154, +0.137]$ | **0.291** |

$\Delta\alpha$ を測る道具としては M1z が圧倒的に優れる．ただし M1z は「正しい2分割を最初から誤差なく持っている」という仮定（§3.3）に依存する．主解析に採る場合はこの仮定を明示し，M2z の結果を併記すること．**なお M1z の検出限界は未測定である**（上記の 0.38 / 0.53 は M2z のリカバリ値）．M1z を主解析にするなら M1z でリカバリを走らせ直す必要がある．

上表は**階層ベイズ**の結果である．同じ M1z でも参加者ごと MAP で推定すると $\Delta\alpha$ は系統的に正へ偏る（次節）．この表の M1z の優位を MAP の当てはめから読み取ってはならない．

#### 参加者ごと MAP の $\Delta\alpha$ は人工物である（円環置換検定）

M1z を参加者ごと MAP で推定すると $\Delta\alpha = +0.041$，54名中40名が正，符号検定 $p = 0.0005$ となり，階層ベイズの $+0.002$ と矛盾するように見えた．これは**推定量の偏り**であって効果ではない．

**交絡**．注意状態ごとの試行数が大きく非対称である．

```
OOZ 試行数 (n_out)   中央値 12   範囲  2-19
非OOZ      (n_in)    中央値 36   範囲 28-46
n_out < n_in の参加者                54 / 54
```

$\alpha_{\text{out}}$ は $\alpha_{\text{in}}$ の約 1/3 の試行数で推定される．データが少ない側は事前分布の中心に強く引かれるため，**真の差がゼロでも** $\alpha_{\text{out}}$ だけが持ち上がる．実際，事前の中心が異なる2つの設定で同じ配置が出る．

| 設定 | $\alpha_{\text{in}}$ | $\alpha_{\text{out}}$ | 事前の中心 |
|---|---|---|---|
| R1（Beta(2,2)） | 0.4315 | 0.4768 | 0.5 |
| R4（logit-N$(-1, 1.2)$） | 0.2053 | 0.2462 | 0.269 |

どちらも $\alpha_{\text{out}}$ が $\alpha_{\text{in}}$ と事前中心のちょうど間にある．**54名全員が同じ向きの非対称を持つため，「OOZ で学習率が高い」と「OOZ は試行が少ないのでシュリンケージが強い」は観測データ上で完全に交絡しており，回帰的な統制では分離できない．**

**方法**（`zone_permutation.py`）．各参加者の有効試行の zone ラベル列を円環状にずらし，同一の推定手続きを通す．3000回．

| 保たれる | 壊れる |
|---|---|
| 実際の選択・報酬・刺激，各参加者の真の学習率 | どの試行が実際に OOZ だったか |
| **試行数の非対称（12 対 36）**，OOZ のラン長構造 | |
| 推定手続き（事前・座標系・多点スタート数） | |

有効試行の並びの中だけで回す（無効試行を跨ぐと有効試行中の OOZ 個数が変わり，保存すべき非対称が崩れる）．i.i.d. のシャッフルではなく円環シフトを使うのは，OOZ ラベルが HMM 由来でまとまって出るためラン長構造も保つ必要があるから．

**結果**（`outputs/pomdp/zone_permutation/`）．

| 統計量 | 観測 | 帰無平均 | 帰無95%区間 | 片側 $p$ |
|---|---|---|---|---|
| $\overline{\Delta\alpha}$ | $+0.0409$ | $+0.0307$ | $[+0.0007, +0.0623]$ | 0.262 |
| $\Delta\alpha$ の中央値 | $+0.0589$ | $+0.0414$ | $[+0.0162, +0.0670]$ | 0.095 |
| 正の人数 | 40 / 54 | 36.2 / 54 | $[31, 42]$ | 0.126 |
| $t$ | 2.183 | 1.791 | $[0.038, 3.752]$ | 0.333 |
| $\mathrm{SD}(\Delta\alpha)$（個人差） | 0.1377 | 0.1274 | $[0.1013, 0.1552]$ | 0.236 |

**決め手は $p$ 値ではなく帰無分布の位置である．2.5%点が $+0.0007$，すなわち帰無分布の95%がまるごと正側にある．** ゾーンラベルを壊しても推定量はほぼ必ず $\Delta\alpha > 0$ を返す．$\alpha$ の水準も再現される（観測 $\alpha_{\text{in}} = 0.2053$ / $\alpha_{\text{out}} = 0.2462$ に対し帰無 $0.2098$ / $0.2405$）．観測 $+0.0409$ のうち **75%（$+0.0307$）が非対称だけで説明**され，残差 $+0.0102$ は帰無 SD $0.0159$ の 0.64 倍でノイズの中にある．

**当初の $p$ 値は計算ではなく帰無を誤っていた．**

| 統計量 | 素朴な帰無 | 正しい帰無 | 観測 | 素朴な $p$ | 正しい $p$ |
|---|---|---|---|---|---|
| 正の人数 | 27 / 54 | **36.2 / 54** | 40 / 54 | 0.0005 | **0.126** |
| $\overline{\Delta\alpha}$ | 0 | $+0.0307$ | $+0.0409$ | 0.0335 | **0.262** |

$t$ 検定・Wilcoxon・符号検定はいずれも「真の差がゼロなら $\Delta\alpha$ の推定値は 0 を中心に散らばる」を前提とする．符号検定の頑健性は分布形に対するものであって，**推定量の偏りには何の防御にもならない**．

**個人差についても同じ結論である．** $\Delta\alpha$ の参加者間 SD 0.1377 は帰無でもほぼ全部再現される（$p = 0.236$）．観測された散らばりは真の個人差ではなく，参加者ごとに $n_{\text{out}}$（2〜19）と尤度の形が違うことによる推定ノイズである．

小シフトの引きでは置換後のラベルが実際のラベルに似るため帰無が観測側へ寄る（検定が保守的になる）懸念があったが，実測では無関係だった：$\rho(\text{mean\_shift}, \overline{\Delta\alpha}) = -0.020$（$p = 0.274$），シフト小群 $+0.0308$ / 大群 $+0.0306$．

**この検定が言っていないこと．** $\Delta\alpha = 0$ の証明ではない．帰無の95パーセンタイルが $+0.0572$ なので，有意になるには推定値がそこを超える必要がある．バイアス $+0.031$ の上に $+0.026$ 以上を積む必要があり，しかもシュリンケージは真の効果自体も縮めるため，**真の $\Delta\alpha$ は 0.03 よりかなり大きくなければ見えない**（正確な検出限界にはパラメトリックなシミュレーションが要る）．また消せた対抗仮説はこれ1つだけで，OOZ 試行で $\beta$ や $\lambda$ も異なりモデルがそれを $\alpha$ に押し込んでいる可能性は残る．

**帰結．**

1. **参加者ごと MAP の「有意な $\Delta\alpha$」は撤回する．** 人工物で説明がつく範囲にある．
2. **階層ベイズを主解析に選んだ判断は支持された．** 不均等な試行数を部分プーリングで扱うのが本来の対処であり，$+0.002$ は MAP の $+0.041$ より信頼できる．ただし階層版が同じ偏りから自由であることは未検証である（同じ置換検定を階層推定に適用すれば確認できるが計算コストが高い）．
3. **推定量の不一致が残る．** `fit.py` の `contrast_alpha` は $\sigma(\mu_{\text{out}}) - \sigma(\mu_{\text{in}})$（群平均を変換してから引いた量）であり，$N^{-1}\sum_i (\alpha_{\text{out},i} - \alpha_{\text{in},i})$（個人ごとに引いてから平均した量）とは $\sigma$ の非線形性の分だけ一致しない．両者を比べるなら階層事後から個人ごとの $\Delta\alpha$ を取って平均する必要がある．

#### 結論

$\Delta\alpha \approx 0$ を支持する独立な経路が5つある．

```
1. M2z の事後               -0.103 [-0.548, +0.263]
2. M1z の事後               +0.002 [-0.154, +0.137]
3. LOO で M2 対 M2z          +0.3 (0.11 SE) = 区別不能
4. 層別後の learner 群       -0.102 [-0.531, +0.254]
5. 層別 vs ランダム分割       差 -0.160，対照 +0.009 = 区別不能
6. M1z 参加者ごと MAP        +0.041，置換帰無 +0.031 (p=0.26) = 人工物
```

ただし 1・3・4・5 は検出限界 0.4 のもとでの結果であり，「差が無い」ではなく「検出できる大きさの差が無い」である．2（M1z）のみ区間が $\pm 0.15$ に収まっており，実質的な情報を持つ．

経路6は他と性質が違う．これは $\Delta\alpha \approx 0$ の**独立な証拠ではなく**，$\Delta\alpha \neq 0$ に見えた唯一の結果を取り下げるものである．同時に，MAP と階層ベイズの食い違い（$+0.041$ 対 $+0.002$）という未解決点も解消する．**したがって現時点で $\Delta\alpha \neq 0$ を支持する経路は一つも残っていない．**
