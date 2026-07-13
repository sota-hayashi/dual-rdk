# HMM妥当性検証 実装仕様書
## Recovery Analysis + External Validation

---

## 1. 概要

### 目的
Gaussian HMMによる状態推定の妥当性を2つの独立した観点から評価する。

```
観点A：試行数の妥当性（Recovery Analysis）
  「48試行という制約でパラメータが正しく推定できるか」

観点B：モデルの妥当性（External Validation）
  「推定された状態系列が実際の注意状態を反映しているか」
```

### 2つの評価の関係

```
Recovery Analysis
  └── モデルが正しいと仮定した上で
      試行数が十分かを確認
  └── 結果が悪ければ：推定値を個人差比較に使えない

External Validation
  └── 試行数を所与として
      推定状態系列が外部指標（log_rt）と関係するかを確認
  └── 結果が良ければ：モデルが現実を反映していると判断

2つを組み合わせることで実用的な妥当性評価が可能
```

---

## 2. 観点A：Recovery Analysis

### 2-1. 目的

既知のパラメータからデータを生成し、そのデータから元のパラメータが
復元できるかを確認することで、**48試行という条件でHMMが機能するか**を検証する。

### 2-2. 入力

```
true_params_list : list[dict]
  # 検証する「真のパラメータ」のパターンリスト
  # 以下の3パターンを検証する

  パターンA（disengagedが長く持続する場合）:
    means    = [10.0, 35.0]       # μ_engaged=10°, μ_disengaged=35°
    covars   = [50.0, 300.0]      # それぞれの分散
    transmat = [[0.95, 0.05],     # E→E=0.95（dwell time=20試行）
                [0.10, 0.90]]     # D→D=0.90（dwell time=10試行）
    startprob= [0.8, 0.2]

  パターンB（中程度の持続）:
    means    = [10.0, 30.0]
    covars   = [50.0, 300.0]
    transmat = [[0.90, 0.10],     # E→E=0.90（dwell time=10試行）
                [0.20, 0.80]]     # D→D=0.80（dwell time=5試行）
    startprob= [0.8, 0.2]

  パターンC（グローバルモデルに近い実際の結果）:
    means    = [9.5,  28.5]       # グローバルモデルの推定値
    covars   = [47.6, 325.5]
    transmat = [[0.866, 0.134],   # グローバルモデルの推定値
                [0.304, 0.696]]
    startprob= [0.8, 0.2]

n_trials : int = 48               # シミュレートする試行数（実験と同じ）
n_simulations : int = 100         # 各パターンで繰り返すシミュレーション回数
```

### 2-3. 出力

```python
recovery_results : dict = {
    "pattern_A": {
        # パラメータの復元精度
        "mean_engaged": {
            "true": 10.0,
            "estimated_mean": float,   # 100回の推定値の平均
            "estimated_std":  float,   # 100回の推定値の標準偏差
            "bias":           float,   # estimated_mean - true
        },
        "mean_disengaged": { ... },    # 同上
        "transmat_EE": { ... },        # E→Eの復元精度
        "transmat_DD": { ... },        # D→Dの復元精度

        # 状態系列の復元精度
        "state_accuracy": {
            "mean": float,             # 真の状態系列との一致率（100回の平均）
            "std":  float,
        },
    },
    "pattern_B": { ... },
    "pattern_C": { ... },
}
```

### 2-4. アルゴリズム

```
for pattern in [A, B, C]:
  estimated_params_list = []

  for sim in range(n_simulations=100):

    Step 1: 既知パラメータからデータを生成
      model_gen = GaussianHMM(パラメータをtrue_paramsに設定)
      X_sim, states_true = model_gen.sample(n_samples=48)
      # X_sim      : shape (48, 1)  シミュレートされたabs_angular_error
      # states_true: shape (48,)    真の状態系列

    Step 2: シミュレートデータにHMMを適用
      model_rec = GaussianHMM(グローバルfittingと同じ設定)
      model_rec.fit(X_sim)
      states_estimated = model_rec.predict(X_sim)

    Step 3: アライメント
      # 状態番号が入れ替わっている可能性があるため
      # μの大小で0=engaged, 1=disengagedに統一

    Step 4: パラメータと状態系列の比較を記録
      estimated_params_list.append({
          "mean_engaged":    model_rec.means_[0],
          "mean_disengaged": model_rec.means_[1],
          "transmat_EE":     model_rec.transmat_[0][0],
          "transmat_DD":     model_rec.transmat_[1][1],
          "state_accuracy":  accuracy(states_true, states_estimated),
      })

  → 100回分の統計（平均・標準偏差・bias）を計算して保存
```

### 2-5. 判断基準

```
判断基準1：パラメータの偏り（bias）
  |bias| < true_value × 0.10   → 許容範囲（10%以内の偏り）
  |bias| >= true_value × 0.10  → 系統的な過小/過大推定あり

判断基準2：パラメータのばらつき（std）
  std < true_value × 0.20      → 安定した推定（20%以内）
  std >= true_value × 0.20     → 不安定（試行数が不足）

判断基準3：状態系列の一致率
  state_accuracy >= 0.70       → 許容範囲
  state_accuracy < 0.70        → 状態系列の解釈が困難

→ 3つの基準をすべて満たすパターンのみ
  「48試行でHMMが機能する」と判断する
```

---

## 3. 観点B：External Validation

### 3-1. 目的

推定された状態系列が**実際の注意状態を反映しているか**を、
HMMの入力に使っていない外部指標（log_rt）との関係で検証する。

モデルの仮定が正しければ：
```
disengaged状態の試行 → 注意が途切れている
                     → 反応時間が長くなる or 変動が大きくなる
                     → log_rtが大きいはず
```

### 3-2. 入力

```
hmm_results   : list[dict]   # gaussian_hmm_spec.mdの出力結果
df_original   : DataFrame    # 元のDataFrame（log_rt列を含む）

  必須カラム：
    participant_id  : str
    trial           : int
    log_rt          : float    # log変換済みの反応時間
    abs_angular_error: float   # HMMの入力に使った変数
```

### 3-3. 出力

```python
validation_results : dict = {

    # --- 試行レベルの検証（within-participant）---
    "trial_level": {
        "log_rt_by_state": {
            "engaged_mean":    float,   # engaged試行のlog_rt平均（全参加者）
            "disengaged_mean": float,   # disengaged試行のlog_rt平均（全参加者）
            "effect_size":     float,   # Cohen's d
            "p_value":         float,   # 混合線形モデルまたはt検定
        },
        "local_cv_rt_by_state": {
            # 局所的なCV(rt)（前後3試行の移動SD/移動mean）
            # disengaged状態でCV(rt)が高いかを検証
            "engaged_mean":    float,
            "disengaged_mean": float,
            "effect_size":     float,
            "p_value":         float,
        },
    },

    # --- 参加者レベルの検証（between-participant）---
    "participant_level": {
        "disengaged_rate_vs_cv_rt": {
            # disengaged割合 と CV(rt)（試行全体）の相関
            # 研究会の結果との接続
            "r":       float,    # Pearsonの相関係数
            "p_value": float,
        },
        "disengaged_rate_vs_learning": {
            # disengaged割合 と target_choice_rate_diff の相関
            "r":       float,
            "p_value": float,
        },
    },

    # --- 状態遷移と反応時間の関係 ---
    "transition_level": {
        # disengagedに遷移した直後の試行でlog_rtが長いか
        "log_rt_after_transition_to_disengaged": {
            "mean":    float,
            "vs_baseline_p_value": float,
        },
        # engagedに復帰した直後の試行でlog_rtが短くなるか
        "log_rt_after_transition_to_engaged": {
            "mean":    float,
            "vs_baseline_p_value": float,
        },
    },
}
```

### 3-4. アルゴリズム

```
Step 1: 状態系列とlog_rtを試行レベルで結合
  df_merged = hmm_results（viterbi_states）と
              df_original（log_rt）を
              participant_id × trial でmerge

Step 2: 局所的なCV(rt)を計算
  window = 5（前後2試行）
  local_cv_rt = rolling(window).std() / rolling(window).mean()
  ※ NaNが生じる端の試行は解析から除外

Step 3: 試行レベルの検証
  engaged試行 と disengaged試行 で
  log_rt・local_cv_rtの差を検定
  （参加者の個人差を制御するため混合線形モデルを推奨）

Step 4: 参加者レベルの検証
  各参加者のdisengaged割合を計算
  → 研究会で使用したCV(rt)・学習指標との相関を計算

Step 5: 状態遷移レベルの検証
  viterbi_statesから遷移イベントを抽出
  （0→1: engagedからdisengagedへの遷移）
  （1→0: disengagedからengagedへの遷移）
  → 遷移直後の試行のlog_rtをベースラインと比較
```

### 3-5. 判断基準

```
試行レベル：
  disengaged状態でlog_rtが有意に長い → モデルが妥当
  効果量 Cohen's d > 0.3 が目安

参加者レベル：
  disengaged割合とCV(rt)の正の相関が有意
  → 研究会の結果と接続できる
  → HMMの状態系列が被験者間の変動指標と整合している

遷移レベル：
  状態遷移と反応時間の時間的な関係が確認できる
  → 状態が単なるクラスタリングではなく
    動的な変化を捉えていることの証拠
```

---

## 4. 2つの評価の統合判断

```
                Recovery Analysis    External Validation
                （試行数の妥当性）   （モデルの妥当性）
                ─────────────────────────────────────────
結果が両方OK  → HMMの結果を信頼して解析を進める

RecoveryのみNG → 推定の不安定性を明示した上で
                 External Validationの結果を報告する
                 （探索的な解析として位置づける）

ExternalのみNG → モデルの仮定が現実と合っていない
                 → 状態数の変更・入力変数の変更を検討

両方NG        → HMMの適用を断念し
                 変化点検出など別の手法を検討する
```

---

## 5. 依存ライブラリ

```
hmmlearn >= 0.3.0
numpy
pandas
scipy（t検定・相関）
statsmodels（混合線形モデル）
```
