# Gaussian HMM 実装仕様書
## dual-RDK課題における角度誤差からの隠れ状態推定

---

## 1. 概要

### 目的
dual-RDK課題において参加者が報告した角度の**絶対誤差**（連続値）から、試行ごとの隠れ状態（engaged / disengaged）の時系列を推定する。

### タスクの構造（背景）

```
各試行：
  白ドット群（方向A）と黒ドット群（方向B）が同時表示
  参加者は1つの方向を連続角度で報告する

試行の種類：
  Type W: 白がターゲット（白方向 = 高報酬）、黒がディストラクター
  Type B: 黒がターゲット（黒方向 = 高報酬）、白がディストラクター

angular_errorの計算（呼び出し元で計算済みであること）：
  target_AE     = response_angle - target_direction     （-180~180で正規化済み）
  distractor_AE = response_angle - distractor_direction （-180~180で正規化済み）

  |target_AE| < |distractor_AE| のとき
    → angular_error = target_AE    （ターゲット寄りの回答）
  それ以外のとき
    → angular_error = distractor_AE （ディストラクター寄りの回答）

HMMへの入力：
  abs_angular_error = |angular_error|   （0° 〜 180°の範囲）
  ※ 符号を捨てることで状態の分離を明確にする
```

### 採用モデル
- **hmmlearn** の `GaussianHMM`
- **2段階fitting**（Ashwood et al. 2022, Algorithm 1 に基づく）
  1. 全参加者データを結合した**グローバルfitting**
  2. グローバルパラメータを初期値とした**個人fitting**
- 状態系列の推定には**ビタビアルゴリズム**を使用

### 設定

| 項目 | 値 |
|---|---|
| 状態数 | **2**（engaged / disengaged） |
| 観測変数 | `abs_angular_error`（絶対角度誤差、0°〜180°） |
| セッション数 | 1参加者 = 1セッション |
| 参加者数 | 30人以上 |
| 1参加者あたりの最大試行数 | 48試行（NaN除外後） |

---

## 2. 入力データ

### DataFrameの構造

```
必須カラム：
  participant_id    : str or int  # 参加者ID
  trial             : int         # 試行番号（0-indexed、NaN除外済み）
  angular_error     : float       # 符号付き角度誤差（-180°〜180°）
                                  # 呼び出し元で計算済みであること

自動計算カラム（パイプライン内部で生成）：
  abs_angular_error : float       # |angular_error|（0°〜180°）
                                  # これがHMMへの入力となる

任意カラム（あれば状態解釈の補助に使える）：
  reward_points     : int         # 報酬の有無（1/0）
  rt                : float       # 反応時間（秒）
  target_group      : str         # "white" or "black"
```

### 具体例

```
participant_id  trial  angular_error  abs_angular_error  reward   rt
P01             0       -3.2           3.2                9       0.82
P01             1      +12.5          12.5                7       0.91
P01             2      +87.3          87.3                0       1.43
P01             3      -112.5         112.5               0       1.21
P01             4       -5.1           5.1                8       0.78
...
P02             0       +2.8           2.8                9       0.95
...
```

---

## 3. 出力

### 出力形式
参加者ごとの辞書を格納したリスト。

```python
results: list[dict] = [
    {
        # 識別情報
        "participant_id"    : str,
        "n_trials"          : int,          # 有効試行数

        # ビタビアルゴリズムの出力
        "viterbi_states"    : np.ndarray,   # shape: (n_trials,)
                                            # 各試行の最尤状態ラベル（整数）
                                            # 0 = engaged
                                            # 1 = disengaged
                                            # ※アライメント後の番号

        # ガウス分布パラメータ（個人モデル）
        "means"             : np.ndarray,   # shape: (2, 1)
                                            # means[0] = engaged状態のμ（小さいはず）
                                            # means[1] = disengaged状態のμ（大きいはず）
        "covars"            : np.ndarray,   # shape: (2, 1, 1)

        # 遷移行列（個人モデル）
        "transmat"          : np.ndarray,   # shape: (2, 2)
                                            # transmat[i,j] = P(z_t=j | z_{t-1}=i)
                                            # 例:
                                            # [[0.95, 0.05],   engaged → ...
                                            #  [0.08, 0.92]]   disengaged → ...

        # 状態ラベルの対応（アライメント後）
        "state_labels"      : dict,         # {0: "engaged", 1: "disengaged"}

        # モデル評価
        "log_likelihood"    : float,        # 個人モデルの対数尤度
    },
    ...
]
```

### 出力例（参加者P01）

```python
{
    "participant_id": "P01",
    "n_trials": 46,
    "viterbi_states": np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, ...]),
    "means":  np.array([[8.3], [72.1]]),
    "covars": np.array([[[45.2]], [[820.4]]]),
    "transmat": np.array([
        [0.95, 0.05],   # engaged    → (engaged, disengaged)
        [0.08, 0.92],   # disengaged → (engaged, disengaged)
    ]),
    "state_labels": {0: "engaged", 1: "disengaged"},
    "log_likelihood": -163.2,
}
```

---

## 4. アルゴリズム設計

### 全体フロー

```
[入力] DataFrame（全参加者、angular_error列を含む）
   │
   ▼
Step 1: データ準備
   │  abs_angular_error = |angular_error| を計算して列を追加
   │  各参加者の abs_angular_error を (n_trials, 1) の配列に整形
   │  lengths リストを構築（参加者ごとの試行数）
   │  X_all: 全参加者データを縦に結合 → shape (全試行数, 1)
   │
   ▼
Step 2: グローバルfitting
   │  n_init=20 回の異なるランダムシードで GaussianHMM.fit(X_all, lengths) を実行
   │  各fittingの対数尤度を記録
   │  最大対数尤度のモデルをグローバルモデルとして保存
   │
   ▼
Step 3: 状態ラベルのアライメント（グローバルモデル基準）
   │  グローバルモデルの means を確認
   │  μが小さい状態 → engaged  （絶対誤差が小さい = 正確な報告）
   │  μが大きい状態 → disengaged（絶対誤差が大きい = 不正確な報告）
   │  → alignment_map = {小μの状態番号: 0, 大μの状態番号: 1} を作成
   │
   ▼
Step 4: 個人fitting（参加者ごとにループ）
   │  グローバルパラメータを初期値として設定（init_params="" で自動初期化を無効化）
   │  個人データ X_i で GaussianHMM.fit(X_i) を実行（EMを収束まで）
   │  → 個人ごとのパラメータを取得
   │
   ▼
Step 5: 状態系列の推定
   │  GaussianHMM.predict(X_i) → ビタビアルゴリズムで最尤状態系列を取得
   │  alignment_map で状態番号を統一（0=engaged, 1=disengaged）
   │
   ▼
[出力] results リスト
```

---

## 5. グローバルfittingの詳細

### `lengths` 引数の必須性

```python
# NG: lengths を渡さない場合
model.fit(X_all)
# → 参加者をまたぐ試行間にも遷移があるとみなされる（遷移行列が崩れる）

# OK: lengths を渡す場合
model.fit(X_all, lengths=lengths)
# → 参加者ごとに独立したシーケンスとして扱われる
```

### 複数初期値による安定化

```
n_init  = 20      # 初期値の試行回数（Ashwood et al. 2022 に準拠）
n_iter  = 200     # EMの最大反復回数
tol     = 1e-4    # 収束判定の閾値（対数尤度の変化量）

各シードでfitting後、対数尤度が最大のモデルをグローバルモデルとして採用
```

---

## 6. 状態ラベルのアライメント

### アルゴリズム

```python
def align_states(means: np.ndarray) -> dict:
    """
    グローバルモデルのμに基づいて状態ラベルを割り当てる。

    abs_angular_error を観測値としているため：
      μが小さい状態 → engaged    （絶対誤差が小さい）
      μが大きい状態 → disengaged （絶対誤差が大きい）

    Parameters
    ----------
    means : np.ndarray, shape (2, 1)

    Returns
    -------
    dict : {元の状態番号: 統一後の状態番号}
           例: {1: 0, 0: 1}  （元の状態1がengaged、元の状態0がdisengaged）
    """
    means_flat   = means.flatten()          # (2,)
    engaged_idx  = np.argmin(means_flat)    # μが小さい方 → engaged → 統一番号0
    disengaged_idx = np.argmax(means_flat)  # μが大きい方 → disengaged → 統一番号1

    return {
        engaged_idx    : 0,   # engaged
        disengaged_idx : 1,   # disengaged
    }

# 使用例
alignment_map = align_states(global_model.means_)
# viterbi_statesの番号を統一
viterbi_states_aligned = np.vectorize(alignment_map.get)(viterbi_states_raw)
```

---

## 7. hmmlearnの設定

```python
from hmmlearn import hmm

# -----------------------------------------------
# グローバルfitting（n_init回繰り返す）
# -----------------------------------------------
best_score = -np.inf
best_global_model = None

for seed in range(n_init):  # n_init = 20
    model = hmm.GaussianHMM(
        n_components    = 2,        # 状態数
        covariance_type = "full",   # 1次元なので "diag" でも同じ
        n_iter          = 200,
        tol             = 1e-4,
        random_state    = seed,
    )
    model.fit(X_all, lengths=lengths)
    score = model.score(X_all, lengths=lengths)  # 対数尤度
    if score > best_score:
        best_score = score
        best_global_model = model

# -----------------------------------------------
# 個人fitting（参加者ごと）
# -----------------------------------------------
individual_model = hmm.GaussianHMM(
    n_components    = 2,
    covariance_type = "full",
    n_iter          = 200,
    tol             = 1e-4,
    init_params     = "",     # 自動初期化を無効化（グローバル初期値を使うため）
    params          = "stmc", # 学習対象: startprob, transmat, means, covars
)
# グローバルパラメータを初期値として設定
individual_model.startprob_ = best_global_model.startprob_
individual_model.transmat_  = best_global_model.transmat_
individual_model.means_     = best_global_model.means_
individual_model.covars_    = best_global_model.covars_

individual_model.fit(X_i)   # X_i: shape (n_trials_i, 1)

# ビタビアルゴリズムで最尤状態系列を取得
viterbi_states_raw = individual_model.predict(X_i)

# アライメントで状態番号を統一
viterbi_states_aligned = np.vectorize(alignment_map.get)(viterbi_states_raw)
```

---

## 8. 仮定と制約まとめ

| 項目 | 仮定 | 違反した場合のリスク |
|---|---|---|
| 観測分布 | 各状態でガウス分布に従う | 裾が重い分布の場合は推定が歪む |
| 観測変数 | abs_angular_error（0°〜180°） | 符号付きangular_errorを使うと状態分離が不明確になる |
| 状態の定常性 | セッション内でμ・σ²が変化しない | 学習進行で分布が変化すると推定が歪む |
| 状態の持続性 | 遷移行列の対角成分が大きい（状態が複数試行持続する） | 試行ごとに頻繁に切り替わる場合は検出できない |
| 試行数 | 最大48試行 | 少ないためグローバル初期化が特に重要 |
| セッション数 | 1参加者 = 1セッション | 複数セッションの場合はlengthsの設計を変更する |

---

## 9. 将来的な拡張

```
状態数の変更：
  N_STATES = 3 に変更し、aligned_mapを3状態用に更新する
  （例: engaged / biased_target / biased_distractor）

観測変数の追加（多変量Gaussian HMM）：
  abs_angular_error に加えて rt（反応時間）を追加することで
  状態の分離がより明確になる可能性がある
  X_i: shape (n_trials, 2)  ← [abs_angular_error, rt]

符号付きangular_errorへの切り替え：
  biased_right / biased_leftの分離が必要になった場合は
  abs_angular_error → angular_error に変更し
  状態数を3に増やしてalignment_mapを更新する
```

---

## 10. 依存ライブラリ

```
hmmlearn >= 0.3.0
numpy
pandas
scipy
```
