# 2-state HMM (探索/定着) を chosen_item（1/0/-1）だけで当てる — 実装インストラクション

この instruction.md は、**chosen_item（1=target, 0=distractor, -1=else）**のみを観測系列として、
各被験者の行動に **2状態HMM（探索 / 定着）** をフィットし、状態系列と指標を抽出するための実装手順をまとめたものです。
VS Code 等での編集を想定した Markdown 形式です。

---

## 0. ゴール（出力物）

被験者ごとに以下を得る：

- HMM パラメータ
  - 遷移行列 A（2×2）
  - 初期状態確率 π（2,）
  - 放出確率（各状態で chosen_item が 1/0/-1 を出す確率）B（2×3）
- 推定状態系列（Viterbi による最尤状態列） z_t ∈ {0,1}
- サマリー指標
  - 状態の滞在率（fraction of time）
  - 状態遷移回数（switch count）
  - 各状態の平均継続長（run length）
  - （任意）各状態での target率 / distractor率 / else率

---

## 1. 前提（データ仕様）

各被験者について、trial 時系列で並んだ chosen_item を持つこと。

- chosen_item ∈ {1, 0, -1}
  - 1: target を選択したと分類
  - 0: distractor を選択したと分類
  - -1: else（無効/適当/判定不能など）

注意：HMM は「順序」を使うので、trial順にソートされている必要がある。

---

## 2. 2状態HMMのモデル定義

### 2.1 潜在状態（latent states）
- state 0: 探索（exploration）
- state 1: 定着（exploitation / stabilized）

※ このラベルは “解釈” であり、推定後に状態の放出確率から割り当てる（後述）。

### 2.2 観測（emissions）
観測 y_t はカテゴリ値（3カテゴリ）。
- y_t は整数カテゴリとして表現する（例：{-1,0,1} を {0,1,2} にリマップ）

各状態 k における放出確率：
- P(y_t = c | z_t = k) = B[k, c]

### 2.3 状態遷移
- P(z_t=j | z_{t-1}=i) = A[i, j]
- A は 2×2、各行は和が 1

---

## 3. 実装方針（推奨）

### 3.1 ライブラリ候補（Python）

- `hmmlearn` の `MultinomialHMM`（カテゴリ系列）
  - version によりワンホット入力が必要な場合あり
- 代替：`pomegranate`（Categorical HMM に強い）
- さらに代替：`ssm`（状態空間/HSMM等も可能）

まずは `pomegranate` または `hmmlearn` で実装し、安定しない場合に切り替える。

---

## 4. 手順（Step-by-step）

### Step 1: 前処理（カテゴリの整数化）
目的：HMM実装が扱いやすい表現にする。

- chosen_item を 3カテゴリ整数に変換
  - 例：{-1,0,1} → {0,1,2}
- 欠損や除外試行がある場合は、trial順のまま落とす
- 被験者ごとに系列 `y`（shape: [T]）を作る

出力：被験者ごとに `seqs = [y_subj1, y_subj2, ...]`

### Step 2: モデル初期化（重要）
目的：局所解を避ける。2状態はラベルスイッチが起きる。

- 状態数を固定：n_states = 2
- 遷移行列 A は「滞在しやすい」初期値が安定
  - 例：A = [[0.90, 0.10],
            [0.10, 0.90]]
- 放出確率 B は弱い事前仮定で初期化（任意）
  - 探索：target/distractor が混ざる
  - 定着：target が多い

※ 最終的な探索/定着ラベルは Step 5 で決めるので、初期ラベルは仮で良い。

### Step 3: フィット（EM）
目的：A, B, π をデータから推定する。

- 各被験者に個別にフィット（まずはここから）
- 収束：log-likelihood が改善しなくなったら停止
- **複数初期値（random restarts）**を行い、最良の log-likelihood の解を採用（推奨）
  - 例：n_init = 10〜30

出力：subjectごとに fitted model（A, B, π, loglik）

### Step 4: デコード（Viterbi）
目的：各trialがどの状態に属するか（最尤状態列）を得る。

- Viterbi で状態列 `z` を推定
- z_t ∈ {0,1}

出力：trial-level state series `z`

### Step 5: 状態の意味づけ（探索/定着の割り当て）
目的：推定された state0/state1 を “探索/定着” に対応づける。

推奨ルール：
- 各状態の放出確率 B[k, target] を比較し、
  - target確率が高い状態 → **定着**
  - target確率が低い（または混合が強い）状態 → **探索**
- もし「-1 が多い状態＝逸脱/無効」としたい場合は、
  - B[k, else] を用いて別基準で割り当ててもよい

出力：state_id_to_label（例：{0:'explore', 1:'exploit'}）

### Step 6: 指標の計算（被験者ごと）
目的：後段解析（reward, awareness, MW指標との関係）に使える summary を作る。

- 滞在率：mean(z==exploit)
- 遷移回数：sum(z[t]!=z[t-1])
- 平均継続長：run-length encoding で各状態の平均連長
- 状態別の観測割合：P(chosen_item=1 | exploit) など

出力：`hmm_summary.csv` に保存可能な dict / DataFrame

---

## 5. 可視化（最小セット）

被験者ごとに：

- 上段：chosen_item（-1/0/1）を時系列でプロット
- 下段：推定状態 z_t を explore/exploit の帯で表示
- 併記：B（棒グラフ）と A（ヒートマップ）

目的：
- 「山あり谷あり」が状態遷移として表現できているか
- 定着状態が本当に target を多く出しているか

---

## 6. 期待される結果パターン（解釈ガイド）

### パターンA：探索→定着（典型）
- 序盤 explore が多い
- 中盤以降 exploit が増える
- A の対角が高い（状態が持続）

解釈：単調増加型に近い学習。

### パターンB：最初から定着（早期到達）
- ほぼ全trialが exploit
- chosen_item=1 が序盤から多い
- logistic の傾きが出ないことがある

解釈：変化がないのではなく、初期から安定。

### パターンC：探索↔定着の往復（再探索）
- exploit の途中に explore が挿入される
- switch count が多い
- A の非対角が相対的に大きい

解釈：注意/戦略の揺らぎ、局所最適からの離脱の可能性。

### パターンD：else が支配的
- ある状態で -1 の放出確率が非常に高い

解釈：探索ではなく「無効/逸脱」状態の可能性。2状態で吸収しきれないなら 3状態（invalid を追加）も検討。

---

## 7. 注意点（落とし穴）

- chosen_item=-1 が少数だと推定が不安定になりやすい
  - まず -1 を除外して 2カテゴリ（0/1）で試すのも手
- 2状態は表現力が限定的
  - -1 を独立状態として扱うなら 3状態HMMが自然
- 被験者ごとの trial 数が短いと過学習しやすい
  - random restarts + 擬似カウント（正則化）を検討

---

## 8. 最終アウトプット（ファイル）

- `hmm_params_subject_<id>.json`（A, B, π, loglik）
- `hmm_states_subject_<id>.csv`（trial, chosen_item, state_label）
- `hmm_summary.csv`（subject_id, frac_exploit, switch_count, mean_run_exploit, ...）

---

## 9. 実装依頼用の短い要件（AIに渡す用）

- 入力：被験者ごとの chosen_item 時系列（-1/0/1）
- 2状態 categorical HMM を各被験者に fit（random restarts 付き）
- Viterbi で状態列をデコード
- 放出確率の target 成分が大きい状態を exploit とラベル付け
- 指標（滞在率、遷移回数、平均連長、状態別カテゴリ割合）を算出
- 可視化：chosen_item と状態列を同一図に表示
- 出力：params/state/summary を保存

## memo
instruction_gpt.mdを熟読し実行して．hhmはstats/models.py，プロット用の図はviz/plots.pyにコードを書くようにしてください．models.pyには各被験者のデータフレームに対するhhmのコードを書き，そしてそれを全ての被験者のデータフレームconcat_list（all_df_learning）を入力とする場合で回すためのコードをmodels.pyとpipelines.pyに書いてください．