# dual-rdk リポジトリ構造再編 要件定義書

- 作成日: 2026-07-07
- 対象: `dual-rdk` リポジトリ全体（解析コード・前処理コード・データ・成果物・文書）
- 目的: ルート直下へのファイル逐次追加によって混在した構造を解消し、役割ごとに分離されたクリーンなフォルダ構造を定義する

---

## 1. 背景と現状分析

### 1.1 現状の構成

現在のリポジトリには、性質の異なる以下の6種類の資産が混在している。

| 種類 | 現在の場所 |
|---|---|
| 解析用 Python パッケージ | `analysis/`, `common/`, `features/`, `io_data/`, `stats/`, `viz/` + ルートの `analysis.py`, `utils.py` |
| 実験データ（入力） | `data00/`, `data01/`, `data_online_experiment/`, `data_excluded/` |
| 解析結果（出力） | `fig/`, `results/`, `hmm_summary/` |
| モデル仕様書・メモ | ルート直下の `*_spec.md` ×11本、`memo.md`, `explanation.md`, `hmm_gpt.md`, `glm_hmm_method_summary.md`, `instruction.md`, `slide_script.md` |
| 実験タスク本体（Web） | ルート直下の `index.html`, `demo_dual_rdk.html`, `demo_index.html`, `upload.php`, `whereami.php` |
| 論文・報告・メディア | `ieicej3.4a/`, `progress_report/`, ルート直下の `GreenIoT_0519.pdf`, `rdk_demo.mp4`, `rdk_demo.webm` |

### 1.2 問題点

1. **ルート直下の肥大化**: モデルを1つ追加するたびに `xxx_spec.md` がルートに追加される運用になっており、ルート直下に約30個のファイルが並ぶ。新規参加者（将来の自分を含む）がエントリポイントを特定できない。
2. **データディレクトリの命名が無意味**: `data00`, `data01` は中身（パイロットCSV / パイロットJSON）を表しておらず、`data_excluded` が除外基準と紐づく形で管理されていない。またデータ形式（CSV/JSON）と収集フェーズ（パイロット/本実験）の区別が名前から読めない。
3. **入力と出力の混在**: `hmm_summary/` は解析の*出力*だが、`common/config.py` から*入力*としても参照されており、生データと導出データの境界が曖昧。再現性（生データ＋コード→出力を再生成できること）が保証できない。
4. **出力ファイル名によるバージョン管理**: `hmm_summary_normalized_ver2.0_A=0.6.csv` のようにパラメータとバージョンをファイル名に埋め込んでおり、どのコード・どの設定で生成されたか追跡できない。
5. **パスのハードコード分散**: `stats/models.py` に `"results/q_learning"` 等、`viz/plots.py` に `"results/rw_learning/continuous/..."` 等の相対パスが直書きされ、`common/config.py` への集約が不完全。カレントディレクトリ依存で壊れる。
6. **実験タスク（配信物）と解析コードの同居**: `index.html` / `upload.php` はオンライン実験のデプロイ物であり、解析リポジトリのライフサイクルと異なる。
7. **論文・発表資産の散在**: `ieicej3.4a/` には LaTeX 中間生成物（`.aux`, `.log`, `.synctex.gz`）がコミットされており、動画・スライドPDFがルートに置かれている。

---

## 2. 目的とゴール

- **G1**: ルート直下は「リポジトリ全体の説明・設定ファイル・トップレベルディレクトリ」のみとする（目安: ファイル10個以下）。
- **G2**: 「コード / データ(入力) / 出力(導出物) / 文書 / 実験タスク / 論文・発表」の6区分をトップレベルで分離する。
- **G3**: 生データ（`data/raw/`）は読み取り専用とし、コード実行によって変更されない。全出力は `outputs/` 配下に再生成可能な形で書き出す。
- **G4**: 全パスを `config.py`（1ファイル）に集約し、リポジトリルート基準の絶対パスで解決する。
- **G5**: 新しいモデル・仕様書・出力を追加する際の「置き場所ルール」を明文化し、逐次追加が再びルートを汚さないようにする。

## 3. 非ゴール（スコープ外）

- 解析コードのロジック・アルゴリズムの変更
- 出力ファイルの中身・フォーマットの変更
- 過去の出力ファイル（`hmm_summary` の ver 系列など）の削除（移動のみ行い、削除判断は別途）

---

## 4. 提案フォルダ構造

```
dual-rdk/
├── README.md                     # タスク概要 + リポジトリの歩き方（各ディレクトリの説明を追記）
├── requirements.txt
├── .gitignore
│
├── src/dualrdk/                  # ★ 解析用 Python パッケージ（import 対象はすべてここ）
│   ├── __init__.py
│   ├── config.py                 # ← common/config.py（全パス・定数・除外リストを集約）
│   ├── io/                       # ← io_data/（load.py, utils.py）
│   ├── features/                 # ← features/（behavior.py, lapses.py, reward.py）
│   ├── models/                   # ← stats/（q_learning_*, rw_*, gaussian_hmm, von_mises_*, metrics, models）
│   ├── viz/                      # ← viz/（plots.py）
│   └── pipelines/                # ← analysis/ の pipelines*.py
│
├── scripts/                      # ★ 実行エントリポイント（薄いランチャーのみ、ロジック禁止）
│   ├── run_analysis.py           # ← analysis/run_analysis.py, ルートの analysis.py（重複を統合）
│   └── run_hmm_validation.py     # ← analysis/run_hmm_validation.py
│
├── data/                         # ★ 入力データ（読み取り専用）
│   ├── raw/
│   │   ├── pilot_csv/            # ← data00/（対面パイロット CSV）
│   │   ├── pilot_json/           # ← data01/（パイロット JSON）
│   │   └── online/               # ← data_online_experiment/（本実験 80名 JSON）
│   ├── excluded/                 # ← data_excluded/
│   └── README.md                 # 各データセットの由来・形式・除外基準を記載
│
├── outputs/                      # ★ コードが生成する全出力（原則 .gitignore、必要なものだけ追跡）
│   ├── figures/                  # ← fig/ のうちコード生成の図
│   ├── results/                  # ← results/（q_learning/, rw_learning/, ... のサブ構造は維持）
│   └── summaries/hmm/            # ← hmm_summary/
│
├── docs/                         # ★ 文書
│   ├── specs/                    # ← ルートの *_spec.md 全11本
│   ├── methods/                  # ← glm_hmm_method_summary.md, explanation.md
│   └── notes/                    # ← memo.md, hmm_gpt.md, instruction.md, slide_script.md
│
├── experiment/                   # ★ オンライン実験タスク（デプロイ物）
│   ├── index.html
│   ├── demo/                     # demo_dual_rdk.html, demo_index.html
│   └── server/                   # upload.php, whereami.php
│
├── reports/                      # ★ 論文・報告・発表
│   ├── ieicej3.4a/               # LaTeX ソース（中間生成物は .gitignore へ）
│   ├── progress_report/
│   └── assets/                   # GreenIoT_0519.pdf, rdk_demo.mp4, rdk_demo.webm,
│                                 # fig/ のうち手作業で作った図版（stimuli.pdf, procedure 系など）
│
└── (削除) __pycache__/, .DS_Store  # .gitignore の修正で恒久的に除外
```

### 4.1 設計判断の根拠

- **`src/` レイアウト**: パッケージを `src/dualrdk/` に置くことで、カレントディレクトリ偶然依存の import を防ぎ、`pip install -e .` による正規のパッケージ化への移行余地を残す。既存の `from io_data.load import ...` 形式は `from dualrdk.io.load import ...` に統一される。
- **`analysis/` と `stats/` の再命名**: 現状 `analysis`（パイプライン）と `stats`（モデル定義＋統計検定）の役割境界が名前から読めない。`pipelines`（実行順序の組み立て）と `models`（モデル定義）に改名して責務を明示する。
- **`fig/` の二分**: `fig/` にはコード生成の図（`rt_cv_vs_*.pdf` 等）と手作業の図版（`stimuli.pdf`, `dual-rdk-procedure.pdf` 等）が混在している。前者は再生成可能なので `outputs/figures/`（gitignore）、後者は再生成不能な資産なので `reports/assets/`（git 追跡）に分ける。
- **`hmm_summary/` の扱い**: 出力だが下流解析の入力にもなる「中間生成物」。`outputs/summaries/` に置き、下流が参照する確定版のみ git 追跡し、パスは `config.py` 経由でのみ参照する。

---

## 5. 機能要件

| ID | 要件 |
|---|---|
| FR-1 | §4 の構造どおりに全ファイルを `git mv` で移動する（履歴を保持する）。 |
| FR-2 | パッケージ名を `dualrdk` とし、`io_data→dualrdk.io`, `stats→dualrdk.models`, `analysis→dualrdk.pipelines`, `common→dualrdk`(config), `features→dualrdk.features`, `viz→dualrdk.viz` の import 書き換えを全ファイルに適用する。 |
| FR-3 | `config.py` に `REPO_ROOT = Path(__file__).resolve().parents[2]` を定義し、`DATA_DIR`, `OUTPUT_DIR`, `FIG_DIR`, `RESULTS_DIR`, `HMM_SUMMARY_DIR` をそこから導出する。コード中の文字列リテラルパス（`stats/models.py`, `viz/plots.py`, `analysis/pipelines_hmm.py` 等）をすべて config 参照に置換する。 |
| FR-4 | ルートの `analysis.py` と `analysis/run_analysis.py`（同一内容）を `scripts/run_analysis.py` に統合する。ルートの `utils.py`（re-export ハブ）は廃止し、利用箇所を直接 import に置換する。 |
| FR-5 | `.gitignore` に `outputs/`（追跡対象を除く）、`__pycache__/`（現状 `__pychache__` と typo している）、`.venv/`、LaTeX 中間生成物（`*.aux`, `*.log`, `*.synctex.gz`, `*.fls`, `*.fdb_latexmk`, `*.dvi`）を追加する。 |
| FR-6 | `data/README.md` を新設し、pilot_csv / pilot_json / online の各データセットの収集時期・形式・件数と、`data/excluded/` の除外理由（config の `EXCLUDED_SUBJECTS` との対応）を記載する。 |
| FR-7 | ルート `README.md` に「リポジトリ構成」節を追加し、各トップレベルディレクトリの役割と新規ファイルの置き場所ルール（§7）を記載する。 |

## 6. 非機能要件

| ID | 要件 |
|---|---|
| NFR-1 | **再現性**: 移行後、`python scripts/run_analysis.py` がリポジトリルート以外のカレントディレクトリからでも動作する（パスが cwd 非依存）。 |
| NFR-2 | **後方互換の確認**: 移行前後で `scripts/` の各エントリポイントが import エラーなく起動することを確認する（出力の数値一致までは必須としない）。 |
| NFR-3 | **履歴保持**: すべての移動は `git mv` で行い、`git log --follow` で追跡可能とする。 |
| NFR-4 | **段階的移行**: 1コミット=1関心事（例:「データ移動」「docs 移動」「import 書き換え」）で分割し、途中状態でもロールバック可能とする。 |

## 7. 運用ルール（逐次追加の再発防止）

1. **新しいモデルの仕様書** → `docs/specs/<model_name>_spec.md`。ルート直下への `.md` 追加は README のみ許可。
2. **新しいモデルの実装** → `src/dualrdk/models/`。実行が必要なら `scripts/` に薄いランチャーを追加。
3. **新しい出力** → `outputs/results/<model_name>/` を出力先とし、パスは必ず `config.py` に定数として追加してから参照する。ファイル名へのパラメータ埋め込み（`_ver2.0_A=0.6` 等）は行わず、パラメータ違いはサブディレクトリまたは結果 CSV 内のカラムで表現する。
4. **新しいデータ** → `data/raw/` 配下に収集フェーズが分かる名前でディレクトリを切り、`data/README.md` に1行追記する。
5. **発表・論文資産**（PDF・動画・スライド）→ `reports/assets/`。ルート直下へのバイナリ追加は禁止。

## 8. 移行手順（推奨順序）

1. `.gitignore` 修正 + `__pycache__` / `.DS_Store` の追跡解除（コード無変更）
2. `docs/`, `reports/`, `experiment/` への文書・資産の `git mv`（コード無変更、影響ゼロ）
3. `data/` へのデータ移動 + `config.py` の `DATA_PATH` 系更新
4. `outputs/` への出力移動 + ハードコードパスの config 集約（FR-3）
5. `src/dualrdk/` へのパッケージ移動 + import 一括書き換え（FR-2, FR-4）
6. エントリポイント起動確認（NFR-2）+ README 更新（FR-7）

各ステップを独立コミットとし、ステップ5のみが既存の実行方法（`python analysis/run_analysis.py` 等）を変更する破壊的変更である点に留意する。

## 9. リスクと対応

| リスク | 対応 |
|---|---|
| import 書き換え漏れによる実行時エラー | ステップ5の後に全エントリポイントの起動確認（NFR-2）。`grep` で旧パッケージ名（`io_data`, `common.config` 等）の残存を機械的に検査する。 |
| `hmm_summary` 内のどの CSV が「確定版」か不明 | 移動時は全ファイルを `outputs/summaries/hmm/` に保持し、削除判断は行わない（非ゴール）。`config.py` が参照する2ファイルのみ git 追跡を維持する。 |
| 実験タスク（PHP）のデプロイパス変更 | `experiment/` への移動はリポジトリ内の整理であり、サーバ側のデプロイ手順に影響する場合は `experiment/README.md` にデプロイ方法を記録してから移動する。 |
