from pathlib import Path

# リポジトリルート（このファイルの位置基準で解決し、cwd に依存しない）
REPO_ROOT = Path(__file__).resolve().parents[2]

# 入力データ（読み取り専用）
DATA_DIR = REPO_ROOT / "data"
PILOT_CSV_DATA_DIR = DATA_DIR / "raw" / "pilot_csv"
PILOT_JSON_DATA_DIR = DATA_DIR / "raw" / "pilot_json"
ONLINE_DATA_DIR = DATA_DIR / "raw" / "online"
EXCLUDED_DATA_DIR = DATA_DIR / "excluded"

# デフォルトのデータパスはJSONに差し替え（必要に応じて変更してください）
DATA_PATH = ONLINE_DATA_DIR / "6977bd8c4a66002ceaa54c1d.json"

# コードが生成する全出力（原則 .gitignore、確定版サマリのみ追跡）
OUTPUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
HMM_SUMMARY_DIR = OUTPUT_DIR / "summaries" / "hmm"

SUMMARY_PATH = HMM_SUMMARY_DIR / "hmm_summary_decisive_prob.csv"

PRACTICE_ROWS = 32
ROWS_PER_SESSION = 96
TRIALS_PER_SESSION = 48
ROWS_FOR_AWARENESS = 48

GAUSSIAN_HMM_SUMMARY_PATH = HMM_SUMMARY_DIR / "gaussian_hmm_summary.csv"

# 除外参加者リスト
EXCLUDED_SUBJECTS = [
    # 色バイアスが強い被験者群（80%以上の偏りを示す）
    # in group on-to-on
    "5671131573f58b0005664333",  # b=3, w=45
    "65fd4ff0fac6ac4525f54b88",  # b=45, w=0
    "66a50231bda26954b4e43e7d",  # b=39, w=9
    "696c3f1675addf129b4bff87",  # b=43, w=5
    # in group off-to-on
    "62b2080f8f89f2f15c47d9ba",  # b=39, w=5
    # in group on-to-off
    "69126cc06844917f79f2ec58",  # b=37, w=8
    "65794b62e4bbf95a4f2c9f03",  # b=30, w=6
    "616033a44ba802b7e18daaa9",  # b=32, w=4
    # in group off-to-off
    "66cb3e461d3bc3f143a834bb",  # b=44, w=3
    "67d1be8cb9034e17620cd166",  # b=46, w=0
    "665f23fcab11c11fb972a667",  # b=46, w=0
    # in group on-off-cycling
    "693348df632d7f923e83d2bb",  # b=43, w=2
    "68238de3a3ba8b99fef9b7ca",  # b=46, w=2
    "677d283c3ac4eacdfc7a59b4",  # b=39, w=1
    "611ce44efa3822c780ae383e",  # b=35, w=2
    # task-irrelevantな試行が多い被験者
    "67800b133eced63d8ec0cde8",  # 16試行
    "697cc09a8dd7b2c8061ff4e5",  # 18試行
    # learningとawarenessの両方でターゲット選択割合の向上傾向が確認されている被験者
    "6977bd8c4a66002ceaa54c1d",
    "6978e94c86ef2c792e089759",
]
