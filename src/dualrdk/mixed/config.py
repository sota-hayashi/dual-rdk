"""混合モデル解析の設定値。

ここに置く定数は「解析上の判断」であり、データの読み込み規則
(dualrdk.config) とは分けて管理する。値を変えたら outputs/mixed 以下の
run_config.json に記録が残るので、後から再現できる。
"""
from pathlib import Path

from dualrdk.config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# 出力先
# ---------------------------------------------------------------------------
MIXED_OUTPUT_DIR = OUTPUT_DIR / "mixed"


def run_dir(name: str) -> Path:
    """解析名ごとの出力ディレクトリを作って返す。"""
    d = MIXED_OUTPUT_DIR / name
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 試行除外
# ---------------------------------------------------------------------------
# 絶対上限。これを超える試行は「離席」とみなして除外する。
#
# 15秒の根拠（2026-08 の検討）:
#   - 除外されるのは 9 試行のみ（最小でも 19.8 秒）で、課題内の反応と解釈できない
#   - 参加者ごとの MAD を基準にした相対規則は、ばらつきの大きい参加者ほど多く削り、
#     rt_cv（＝参加者内 SD）の個人差を定義上圧縮してしまう
#       両側3MAD: Spearman r(除外前, 除外後 SD) = .691
#       上側3MAD: .876
#       絶対15s : .967   ← 個人差がほぼ保存される
#   - 除外試行の試行位置の偏りが唯一有意でない（KS p = .120）。相対規則では
#     遅い試行がセッション後半に集中するため除外が後半に偏り、trial の傾きを
#     機械的に急にする
#   - 速い側は除外しない。全データの RT 最小値が 547 ms で、予期反応と
#     呼べる試行が存在しないため（RT < 500 ms は 0 試行）
RT_CEILING_MS = 15000.0
RT_FLOOR_MS = None  # 速い側は除外しない（上記参照）

# ---------------------------------------------------------------------------
# 推定
# ---------------------------------------------------------------------------
# statsmodels の MixedLM は最適化手法によって破綻することがある。
# RT 上限 15 秒のデータでは bfgs / cg が logLik を 230 も取り違える
# （lbfgs: -1399.70 に対し bfgs/cg: -1630.50、tau1 も 0.0035 -> 0.369）。
# lbfgs を既定とし、CHECK_OPTIMIZERS で他手法との一致を毎回検証する。
DEFAULT_OPTIMIZER = "lbfgs"
CHECK_OPTIMIZERS = ("lbfgs", "powell", "nm", "bfgs", "cg")

# REML はランダム効果構造の比較に用いる（固定効果部分が同一のときに妥当）。
# 固定効果が異なるモデルを比べるときは ML に切り替えること。
DEFAULT_REML = True

# 参加者ブートストラップの反復数
N_BOOTSTRAP = 300
BOOTSTRAP_SEED = 1

# 図の体裁
FIG_DPI = 150
FIG_STYLE = "seaborn-v0_8-whitegrid"

# 図中の日本語を表示するためのフォント候補（先に見つかったものを使う）
CJK_FONT_CANDIDATES = (
    "Hiragino Sans",
    "Hiragino Maru Gothic Pro",
    "YuGothic",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "Arial Unicode MS",
)
