"""解析パイプライン一括実行のエントリポイント（薄いランチャー、ロジック禁止）。"""
import sys
from pathlib import Path

# pip install -e . への移行までは src/ を直接パスに追加して dualrdk を解決する
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dualrdk.pipelines.pipelines import run_default


def main():
    run_default()


if __name__ == "__main__":
    main()
