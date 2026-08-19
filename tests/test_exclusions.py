"""参加者除外基準の検証。

基準4（rt_cv 上位10%）は標本内の相対位置で決まるため config.EXCLUDED_SUBJECTS に
凍結してある。凍結値が現データからの再計算と一致することをここで担保する。
データを追加・変更して不一致になったら、凍結値を更新するかどうかを明示的に
判断する必要がある（黙って N が変わるのを防ぐのがこのテストの目的）。
"""
from pathlib import Path

import pytest

from dualrdk.config import (
    EXCLUDED_SUBJECTS,
    ONLINE_DATA_DIR,
    RT_CV_EXCLUDED_SUBJECTS,
    RT_CV_EXCLUSION_QUANTILE,
)
from dualrdk.features.behavior import select_subjects_by_rt_cv
from dualrdk.io.load import load_all_concatenated

pytestmark = pytest.mark.skipif(
    not Path(ONLINE_DATA_DIR).is_dir(), reason="生データが無い環境ではスキップ"
)

N_EXPECTED_AFTER_EXCLUSION = 54


N_AFTER_CRITERIA_1_TO_3 = 61


@pytest.fixture(scope="module")
def learning_without_rt_cv_exclusion():
    """基準1〜3のみを適用した状態（= 凍結値を計算したときの標本、61名）。

    rt_cv の閾値は 0.90 分位点、すなわち **標本に依存する**。したがって
    凍結値を検証するには、凍結したときとまったく同じ標本を復元しなければ
    ならない。`subjects_include` では内部の除外リストが無条件に効いてしまい
    54 名になるので、`apply_exclusion_list=False` で全員読んでから基準1〜3
    分だけを手で落とす。
    """
    static_only = set(EXCLUDED_SUBJECTS) - set(RT_CV_EXCLUDED_SUBJECTS)
    _, learning_all, _ = load_all_concatenated(
        Path(ONLINE_DATA_DIR), apply_exclusion_list=False
    )
    learning = [(s, df) for s, df in learning_all if s not in static_only]
    assert len(learning) == N_AFTER_CRITERIA_1_TO_3, (
        f"基準1〜3適用後が {len(learning)} 名（{N_AFTER_CRITERIA_1_TO_3} 名を期待）"
    )
    return learning


def test_frozen_rt_cv_exclusion_matches_recomputation(learning_without_rt_cv_exclusion):
    """凍結した7名が再計算結果と一致する。"""
    _, excluded, threshold = select_subjects_by_rt_cv(
        learning_without_rt_cv_exclusion, q=RT_CV_EXCLUSION_QUANTILE
    )
    assert sorted(excluded) == sorted(RT_CV_EXCLUDED_SUBJECTS), (
        "rt_cv 基準の凍結値がデータと一致しない。"
        f"再計算={sorted(excluded)} 凍結={sorted(RT_CV_EXCLUDED_SUBJECTS)} 閾値={threshold:.6f}"
    )


def test_final_sample_size():
    """全4基準を適用した最終的な解析対象は54名。"""
    _, learning, _ = load_all_concatenated(Path(ONLINE_DATA_DIR))
    assert len(learning) == N_EXPECTED_AFTER_EXCLUSION


def test_exclusion_list_has_no_duplicates():
    assert len(EXCLUDED_SUBJECTS) == len(set(EXCLUDED_SUBJECTS))
