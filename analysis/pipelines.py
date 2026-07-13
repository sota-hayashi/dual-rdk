from pathlib import Path
from io_data.load import load_all_concatenated
from analysis.pipelines_behavior import run_behavior, get_subjects_by_behavior_data
from analysis.pipelines_hmm import run_hmm



def run_default():
    # 1) データを一度だけロード
    all_data_practice, all_data_learning, all_data_awareness = load_all_concatenated(
        Path("data_online_experiment"),
        subjects_include=None
    )

    # 2) 行動基準で被験者を抽出
    subjects_behavior_on, subjects_behavior_off, behavioral_df = get_subjects_by_behavior_data(all_data_learning, threshold=1)

    # 3) フィルタして再ロード
    subjects = subjects_behavior_on
    if subjects:
        all_data_practice, all_data_learning, all_data_awareness = load_all_concatenated(
            Path("data_online_experiment"),
            subjects_include=subjects
        )
        behavioral_df = None  # データ再ロード後は再計算が必要

    # 4) HMM解析
    hmm_df = run_hmm(
        all_data_learning=all_data_learning,
        train=False,
        input_type="angular_error",
    )

    # 5) 行動解析
    run_behavior(
        all_data_learning, all_data_awareness,
        subjects_behavior_on, subjects_behavior_off,
        hmm_df=hmm_df, behavioral_df=behavioral_df,
    )
