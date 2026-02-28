from pathlib import Path
from common.config import DATA_PATH, SUMMARY_PATH
from io_data.load import load_and_prepare, load_all_concatenated
from features.behavior import relabel_hmm_states
from analysis.pipelines_behavior import run_behavior, get_subjects_by_behavior_data
from analysis.pipelines_hmm import run_hmm, get_subjects_by_hmm_category
from viz.plots import plot_logistic_regression_per_subject



def run_default():
    # 1) データを一度だけロード
    all_data_practice, all_data_learning, all_data_awareness = load_all_concatenated(
        Path("data_online_experiment"),
        subjects_include=None
    )

    # 2) HMM基準で被験者を抽出
    group1 = "on-to-on"
    group2 = "off-to-on"
    group3 = "on-to-off"
    group4 = "off-to-off"
    group5 = "on-off-cycling"
    categories = [group4]

    subjects_hmm, _ = run_hmm(all_data_learning=all_data_learning, categories=categories, train=False)

    # 3) 行動基準で被験者を抽出
    subjects_behavior_on, subjects_behavior_off = get_subjects_by_behavior_data(all_data_learning, threshold=2)

    # 4) 両方の条件を満たす被験者を抽出（AND）
    subjects = subjects_behavior_off
    # 5) フィルタして行動解析
    if subjects:
        all_data_practice, all_data_learning, all_data_awareness = load_all_concatenated(
            Path("data_online_experiment"),
            subjects_include=subjects
        )
    run_behavior(all_data_learning, all_data_awareness, subjects_behavior_on, subjects_behavior_off)
