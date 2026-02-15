import numpy as np

from features.behavior import (
    compute_task_relevant_choice_rate_subjects,
    compute_target_choice_prob_by_task_irrelevant_switch,
    compute_target_choice_rate_subjects,
    compute_AE_variance,
)
from stats.metrics import (
    t_test_count_target_choice_between_periods,
    t_test_count_target_choice_between_subjects,
    t_test_learning_rate_from_switch_probs,
)
from viz.plots import (
    plot_hist_with_lognormal_fit,
    plot_exploit_target_prob_by_switch,
    plot_logistic_regression_per_subject,
    plot_MW_vs_reward_across_subjects,

)
from common.config import (
    SUMMARY_PATH,
    DATA_PATH,
)
from io_data.load import (
    load_and_prepare,
)


def get_subjects_by_behavior_data(
    all_data_learning,
    threshold: float = 0.8
):
    """
    行動指標に基づいて被験者を抽出する。
    例: task_relevant_choice_count >= threshold を満たす被験者。
    """
    task_relevant_choice_df = compute_task_relevant_choice_rate_subjects(all_data_learning)
    subjects_behavior_1 = task_relevant_choice_df.loc[
        task_relevant_choice_df["task_irrelevant_choice_count"] < threshold - 1,
        "subject"
    ].tolist()
    subjects_behavior_2 = task_relevant_choice_df.loc[
        task_relevant_choice_df["task_irrelevant_choice_count"] >= threshold,
        "subject"
    ].tolist()

    return subjects_behavior_1, subjects_behavior_2


def run_behavior(all_data_learning, all_data_awareness=None, subjects_behavior_on=None, subjects_behavior_off=None):
    task_relevant_choice_df = compute_task_relevant_choice_rate_subjects(all_data_learning)
    target_choice_df = compute_target_choice_rate_subjects(all_data_learning)
    target_choice_prob_by_switch_df = compute_target_choice_prob_by_task_irrelevant_switch(all_data_learning)
    # plot_exploit_target_prob_by_switch(target_choice_prob_by_switch_df)
    # print(t_test_learning_rate_from_switch_probs(target_choice_prob_by_switch_df))

    print("\nTask relevant choice rate per subject:")
    print(task_relevant_choice_df)

    results = t_test_count_target_choice_between_periods(all_data_learning)
    print("\nT-test: Target choice rate between early and late periods")
    print(f"n = {results['n']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (all): {results['mean_total']}, "
        f"Mean (first half): {results['mean_first_half']}, "
        f"Mean (second half): {results['mean_second_half']}"
    )

    results = t_test_count_target_choice_between_subjects(all_data_learning, subjects_behavior_on, subjects_behavior_off)
    print("\nT-test: Target choice rate between subjects with high vs low task-relevant choice")
    print(f"n_group1 = {results['n_group1']}, n_group2 = {results['n_group2']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (high task-relevant choice): {results['mean_delta_group1']}, "
        f"Mean (low task-relevant choice): {results['mean_delta_group2']}"
    )

    # Example: AE variance histogram
    # ae_variances = compute_AE_variance(all_data_learning)
    # variance_list = [var for subj_id, var in ae_variances]
    # plot_hist_with_lognormal_fit(
    #     variance_list,
    #     bin_size=50.0,
    #     xlabel="Angular Error Variance",
    #     title="Histogram of Angular Error Variances Across Subjects"
    # )

    # MWと報酬（平均ターゲット選択確率）の関係をプロット
    # plot_MW_vs_reward_across_subjects(all_data_learning)

    return {
        "all_data_learning": all_data_learning,
        "all_data_awareness": all_data_awareness,
        "task_relevant_choice_df": task_relevant_choice_df,
        "target_choice_df": target_choice_df,
    }
