import numpy as np

from features.behavior import (
    compute_task_relevant_choice_rate_subjects,
    compute_target_choice_prob_by_task_irrelevant_switch,
    compute_target_choice_rate_subjects,
    compute_AE_variance,
)
from stats.metrics import (
    t_test_count_target_choice_between_periods,
    t_test_learning_rate_from_switch_probs,
)
from viz.plots import (
    plot_hist_with_lognormal_fit,
    plot_exploit_target_prob_by_switch,

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
    return task_relevant_choice_df.loc[
        task_relevant_choice_df["task_relevant_choice_count"] >= threshold,
        "subject"
    ].tolist()


def run_behavior(all_data_learning, all_data_awareness=None):
    task_relevant_choice_df = compute_task_relevant_choice_rate_subjects(all_data_learning)
    target_choice_df = compute_target_choice_rate_subjects(all_data_learning)
    target_choice_prob_by_switch_df = compute_target_choice_prob_by_task_irrelevant_switch(all_data_learning)
    plot_exploit_target_prob_by_switch(target_choice_prob_by_switch_df)
    print(t_test_learning_rate_from_switch_probs(target_choice_prob_by_switch_df))

    print("\nTask relevant choice rate per subject:")
    print(task_relevant_choice_df)
    print(
        f"\nMean task relevant choice rate across subjects: "
        f"{task_relevant_choice_df['task_relevant_choice_rate'].mean()}"
    )

    results = t_test_count_target_choice_between_periods(all_data_learning)
    print("\nT-test: Target choice rate between early and late periods")
    print(f"T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (all): {results['mean_total']}, "
        f"Mean (first half): {results['mean_first_half']}, "
        f"Mean (second half): {results['mean_second_half']}"
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

    return {
        "all_data_learning": all_data_learning,
        "all_data_awareness": all_data_awareness,
        "task_relevant_choice_df": task_relevant_choice_df,
        "target_choice_df": target_choice_df,
    }
