import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2, ttest_ind, ttest_1samp, mannwhitneyu

from features.behavior import (
    compute_target_choice_prob_by_task_irrelevant_switch,
    compute_task_relevant_choice_rate_subjects,
    compute_target_choice_rate_subjects,
    compute_AE_standard_deviance,
    compute_RT_standard_deviance,
    compute_slope_of_ae_over_trials_all,
    cancatenate_necessary_behavioral_df,
)
from stats.metrics import (
    t_test_reward_points_between_periods,
    t_test_count_target_choice_between_periods,
    t_test_count_target_choice_between_subjects,
    t_test_learning_rate_from_switch_probs,
    test_shapiro_wilk,
    Mann_Whitney_U_test_count_target_choice_between_subjects,
    t_test_rt_difference_between_target_distractor,
)
from viz.plots import (
    plot_hist_with_lognormal_fit,
    plot_exploit_target_prob_by_switch,
    plot_logistic_regression_per_subject,
    plot_MW_vs_target_choice_across_subjects,
    plot_exp_obj_with_linear_fit,
    plot_frac_ae_target_distractor_by_trial,
    plot_rt_by_trial,

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
    behavioral_df = cancatenate_necessary_behavioral_df(all_data_learning)

    subjects_behavior_1 = behavioral_df.loc[
        # behavioral_df["task_irrelevant_choice_count"] < threshold,
        (behavioral_df["rt_std"] <= behavioral_df["rt_std"].median()),
        "subject"
    ].tolist()
    subjects_behavior_2 = behavioral_df.loc[
        # (behavioral_df["task_irrelevant_choice_count"] >= threshold),
        (behavioral_df["rt_std"] > behavioral_df["rt_std"].median()),
        "subject"
    ].tolist()

    return subjects_behavior_1, subjects_behavior_2


def run_behavior(all_data_learning, all_data_awareness=None, subjects_behavior_on=None, subjects_behavior_off=None):
    # 0) 確認したい参加者データはここで参照
    _, df_learning, df_awareness = load_and_prepare(DATA_PATH)
    # plot_logistic_regression_per_subject(df_learning)
    # plot_logistic_regression_per_subject(df_awareness)
    # plot_frac_ae_target_distractor_by_trial(df_learning)
    # plot_rt_by_trial(df_learning)
    print(df_learning.loc[:, ["num_trial", "chosen_item", "angular_error_target", "angular_error_distractor", "rt"]])

    behavioral_df =  cancatenate_necessary_behavioral_df(all_data_learning)
    # target_choice_prob_by_switch_df = compute_target_choice_prob_by_task_irrelevant_switch(all_data_learning)
    # plot_exploit_target_prob_by_switch(target_choice_prob_by_switch_df)
    # print(t_test_learning_rate_from_switch_probs(target_choice_prob_by_switch_df))

    print("\nTask relevant choice rate per subject:")
    print(behavioral_df.loc[:, ["subject", "valid_trial_count", "task_irrelevant_choice_count", "rt_std", "ae_std", "target_choice_rate", "mean_target_angular_error", "mean_angular_error"]])

    results = t_test_reward_points_between_periods(all_data_learning)
    print("\nT-test: Reward points between early and late periods")
    print(f"n = {results['n']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (all): {results['mean_total']}, "
        f"Mean (first half): {results['mean_first_half']}, "
        f"Mean (second half): {results['mean_second_half']}"
    )

    # # Shapiro-Wilk test for normality of target choice rate differences
    # results = test_shapiro_wilk(all_data_learning)
    # print(f"Shapiro-Wilk test for target choice rate normality: W={results['w_stat']}, p={results['p_value']}")

    # # Mann-Whitney U test for target choice rate between high vs low task-relevant choice groups
    # results = Mann_Whitney_U_test_count_target_choice_between_subjects(all_data_learning, subjects_behavior_on, subjects_behavior_off)
    # print("\nMann-Whitney U test: Target choice rate between subjects with high vs low task-relevant choice")
    # print(f"n_group1 = {results['n_group1']}, n_group2 = {results['n_group2']}, U-statistic: {results['u_stat']}, P-value: {results['p_value']}")
    # print(
    #     f"Mean (high task-relevant choice): {results['mean_delta_group1']}, "
    #     f"Mean (low task-relevant choice): {results['mean_delta_group2']}"
    # )

    # results = t_test_count_target_choice_between_subjects(all_data_learning, subjects_behavior_on, subjects_behavior_off)
    # print("\nT-test: Target choice rate between subjects with high vs low task-relevant choice")
    # print(f"n_group1 = {results['n_group1']}, n_group2 = {results['n_group2']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    # print(
    #     f"Mean (high task-relevant choice): {results['mean_delta_group1']}, "
    #     f"Mean (low task-relevant choice): {results['mean_delta_group2']}"
    # )

    # ae_variances = compute_AE_standard_deviance(all_data_learning)
    rt_variances = compute_RT_standard_deviance(all_data_learning)
    rt_variances_dict = dict(rt_variances)

    # ae_variances_dict = dict(ae_variances)

    # combined_variances = []
    # for subj_id, ae_var in ae_variances:
    #     if subj_id in rt_variances_dict:
    #         rt_var = rt_variances_dict[subj_id]
    #         combined_variances.append((rt_var, ae_var))
    #         print(f"Subject {subj_id}: AE Standard Deviation = {ae_var:.2f}, RT Standard Deviation = {rt_var:.2f}")
    
    # plot_exp_obj_with_linear_fit(combined_variances, title="RT Standard Deviation vs AE Standard Deviation", xlabel="RT Standard Deviation", ylabel="Angular Error Standard Deviation")

    # slope_df = compute_slope_of_ae_over_trials_all(all_data_learning)
    # print("\nSlope of AE over trials per subject:")
    # print(slope_df)

    combined_variances = []
    for index, row in behavioral_df.iterrows():
        subj_id = row["subject"]
        task_relevant_choice_rate = row["task_relevant_choice_rate"] if "task_relevant_choice_rate" in row else np.nan
        target_choice_rate = row["target_choice_rate"] if "target_choice_rate" in row else np.nan
        target_angular_error = row["mean_target_angular_error"] if "mean_target_angular_error" in row else np.nan
        distractor_angular_error = row["mean_distractor_angular_error"] if "mean_distractor_angular_error" in row else np.nan
        angular_error = row["mean_angular_error"] if "mean_angular_error" in row else np.nan
        # if slope < 0:
        #     continue
        rt_std = row["rt_std"] if "rt_std" in row else np.nan
        ae_std = row["ae_std"] if "ae_std" in row else np.nan
        combined_variances.append((rt_std, target_angular_error))
    plot_exp_obj_with_linear_fit(combined_variances, title="RT Standard Deviation vs Target Angular Error", xlabel="RT Standard Deviation", ylabel="Target Angular Error")

    # t_test_results = t_test_rt_difference_between_target_distractor(all_data_learning)
    # print("\nT-test: RT difference between target and distractor choices")
    # print(f"n_target = {t_test_results['n_target']}, n_distractor = {t_test_results['n_distractor']}, T-statistic: {t_test_results['t_stat']}, P-value: {t_test_results['p_value']}")
    # print(
    #     f"Mean RT (target choices): {t_test_results['mean_target_rt']}, "
    #     f"Mean RT (distractor choices): {t_test_results['mean_distractor_rt']}"
    #     f"Mean RT difference (target - distractor): {t_test_results['mean_diff_rt']}"
    # )   


    # # Example: AE variance histogram
    # variance_list = [var for subj_id, var in ae_variances]
    # target_choice_rate_list = behavioral_df["task_relevant_choice_rate"].tolist()
    # plot_hist_with_lognormal_fit(
    #     target_choice_rate_list,
    #     bin_size=0.02,
    #     xlabel="Task-Relevant Choice Rate",
    #     title="Histogram of Task-Relevant Choice Rates Across Subjects"
    # )

    # MWと報酬（平均ターゲット選択確率）の関係をプロット
    # plot_MW_vs_target_choice_across_subjects(all_data_learning)

    return {
        "all_data_learning": all_data_learning,
        "all_data_awareness": all_data_awareness,
        "behavioral_df": behavioral_df,
    }
