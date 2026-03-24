import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2, ttest_ind, ttest_1samp, mannwhitneyu
from io_data.utils import combine_subjects

from features.behavior import (
    compute_target_choice_prob_by_task_irrelevant_switch,
    compute_task_relevant_choice_rate_subjects,
    compute_minimum_AE_standard_deviance,
    compute_RT_standard_deviance,
    compute_RT_mean,
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
    t_test_rt_difference_between_white_black,
    t_test_Target_Angular_Error_between_periods,
    t_test_Minimum_Angular_Error_between_periods,
    test_log_normality,
    compare_distributions,
    evaluate_lognormal_fit,
    fit_exgaussian_and_evaluate,
)
from stats.models import (
    logit_regression,
)
from viz.plots import (
    plot_hist_with_lognormal_fit,
    plot_exploit_target_prob_by_switch,
    plot_logistic_regression_per_subject,
    plot_MW_vs_target_choice_across_subjects,
    plot_exp_obj_with_linear_fit,
    plot_frac_ae_target_distractor_by_trial,
    plot_rt_by_trial,
    qqplot_lognormal,

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
        (behavioral_df["rt_cv"] <= behavioral_df["rt_cv"].quantile(0.45)),
        "subject"
    ].tolist()
    subjects_behavior_2 = behavioral_df.loc[
        # (behavioral_df["task_irrelevant_choice_count"] >= threshold),
        (behavioral_df["rt_cv"] <= behavioral_df["rt_cv"].quantile(0.90)) & (behavioral_df["rt_cv"] > behavioral_df["rt_cv"].quantile(0.45)),
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
    # results = logit_regression(all_data_awareness)
    # print("\nLogistic Regression Results:")
    # print(results.keys())

    print(df_learning.loc[:, ["num_trial", "target_group", "chosen_color", "chosen_item", "angular_error_target", "angular_error_distractor", "reward_points", "prev_win", "win_stay", "lose_switch"]])
    # print(df_learning.loc[df_learning["target_group"] == "white", ["num_trial", "chosen_item", "angular_error_target", "angular_error_distractor"]].reset_index(drop=True))

    behavioral_df =  cancatenate_necessary_behavioral_df(all_data_learning)
    # target_choice_prob_by_switch_df = compute_target_choice_prob_by_task_irrelevant_switch(all_data_learning)
    # plot_exploit_target_prob_by_switch(target_choice_prob_by_switch_df)
    # print(t_test_learning_rate_from_switch_probs(target_choice_prob_by_switch_df))

    print("\nTask relevant choice rate per subject:")
    print(behavioral_df.loc[:, ["subject", "valid_trial_count", "task_irrelevant_choice_count", "rt_std", "rt_cv", "min_ae_std", "target_choice_rate", "target_choice_rate_diff", "mean_target_angular_error", "mean_min_angular_error"]])
    print("\nMean win-stay rate:")
    print(behavioral_df["win_stay_rate"].mean())
    print("\nMean lose-switch rate:")
    print(behavioral_df["lose_switch_rate"].mean())

    results = t_test_count_target_choice_between_periods(all_data_learning)
    print("\nT-test: Target choice count between early and late periods")
    print(f"n = {results['n']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (all): {results['mean_total']}, "
        f"Mean (first half): {results['mean_first_half']}, "
        f"Mean (second half): {results['mean_second_half']}"
    )

    results = t_test_reward_points_between_periods(all_data_learning)
    print("\nT-test: Reward points between early and late periods")
    print(f"n = {results['n']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (all): {results['mean_total']}, "
        f"Mean (first half): {results['mean_first_half']}, "
        f"Mean (second half): {results['mean_second_half']}"
    )

    results = t_test_Minimum_Angular_Error_between_periods(all_data_learning)
    print("\nT-test: Minimum Angular Error between early and late periods")
    print(f"n = {results['n']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (all): {results['mean_total']}, "
        f"Mean (first half): {results['mean_first_half']}, "
        f"Mean (second half): {results['mean_second_half']}"
    )

    results = t_test_Target_Angular_Error_between_periods(all_data_learning)
    print("\nT-test: Target Angular Error between early and late periods")
    print(f"n = {results['n']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(
        f"Mean (all): {results['mean_total']}, "
        f"Mean (first half): {results['mean_first_half']}, "
        f"Mean (second half): {results['mean_second_half']}"
    )

    # # # Shapiro-Wilk test for normality of target choice rate differences
    # # results = test_shapiro_wilk(all_data_learning)
    # # print(f"Shapiro-Wilk test for target choice rate normality: W={results['w_stat']}, p={results['p_value']}")

    # # # Mann-Whitney U test for target choice rate between high vs low task-relevant choice groups
    # # results = Mann_Whitney_U_test_count_target_choice_between_subjects(all_data_learning, subjects_behavior_on, subjects_behavior_off)
    # # print("\nMann-Whitney U test: Target choice rate between subjects with high vs low task-relevant choice")
    # # print(f"n_group1 = {results['n_group1']}, n_group2 = {results['n_group2']}, U-statistic: {results['u_stat']}, P-value: {results['p_value']}")
    # # print(
    # #     f"Mean (high task-relevant choice): {results['mean_delta_group1']}, "
    # #     f"Mean (low task-relevant choice): {results['mean_delta_group2']}"
    # # )

    # results = t_test_count_target_choice_between_subjects(all_data_learning, subjects_behavior_on, subjects_behavior_off)
    # print("\nT-test: Target choice rate between subjects with high vs low task-relevant choice")
    # print(f"n_group1 = {results['n_group1']}, n_group2 = {results['n_group2']}, T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    # print(
    #     f"Mean (high task-relevant choice): {results['mean_delta_group1']}, "
    #     f"Mean (low task-relevant choice): {results['mean_delta_group2']}"
    # )


    

    # # slope_df = compute_slope_of_ae_over_trials_all(all_data_learning)
    # # print("\nSlope of AE over trials per subject:")
    # # print(slope_df)

    # combined_variances = []
    # for index, row in behavioral_df.iterrows():
    #     subj_id = row["subject"]
    #     task_relevant_choice_rate = row["task_relevant_choice_rate"] if "task_relevant_choice_rate" in row else np.nan
    #     target_choice_rate = row["target_choice_rate"] if "target_choice_rate" in row else np.nan
    #     target_choice_rate_diff = row["target_choice_rate_diff"] if "target_choice_rate_diff" in row else np.nan
    #     target_angular_error = row["mean_target_angular_error"] if "mean_target_angular_error" in row else np.nan
    #     target_angular_error_diff = row["target_angular_error_diff"] if "target_angular_error_diff" in row else np.nan
    #     distractor_angular_error = row["mean_distractor_angular_error"] if "mean_distractor_angular_error" in row else np.nan
    #     angular_error = row["mean_min_angular_error"] if "mean_min_angular_error" in row else np.nan
    #     angular_error_diff = row["angular_error_diff"] if "angular_error_diff" in row else np.nan
    #     reward_points = row["mean_reward_points"] if "mean_reward_points" in row else np.nan
    #     reward_points_diff = row["mean_reward_points_diff"] if "mean_reward_points_diff" in row else np.nan
    #     # if slope < 0:
    #     #     continue
    #     rt_mean = row["rt_mean"] if "rt_mean" in row else np.nan
    #     rt_std = row["rt_std"] if "rt_std" in row else np.nan
    #     rt_cv = row["rt_cv"] if "rt_cv" in row else np.nan
    #     min_ae_std = row["min_ae_std"] if "min_ae_std" in row else np.nan
    #     win_stay_rate = row["win_stay_rate"] if "win_stay_rate" in row else np.nan
    #     lose_switch_rate = row["lose_switch_rate"] if "lose_switch_rate" in row else np.nan
    #     combined_variances.append((rt_cv, target_choice_rate_diff))
    # plot_exp_obj_with_linear_fit(
    #     combined_variances,
    #     title="RT Coefficient of Variation vs Target Choice Rate Diff (second - first)", 
    #     xlabel="RT Coefficient of Variation", 
    #     ylabel="Target Choice Rate Diff (second - first)",
    #     # save_path="./fig/rt_cv_vs_minimum_angular_error.pdf"
    #     )

    t_test_results = t_test_rt_difference_between_target_distractor(all_data_learning)
    print("\nT-test: RT difference between target and distractor choices")
    print(f"n_target = {t_test_results['n_target']}, n_distractor = {t_test_results['n_distractor']}, T-statistic: {t_test_results['t_stat']}, P-value: {t_test_results['p_value']}")
    print(
        f"Mean RT (target choices): {t_test_results['mean_target_rt']} ms, "
        f"Std RT (target choices): {t_test_results['std_target_rt']} ms, "
        f"\nMean RT (distractor choices): {t_test_results['mean_distractor_rt']} ms, "
        f"Std RT (distractor choices): {t_test_results['std_distractor_rt']} ms, "
        f"\nMean RT difference (target - distractor): {t_test_results['mean_diff_rt']} ms"
    )   

    # t_test_results = t_test_rt_difference_between_white_black(all_data_learning)
    # print("\nT-test: RT difference between white and black target groups")
    # print(f"n_white = {t_test_results['n_white']}, n_black = {t_test_results['n_black']}, T-statistic: {t_test_results['t_stat']}, P-value: {t_test_results['p_value']}")
    # print(
    #     f"Mean RT (white target group): {t_test_results['mean_white_rt']} ms, "
    #     f"Std RT (white target group): {t_test_results['std_white_rt']} ms, "
    #     f"\nMean RT (black target group): {t_test_results['mean_black_rt']} ms, "
    #     f"Std RT (black target group): {t_test_results['std_black_rt']} ms, "
    #     f"\nMean RT difference (white - black): {t_test_results['mean_diff_rt']} ms"
    # )

    # # # Example: AE variance histogram
    # combined = combine_subjects(all_data_learning)
    # combined = combined.dropna(subset=["rt"]).copy()
    # rt_values = combined["rt"].tolist()
    # rt_values = np.log(rt_values) 
    # target_choice_rate_list = behavioral_df["task_relevant_choice_rate"].tolist()
    # plot_hist_with_lognormal_fit(
    #     rt_values,
    #     bin_size=0.12,
    #     xlabel="Reaction Time",
    #     title="Histogram of Reaction Time Across Subjects",
    #     # save_path="./fig/rt_histgram_lognormal_fit.pdf"
    # )

    # qqplot_lognormal(variance_list)
    # results = evaluate_lognormal_fit(variance_list)
    # print(results)
    # results = compare_distributions(variance_list)
    # print(results)

    # res = fit_exgaussian_and_evaluate(
    # values=rt_values,
    # n_mc_samples=5000,
    # statistic="ad",   # RTならまず "ad" がよい
    # random_state=0
    # )

    # print(res["fit_exgaussian"])
    # print(res["goodness_of_fit_test"])
    # print(res["fit_quality"])

    # MWと報酬（平均ターゲット選択確率）の関係をプロット
    # plot_MW_vs_target_choice_across_subjects(all_data_learning)

    return {
        "all_data_learning": all_data_learning,
        "all_data_awareness": all_data_awareness,
        "behavioral_df": behavioral_df,
    }
