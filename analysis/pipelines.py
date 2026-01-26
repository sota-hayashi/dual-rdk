from pathlib import Path

from common.config import DATA_PATH, SUMMARY_PATH
from io_data.load import load_and_prepare, load_all_concatenated, load_categorized_subjects, load_hmm_summary
from features.behavior import summarize_chosen_item_errors, analyze_color_accuracy_change_across_subjects, relabel_hmm_states, compute_exploit_target_prob_by_switch, compute_target_choice_rate_per_subject
from features.lapses import label_if_ooz
from viz.plots import plot_MW_vs_reward_across_subjects, plot_rt_by_trial, plot_rt_histogram_all_subjects, plot_reward_by_trial, plot_logistic_regression_per_subject, plot_hmm_subject_result, plot_exploit_target_prob_by_switch, plot_frac_exploit_vs_valid_choice_rate
from stats.metrics import anova_rt_by_chosen_item, anova_reward_by_periods, anova_count_of_target_choice_by_periods, t_test_rt_between_choices, t_test_count_target_choice_between_periods, t_test_reward_points_between_periods
from stats.models import logit_regression, linear_regression, fit_hmm_across_subjects
from common.config import TRIALS_PER_SESSION


def run_default():
    df_practice, df_learning, df_awareness = load_and_prepare(DATA_PATH)
    # print(df_learning.loc[:, [
    #     "rt",
    #     "bad_response",
    #     "num_session",
    #     "num_trial",
    #     "target_group",
    #     "angular_error_target",
    #     "angular_error_distractor",
    #     "reward_points",
    #     "chosen_item",
    # ]].head())
    # print("\nThe number of trials answered with no movement (bad_response == True):")
    # print(df_learning["bad_response"].sum())

    # plot_reward_by_trial(df_learning)

    # plot_logistic_regression_per_subject(df_learning)
    # plot_logistic_regression_per_subject(df_awareness)

    # df_awareness = df_awareness.loc[df_awareness["chosen_item"].isin([0, 1])]
    # print(df_awareness["chosen_item"].mean())

    # plot_rt_by_trial(df_learning)

    group1 = "explore-to-exploit"
    group2 = "immediate-exploit"
    group3 = "explore-exploit-cycling"
    group4 = "other"
    group5 = "exploit-to-explore"
    subject_states_list = load_categorized_subjects(SUMMARY_PATH, needed_categories=[group5])
    subjects = [subject for subject, states, state_labels, observations in subject_states_list]
    print(f"\nSubjects included for analysis: {subjects}")
    all_data_practice, all_data_learning, all_data_awareness = load_all_concatenated(
        Path("data_online_experiment"), 
        subjects_include = subjects
        )
    # plot_frac_exploit_vs_valid_choice_rate(SUMMARY_PATH, subjects_include=subjects)

    # subject_probs = compute_exploit_target_prob_by_switch(subject_states_list)
    # plot_exploit_target_prob_by_switch(subject_probs)

    # target_choice_df = compute_target_choice_rate_per_subject(all_data_learning)
    # print("\nTarget choice rate per subject:")
    # print(target_choice_df)

    # target_choice_df = compute_target_choice_rate_per_subject(all_data_awareness)
    # print("\nTarget choice rate per subject:")
    # print(target_choice_df)

    # for subj_id, df in all_data_learning:
    #     print(f"\nSubject ID: {subj_id}")
    #     print(df.loc[df["bad_response"] == True, ["reward_points", "chosen_item"]])
    # chosen_item_stats = summarize_chosen_item_errors(all_data_practice)
    # print("\nChosen item angular error stats (by subject):")
    # print(chosen_item_stats)
    # results = analyze_color_accuracy_change_across_subjects(all_data_learning)
    # print("\nColor accuracy change analysis across subjects:")
    # print(results['significant'])

    # plot_rt_histogram_all_subjects(all_data_learning, bin_ms=500)

    # ooz_labeled = label_if_ooz(all_data_learning)
    # if ooz_labeled:
    #     # for subj_id, df in ooz_labeled:
    #     #     print(f"\nOOZ labeling for subject {subj_id}:")
    #     #     print(f"\nThe percentage of OOZ: {df.loc[df['ooz']== 1, 'ooz'].value_counts() * 2}%")
    #     #     print(f"\nThe mean reward points: {df['reward_points'].mean()}")
    #     plot_MW_vs_reward_across_subjects(ooz_labeled, ooz_index="rt_based", n_trial=TRIALS_PER_SESSION)

    # anova_res = anova_rt_by_chosen_item(all_data_learning)
    # anova_res = anova_reward_by_periods(all_data_learning)
    # anova_res = anova_count_of_target_choice_by_periods(all_data_learning)
    
    # print("\nANOVA: RT by chosen_item")
    # print(anova_res["anova"])
    # print("\nGroup stats (RT by chosen_item)")
    # print(anova_res["group_stats"])


    # results = t_test_reward_points_between_periods(all_data_learning)
    # print("\nT-test: Reward points between early and late periods")
    # print(f"T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    # print(f"Mean (first half): {results['mean_first_half']}, Mean (second half): {results['mean_second_half']}")

    results = t_test_count_target_choice_between_periods(all_data_learning)
    print("\nT-test: Count of target choice between early and late periods")
    print(f"T-statistic: {results['t_stat']}, P-value: {results['p_value']}")
    print(f"Mean (first half): {results['mean_first_half']}, Mean (second half): {results['mean_second_half']}")


    # results = logit_regression(all_data_learning)
    # print("\nLogistic Regression Results:")
    # print(results.keys())

    # results = linear_regression(all_data_awareness)
    # print("\nLinear Regression Results:")
    # print(results.keys())
    
    # hmm_output = fit_hmm_across_subjects(
    #     all_data_learning, 
    #     n_states=2, 
    #     n_iter=100, 
    #     n_init=10, 
    #     save_path=SUMMARY_PATH
    # )

    # hmm_output = load_hmm_summary(SUMMARY_PATH)
    # # 状態ラベルを正規化し、比較可能なサマリーを生成
    # if hmm_output is not None:
    #     normalized_summary_df = relabel_hmm_states(hmm_output)
        
    #     print("\nNormalized HMM Summary (comparable across subjects):")
    #     print(normalized_summary_df.head())
        
    #     # 比較可能なサマリーをCSVに保存
    #     normalized_summary_df.to_csv("hmm_summary_normalized_ver2.0.csv", index=False)

        # # 最初の被験者のプロット（これは生のままでOK）
        # first_subject = list(hmm_output.keys())[0]
        # if first_subject in hmm_output:
        #      plot_hmm_subject_result(first_subject, hmm_output[first_subject])
