from pathlib import Path

from common.config import DATA_PATH
from io_data.load import load_and_prepare, load_all_concatenated
from features.behavior import summarize_chosen_item_errors
from features.lapses import label_if_ooz
from viz.plots import plot_MW_vs_reward_across_subjects, plot_rt_by_trial, plot_rt_histogram_all_subjects
from stats.metrics import anova_rt_by_chosen_item


def run_default():
    df_learning, df_awareness = load_and_prepare(DATA_PATH)
    print(df_learning.loc[:, [
        "rt",
        "num_trial",
        "target_group",
        "angular_error_target",
        "angular_error_distractor",
        "reward_points",
        "chosen_item",
    ]].head())

    # plot_rt_by_trial(df_learning)

    all_data_learning, all_data_awareness = load_all_concatenated(Path("data_online_experiment"))
    if all_data_learning:
        chosen_item_stats = summarize_chosen_item_errors(all_data_learning)
        print("\nChosen item angular error stats (by subject):")
        print(chosen_item_stats)

        # plot_rt_histogram_all_subjects(all_data_learning, bin_ms=200)

        ooz_labeled = label_if_ooz(all_data_learning)
        if ooz_labeled:
            # for subj_id, df in ooz_labeled:
            #     print(f"\nOOZ labeling for subject {subj_id}:")
            #     print(f"\nThe percentage of OOZ: {df.loc[df['ooz']== 1, 'ooz'].value_counts() * 2}%")
            #     print(f"\nThe mean reward points: {df['reward_points'].mean()}")
            plot_MW_vs_reward_across_subjects(ooz_labeled, ooz_index="rt_based")

        anova_res = anova_rt_by_chosen_item(all_data_learning)
        print("\nANOVA: RT by chosen_item")
        print(anova_res["anova"])
        print("\nGroup stats (RT by chosen_item)")
        print(anova_res["group_stats"])
