from pathlib import Path

from common.config import DATA_PATH
from io.load import load_and_prepare, load_all_concatenated
from features.behavior import summarize_chosen_item_errors
from viz.plots import plot_mind_wandering_vs_reward


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

    all_data_learning, all_data_awareness = load_all_concatenated(Path("data_online_experiment"))
    if all_data_learning:
        chosen_item_stats = summarize_chosen_item_errors(all_data_learning)
        print("\nChosen item angular error stats (by subject):")
        print(chosen_item_stats)

        result = plot_mind_wandering_vs_reward(all_data_learning)
        print(result)
