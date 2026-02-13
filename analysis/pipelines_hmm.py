from pathlib import Path

from common.config import SUMMARY_PATH
from io_data.load import load_all_concatenated, load_categorized_subjects, load_hmm_summary
from stats.models import fit_hmm_across_subjects
from features.behavior import relabel_hmm_states, compute_exploit_target_prob_by_switch
from viz.plots import plot_hmm_subject_result, plot_exploit_target_prob_by_switch


def get_subjects_by_hmm_category(summary_path=SUMMARY_PATH, categories=None):
    if categories is None:
        categories = []
    subject_states_list = load_categorized_subjects(summary_path, needed_categories=categories)
    subjects = [subject for subject, category, states, state_labels, observations in subject_states_list]
    return subjects, subject_states_list


def run_hmm(
    all_data_learning=None,
    train: bool = False,
    categories=None,
    summary_path=SUMMARY_PATH,
    n_states: int = 2,
    n_iter: int = 100,
    n_init: int = 30,
    random_state: int = 42,
):
    if train:
        if all_data_learning is None:
            raise ValueError("all_data_learning is required when train=True")
        hmm_output = fit_hmm_across_subjects(
            all_data_learning,
            n_states=n_states,
            n_iter=n_iter,
            n_init=n_init,
            random_state=random_state,
            save_path=summary_path,
        )
        return hmm_output

    # analysis-only mode
    subjects, subject_states_list = get_subjects_by_hmm_category(summary_path, categories or [])
    if subject_states_list:
        subject_probs = compute_exploit_target_prob_by_switch(subject_states_list)
        plot_exploit_target_prob_by_switch(subject_probs)
    return {"subjects": subjects, "subject_states_list": subject_states_list}
