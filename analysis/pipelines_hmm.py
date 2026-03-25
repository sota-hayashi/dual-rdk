from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from common.config import SUMMARY_PATH, GAUSSIAN_HMM_SUMMARY_PATH
from io_data.load import (
    load_all_concatenated,
    load_categorized_subjects,
    load_hmm_summary,
    load_gaussian_hmm_summary,
)
from stats.models import fit_hmm_across_subjects
from stats.gaussian_hmm import run_gaussian_hmm
from features.behavior import relabel_hmm_states, compute_exploit_target_prob_by_switch
from viz.plots import plot_hmm_subject_result, plot_exploit_target_prob_by_switch


def get_subjects_by_hmm_category(summary_path=SUMMARY_PATH, categories=None):
    if categories is None:
        categories = []
    subject_states_list = load_categorized_subjects(summary_path, needed_categories=categories)
    subjects = [subject for subject, category, states, state_labels, observations in subject_states_list]
    return subjects, subject_states_list


def run_general_hmm(
    all_data_learning=None,
    train: bool = False,
    categories=None,
    summary_path=SUMMARY_PATH,
    n_states: int = 2,
    n_iter: int = 100,
    n_init: int = 30,
    random_state: int = 4,
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
        hmm_normalized_df = relabel_hmm_states(load_hmm_summary(summary_path))
        hmm_normalized_df.to_csv(summary_path.parent / "hmm_summary_normalized_decisive_prob.csv")

    # analysis-only mode
    subjects, subject_states_list = get_subjects_by_hmm_category(summary_path.parent / "hmm_summary_normalized_decisive_prob.csv", categories or [])
    if subject_states_list:
        subject_probs = compute_exploit_target_prob_by_switch(subject_states_list)
        # plot_exploit_target_prob_by_switch(subject_probs)
    return subjects, subject_states_list


def run_gaussian_hmm_pipeline(
    all_data_learning: Optional[List[Tuple[str, object]]] = None,
    train: bool = False,
    summary_path: Path = GAUSSIAN_HMM_SUMMARY_PATH,
    n_init: int = 20,
    n_iter: int = 200,
    tol: float = 1e-4,
) -> List[Dict]:
    """
    Gaussian HMM パイプラインを実行する。

    train=True のとき:
      - all_data_learning（List[Tuple[str, pd.DataFrame]]）に対して
        run_gaussian_hmm() を実行し、結果を summary_path に保存する。

    train=False のとき:
      - 既存の summary_path から結果を読み込んで返す。

    Parameters
    ----------
    all_data_learning : List[Tuple[str, pd.DataFrame]] or None
        (participant_id, df) のリスト。train=True のときに必須。
    train : bool
        True のとき fitting を実行する。False のとき保存済み結果を読み込む。
    summary_path : Path
        結果CSVのパス（デフォルト: GAUSSIAN_HMM_SUMMARY_PATH）
    n_init : int
        グローバルfittingの初期値試行回数（デフォルト: 20）
    n_iter : int
        EMの最大反復回数（デフォルト: 200）
    tol : float
        収束判定の閾値（デフォルト: 1e-4）

    Returns
    -------
    List[Dict]
        参加者ごとの結果辞書のリスト（run_gaussian_hmm() の出力形式）。
        train=False のときは load_gaussian_hmm_summary() で読み込んだ結果を返す。
    """
    if train:
        if all_data_learning is None:
            raise ValueError("all_data_learning is required when train=True")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results = run_gaussian_hmm(
            all_data_learning,
            n_init=n_init,
            n_iter=n_iter,
            tol=tol,
            save_path=str(summary_path),
        )
        return results

    # analysis-only mode: 保存済みCSVから読み込む
    return load_gaussian_hmm_summary(summary_path)

def run_hmm(
    all_data_learning=None,
    train: bool = False,
    ):
    gaussian_hmm_output = run_gaussian_hmm_pipeline(
        all_data_learning=all_data_learning,
        train=train)
    subjects = [output["participant_id"] for output in gaussian_hmm_output]
    subjects_states = [np.mean(output["viterbi_states"]) for output in gaussian_hmm_output]

    # print(f"Subjects included in Gaussian-HMM analysis: {subjects}")
    # print(f"Viterbi states for each subject: {subjects_states}")

    gaussian_hmm_df = pd.DataFrame({
        "subject": subjects,
        "off_rate": subjects_states
    })
    print(gaussian_hmm_df)

    return gaussian_hmm_df
