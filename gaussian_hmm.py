from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm


REQUIRED_COLUMNS = {"participant_id", "trial", "angular_error"}


@dataclass
class GaussianHMMConfig:
    n_components: int = 2
    covariance_type: str = "full"
    n_init: int = 20
    n_iter: int = 200
    tol: float = 1e-4
    random_state_offset: int = 0
    participant_col: str = "participant_id"
    trial_col: str = "trial"
    angular_error_col: str = "angular_error"
    abs_angular_error_col: str = "abs_angular_error"
    dropna: bool = True


class GaussianHMMError(ValueError):
    """Raised when the input data do not satisfy the Gaussian HMM assumptions."""


def _validate_input(df: pd.DataFrame, cfg: GaussianHMMConfig) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise GaussianHMMError(f"Missing required columns: {sorted(missing)}")
    if cfg.n_components != 2:
        raise GaussianHMMError("This implementation currently supports exactly 2 states.")
    if df.empty:
        raise GaussianHMMError("Input DataFrame is empty.")


def prepare_hmm_dataframe(df: pd.DataFrame, cfg: Optional[GaussianHMMConfig] = None) -> pd.DataFrame:
    """
    Validate and prepare the input DataFrame for Gaussian HMM fitting.

    Adds abs_angular_error = |angular_error| and sorts by participant/trial.
    Rows with NaN angular_error are dropped by default.
    """
    cfg = cfg or GaussianHMMConfig()
    _validate_input(df, cfg)

    out = df.copy()
    if cfg.dropna:
        out = out.dropna(subset=[cfg.angular_error_col])

    out[cfg.abs_angular_error_col] = out[cfg.angular_error_col].abs()

    if not np.isfinite(out[cfg.abs_angular_error_col]).all():
        raise GaussianHMMError("abs_angular_error contains non-finite values.")

    if ((out[cfg.abs_angular_error_col] < 0) | (out[cfg.abs_angular_error_col] > 180)).any():
        raise GaussianHMMError("abs_angular_error must be in [0, 180].")

    out = out.sort_values([cfg.participant_col, cfg.trial_col]).reset_index(drop=True)
    return out


def build_global_sequences(
    df: pd.DataFrame,
    cfg: Optional[GaussianHMMConfig] = None,
) -> Tuple[np.ndarray, List[int], List[Any], Dict[Any, np.ndarray]]:
    """
    Build X_all, lengths, participant_ids, and per-participant sequences.

    Returns
    -------
    X_all : ndarray of shape (n_total_trials, 1)
    lengths : list[int]
    participant_ids : list
    sequences : dict[participant_id, ndarray of shape (n_trials_i, 1)]
    """
    cfg = cfg or GaussianHMMConfig()
    prepared = prepare_hmm_dataframe(df, cfg)

    sequences: Dict[Any, np.ndarray] = {}
    participant_ids: List[Any] = []
    lengths: List[int] = []
    X_list: List[np.ndarray] = []

    for pid, g in prepared.groupby(cfg.participant_col, sort=False):
        Xi = g[cfg.abs_angular_error_col].to_numpy(dtype=float).reshape(-1, 1)
        if Xi.shape[0] == 0:
            continue
        participant_ids.append(pid)
        sequences[pid] = Xi
        lengths.append(Xi.shape[0])
        X_list.append(Xi)

    if not X_list:
        raise GaussianHMMError("No valid participant sequences were built.")

    X_all = np.vstack(X_list)
    return X_all, lengths, participant_ids, sequences


def fit_best_global_hmm(
    X_all: np.ndarray,
    lengths: Sequence[int],
    cfg: Optional[GaussianHMMConfig] = None,
) -> Tuple[hmm.GaussianHMM, float]:
    """
    Fit the global Gaussian HMM with multiple random initializations and return the best model.
    """
    cfg = cfg or GaussianHMMConfig()

    best_score = -np.inf
    best_model: Optional[hmm.GaussianHMM] = None

    for seed in range(cfg.n_init):
        model = hmm.GaussianHMM(
            n_components=cfg.n_components,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            tol=cfg.tol,
            random_state=cfg.random_state_offset + seed,
        )
        model.fit(X_all, lengths=list(lengths))
        score = model.score(X_all, lengths=list(lengths))
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise GaussianHMMError("Failed to fit any global Gaussian HMM model.")

    return best_model, float(best_score)


def align_states(means: np.ndarray) -> Dict[int, int]:
    """
    Align state labels so that 0=engaged (smaller mean) and 1=disengaged (larger mean).
    """
    means_flat = np.asarray(means).reshape(-1)
    if means_flat.shape[0] != 2:
        raise GaussianHMMError("align_states expects exactly 2 state means.")

    engaged_idx = int(np.argmin(means_flat))
    disengaged_idx = int(np.argmax(means_flat))
    return {engaged_idx: 0, disengaged_idx: 1}


def fit_individual_hmm(
    X_i: np.ndarray,
    global_model: hmm.GaussianHMM,
    alignment_map: Dict[int, int],
    cfg: Optional[GaussianHMMConfig] = None,
) -> Dict[str, Any]:
    """
    Fit an individual Gaussian HMM initialized from the global model and return aligned results.
    """
    cfg = cfg or GaussianHMMConfig()

    individual_model = hmm.GaussianHMM(
        n_components=cfg.n_components,
        covariance_type=cfg.covariance_type,
        n_iter=cfg.n_iter,
        tol=cfg.tol,
        init_params="",
        params="stmc",
    )
    individual_model.startprob_ = global_model.startprob_.copy()
    individual_model.transmat_ = global_model.transmat_.copy()
    individual_model.means_ = global_model.means_.copy()
    individual_model.covars_ = global_model.covars_.copy()

    individual_model.fit(X_i)
    viterbi_states_raw = individual_model.predict(X_i)
    viterbi_states_aligned = np.vectorize(alignment_map.get)(viterbi_states_raw)

    return {
        "viterbi_states": np.asarray(viterbi_states_aligned, dtype=int),
        "means": individual_model.means_.copy(),
        "covars": individual_model.covars_.copy(),
        "transmat": individual_model.transmat_.copy(),
        "state_labels": {0: "engaged", 1: "disengaged"},
        "log_likelihood": float(individual_model.score(X_i)),
        "model": individual_model,
    }


def fit_gaussian_hmm_pipeline(
    df: pd.DataFrame,
    cfg: Optional[GaussianHMMConfig] = None,
) -> Dict[str, Any]:
    """
    End-to-end implementation of the Gaussian HMM pipeline specified in gaussian_hmm_spec.md.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain participant_id, trial, angular_error.
    cfg : GaussianHMMConfig, optional

    Returns
    -------
    dict with keys:
        - prepared_df
        - global_model
        - global_log_likelihood
        - alignment_map
        - results (list[dict])
    """
    cfg = cfg or GaussianHMMConfig()
    prepared = prepare_hmm_dataframe(df, cfg)
    X_all, lengths, participant_ids, sequences = build_global_sequences(prepared, cfg)
    global_model, global_score = fit_best_global_hmm(X_all, lengths, cfg)
    alignment_map = align_states(global_model.means_)

    results: List[Dict[str, Any]] = []
    for pid in participant_ids:
        Xi = sequences[pid]
        indiv = fit_individual_hmm(Xi, global_model, alignment_map, cfg)
        results.append(
            {
                "participant_id": pid,
                "n_trials": int(Xi.shape[0]),
                "viterbi_states": indiv["viterbi_states"],
                "means": indiv["means"],
                "covars": indiv["covars"],
                "transmat": indiv["transmat"],
                "state_labels": indiv["state_labels"],
                "log_likelihood": indiv["log_likelihood"],
            }
        )

    return {
        "prepared_df": prepared,
        "global_model": global_model,
        "global_log_likelihood": global_score,
        "alignment_map": alignment_map,
        "results": results,
    }


def results_to_dataframe(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert participant-level HMM results to a compact summary DataFrame.
    """
    rows = []
    for r in results:
        means = np.asarray(r["means"]).reshape(-1)
        rows.append(
            {
                "participant_id": r["participant_id"],
                "n_trials": r["n_trials"],
                "engaged_mean": float(means[0]),
                "disengaged_mean": float(means[1]),
                "log_likelihood": r["log_likelihood"],
                "p_engaged_to_engaged": float(r["transmat"][0, 0]),
                "p_engaged_to_disengaged": float(r["transmat"][0, 1]),
                "p_disengaged_to_engaged": float(r["transmat"][1, 0]),
                "p_disengaged_to_disengaged": float(r["transmat"][1, 1]),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "GaussianHMMConfig",
    "GaussianHMMError",
    "prepare_hmm_dataframe",
    "build_global_sequences",
    "fit_best_global_hmm",
    "align_states",
    "fit_individual_hmm",
    "fit_gaussian_hmm_pipeline",
    "results_to_dataframe",
]
