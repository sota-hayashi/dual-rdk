from typing import Dict, List, Tuple
import warnings
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from io_data.utils import combine_subjects
from common.config import TRIALS_PER_SESSION


def _map_chosen_item(series: pd.Series) -> np.ndarray:
    """Map chosen_item {-1,0,1} to {0,1,2} for categorical HMM."""
    mapped = series.replace({-1: 0, 0: 1, 1: 2})
    return mapped.to_numpy(dtype=int)


def _forward_backward(y: np.ndarray, A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> Dict[str, np.ndarray]:
    """Scaled forward-backward for categorical HMM."""
    T = len(y)
    K = A.shape[0]
    alpha = np.zeros((T, K))
    beta = np.zeros((T, K))
    scale = np.zeros(T)

    alpha[0] = pi * B[:, y[0]]
    scale[0] = alpha[0].sum()
    if scale[0] == 0:
        scale[0] = 1e-12
    alpha[0] /= scale[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, y[t]]
        scale[t] = alpha[t].sum()
        if scale[t] == 0:
            scale[t] = 1e-12
        alpha[t] /= scale[t]

    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, y[t + 1]] * beta[t + 1])) / scale[t + 1]

    gamma = alpha * beta
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        numer = alpha[t][:, None] * A * (B[:, y[t + 1]] * beta[t + 1])[None, :]
        denom = numer.sum()
        if denom == 0:
            denom = 1e-12
        xi[t] = numer / denom

    loglik = np.sum(np.log(scale))
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "xi": xi, "loglik": loglik}


def _viterbi(y: np.ndarray, A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Viterbi decoding for categorical HMM."""
    T = len(y)
    K = A.shape[0]
    logA = np.log(A + 1e-12)
    logB = np.log(B + 1e-12)
    logpi = np.log(pi + 1e-12)

    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = logpi + logB[:, y[0]]

    for t in range(1, T):
        scores = delta[t - 1][:, None] + logA
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(K)] + logB[:, y[t]]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _mean_run_lengths(states: np.ndarray, n_states: int) -> Dict[int, float]:
    """Compute mean run length per state."""
    if len(states) == 0:
        return {k: np.nan for k in range(n_states)}
    runs = {k: [] for k in range(n_states)}
    current = states[0]
    length = 1
    for s in states[1:]:
        if s == current:
            length += 1
        else:
            runs[current].append(length)
            current = s
            length = 1
    runs[current].append(length)
    return {k: (np.mean(runs[k]) if runs[k] else np.nan) for k in range(n_states)}


def fit_hmm_per_subject(
    df: pd.DataFrame,
    n_states: int = 2,
    n_iter: int = 100,
    n_init: int = 10,
    tol: float = 1e-4,
    random_state: int = 0
) -> Dict[str, object]:
    """
    Fit a 2-state categorical HMM to chosen_item sequence for a single subject.
    Returns params, Viterbi states, and summary metrics.
    """
    if "chosen_item" not in df.columns:
        raise ValueError("DataFrame lacks 'chosen_item' column.")

    work = df.dropna(subset=["chosen_item"]).copy()
    if "num_session" in work.columns and "num_trial" in work.columns:
        work = work.sort_values(["num_session", "num_trial"]).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    y = _map_chosen_item(work["chosen_item"])
    if len(y) < 5:
        raise ValueError("Not enough trials for HMM.")

    rng = np.random.default_rng(random_state)
    best = {"loglik": -np.inf}

    for init_id in range(n_init):
        A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
        B = np.array([[0.34, 0.33, 0.33], [0.2, 0.2, 0.6]], dtype=float)
        pi = np.array([0.5, 0.5], dtype=float)

        A = A + rng.normal(0, 0.02, size=A.shape)
        B = B + rng.normal(0, 0.02, size=B.shape)
        A = np.clip(A, 1e-6, None)
        B = np.clip(B, 1e-6, None)
        A = A / A.sum(axis=1, keepdims=True)
        B = B / B.sum(axis=1, keepdims=True)
        pi = np.clip(pi + rng.normal(0, 0.02, size=pi.shape), 1e-6, None)
        pi = pi / pi.sum()

        prev_ll = -np.inf
        for _ in range(n_iter):
            fb = _forward_backward(y, A, B, pi)
            gamma = fb["gamma"]
            xi = fb["xi"]

            pi = gamma[0]
            A = xi.sum(axis=0)
            A = A / A.sum(axis=1, keepdims=True)

            B = np.zeros_like(B)
            for c in range(B.shape[1]):
                mask = (y == c)
                if mask.any():
                    B[:, c] = gamma[mask].sum(axis=0)
            B = B / B.sum(axis=1, keepdims=True)

            ll = fb["loglik"]
            if np.abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if fb["loglik"] > best["loglik"]:
            best = {"A": A, "B": B, "pi": pi, "loglik": fb["loglik"]}

    A = best["A"]
    B = best["B"]
    pi = best["pi"]
    states = _viterbi(y, A, B, pi)

    # Assign labels based on target (chosen_item==1 -> category 2)
    target_cat = 2
    exploit_state = int(np.argmax(B[:, target_cat]))
    explore_state = 1 - exploit_state
    state_labels = {explore_state: "explore", exploit_state: "exploit"}

    switch_count = int(np.sum(states[1:] != states[:-1]))
    run_lengths = _mean_run_lengths(states, n_states)
    frac_exploit = float(np.mean(states == exploit_state))

    obs = work["chosen_item"].to_numpy()
    state_obs_stats = {}
    for state_id in range(n_states):
        mask = states == state_id
        if not mask.any():
            state_obs_stats[state_id] = {"target": np.nan, "distractor": np.nan, "else": np.nan}
            continue
        subset = obs[mask]
        state_obs_stats[state_id] = {
            "target": float(np.mean(subset == 1)),
            "distractor": float(np.mean(subset == 0)),
            "else": float(np.mean(subset == -1))
        }

    summary = {
        "frac_exploit": frac_exploit,
        "switch_count": switch_count,
        "mean_run_explore": run_lengths.get(explore_state, np.nan),
        "mean_run_exploit": run_lengths.get(exploit_state, np.nan),
    }

    return {
        "A": A,
        "B": B,
        "pi": pi,
        "loglik": best["loglik"],
        "states": states,
        "state_labels": state_labels,
        "summary": summary,
        "observations": obs,
        "mapped_observations": y,
    }


def fit_hmm_across_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_states: int = 2,
    n_iter: int = 100,
    n_init: int = 10,
    random_state: int = 0,
    save_path: str = "hmm_summary.csv"
) -> Dict[str, object]:
    """
    Fit 2-state categorical HMM for each subject in concat_list.
    Returns dict with per-subject results and summary DataFrame.
    """
    results = {}
    summaries = []
    for idx, (subj_id, df) in enumerate(concat_list):
        try:
            res = fit_hmm_per_subject(
                df,
                n_states=n_states,
                n_iter=n_iter,
                n_init=n_init,
                random_state=random_state + idx
            )
            results[subj_id] = res
            summaries.append({
                "subject": subj_id,
                **res["summary"],
                # 行列を文字列に変換して追加
                "A": json.dumps(res["A"].tolist()),
                "B": json.dumps(res["B"].tolist()),
                "pi": json.dumps(res["pi"].tolist()),
                "state_labels": res["state_labels"],
                "states": json.dumps(res["states"].tolist()),
                "observations": json.dumps(res["observations"].tolist()),
                "mapped_observations": json.dumps(res["mapped_observations"].tolist()),
                "loglik": res["loglik"],
            })
        except Exception as e:
            print(f"Skipping HMM for {subj_id}: {e}")
            continue

    summary_df = pd.DataFrame(summaries)
    if save_path is not None:
        summary_df.to_csv(save_path, index=False)
    return {"results": results, "summary": summary_df}


def mixed_learning_across_subjects(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_perm: int = 100,
    random_state: int = 0
) -> Dict[str, object]:
    """
    Mixed model across subjects: reward_points ~ num_trial + (1|subject)
    Permutation p-values via shuffling reward_points.
    """
    combined = combine_subjects(concat_list)
    combined = combined.dropna(subset=["reward_points", "num_trial", "subject"])
    if combined.empty:
        return {"model": None, "permutation": None}

    rng = np.random.default_rng(random_state)
    model = smf.mixedlm("reward_points ~ num_trial", data=combined, groups=combined["subject"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        fit = model.fit(reml=False, maxiter=500, disp=False)
    obs_beta = fit.params

    perm_beta = {k: [] for k in obs_beta.index}
    for _ in range(n_perm):
        perm_data = combined.copy()
        perm_data["reward_points"] = rng.permutation(perm_data["reward_points"])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                perm_fit = smf.mixedlm(
                    "reward_points ~ num_trial",
                    data=perm_data,
                    groups=perm_data["subject"]
                ).fit(reml=False, maxiter=300, disp=False)
            for k in obs_beta.index:
                perm_beta[k].append(perm_fit.params.get(k, np.nan))
        except Exception:
            continue

    perm_stats = {}
    for k, obs in obs_beta.items():
        vals = np.array([v for v in perm_beta[k] if np.isfinite(v)])
        if len(vals) == 0:
            perm_stats[k] = {"p_perm": np.nan, "n_perm": 0}
            continue
        extreme = np.sum(np.abs(vals) >= abs(obs))
        p_perm = (extreme + 1) / (len(vals) + 1)
        perm_stats[k] = {"p_perm": p_perm, "n_perm": len(vals)}

    return {
        "model_params": obs_beta.to_dict(),
        "model_tvalues": fit.tvalues.to_dict(),
        "permutation": perm_stats
    }

def logit_regression(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_trial: int = TRIALS_PER_SESSION // 3
) -> Dict[str, object]:
    """
    Logistic regression to predict target choice (0/1) from trial number.
    """
    results = {}
    for subj_id, df in concat_list:
        df = df.dropna(subset=["chosen_item", "num_trial", "rt"])
        df = df[df["chosen_item"].isin([0, 1])].copy()
        # df = df[df["num_trial"] > 17]
        if df.empty:
            continue
        X = df[["num_trial"]]
        X = sm.add_constant(X)
        y = df["chosen_item"]

        model = sm.Logit(y, X)
        result = model.fit(disp=False)

        params = result.params
        pvalues = result.pvalues

        if pvalues.get('num_trial', 1.0) < 0.05:
            results[subj_id] = {
                "intercept": params.get('const'),
                "coef_num_trial": params.get('num_trial'),
                "p_intercept": pvalues.get('const'),
                "p_num_trial": pvalues.get('num_trial'),
                "model": result
            }
    return results

def linear_regression(
    concat_list: List[Tuple[str, pd.DataFrame]]
) -> Dict[str, object]:
    """
    Simple linear regression to predict reward_points from trial number for each subject.
    """
    results = {}
    for subj_id, df in concat_list:
        df = df.dropna(subset=["reward_points", "num_trial"]).copy()
        if df.empty:
            continue
            
        X = df[["num_trial"]]
        X = sm.add_constant(X)
        y = df["reward_points"]

        model = sm.OLS(y, X)
        result = model.fit()

        params = result.params
        pvalues = result.pvalues

        if pvalues.get('num_trial', 1.0) < 0.05:
            results[subj_id] = {
                "intercept": params.get('const'),
                "coef_num_trial": params.get('num_trial'),
                "p_intercept": pvalues.get('const'),
                "p_num_trial": pvalues.get('num_trial'),
                "model": result
            }
    return results
