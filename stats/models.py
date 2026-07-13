from typing import Dict, List, Tuple, Literal, Optional, Union
import warnings
import json

import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from io_data.utils import combine_subjects
from common.config import TRIALS_PER_SESSION
from stats.q_learning_bayesian import fit_q_learning_bayesian
from stats.q_learning_map import fit_q_learning_map, predict_target_choice_probs, _choice_determine
from stats.q_learning_hierarchical_bayesian import (
    fit_q_learning_hierarchical_bayesian,
    predict_target_choice_probs as predict_hier_choice_probs,
)
from stats.q_learning_ooz_map import (
    fit_q_learning_ooz_map,
    predict_target_choice_probs as predict_ooz_choice_probs,
)
from stats.rw_hierarchical_bayesian import fit_rw_hierarchical_bayesian
from stats.von_mises_value_learning_map import fit_von_mises_map, predict_target_choice_von_mises
from stats.directional_q_learning_map import (
    fit_directional_q_learning_map,
    predict_target_choice_probs as predict_directional_choice_probs,
    _prepare_subject as _prepare_directional_subject,
)
from stats.von_mises_value_learning_map import _make_mus, _basis_vec, _value


def _map_chosen_item(series: pd.Series) -> np.ndarray:
    """Map chosen_item {-1,0,1} to {0,1,1} for categorical HMM."""
    mapped = series.replace({-1: 0, 0: 1, 1: 1})
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
        A = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)
        B = np.array([[0.04, 0.96], [0.01, 0.99]], dtype=float)
        pi = np.array([0.5, 0.5], dtype=float)

        # A = A + rng.normal(0, 1e-3, size=A.shape)
        # B = B + rng.normal(0, 1e-3, size=B.shape)
        # A = np.clip(A, 1e-6, None)
        # B = np.clip(B, 1e-6, None)
        # A = A / A.sum(axis=1, keepdims=True)
        # B = B / B.sum(axis=1, keepdims=True)
        # pi = np.clip(pi + rng.normal(0, 1e-3, size=pi.shape), 1e-6, None)
        # pi = pi / pi.sum()

        prev_ll = -np.inf
        for _ in range(n_iter):
            fb = _forward_backward(y, A, B, pi)
            gamma = fb["gamma"]
            xi = fb["xi"]

            pi = gamma[0]
            A = xi.sum(axis=0)
            A = A / A.sum(axis=1, keepdims=True)

            # B = np.zeros_like(B)
            # for c in range(B.shape[1]):
            #     mask = (y == c)
            #     if mask.any():
            #         B[:, c] = gamma[mask].sum(axis=0)
            # # eps = 1e-4
            # # B = (B + eps) / (B + eps).sum(axis=1, keepdims=True)
            # B = B / B.sum(axis=1, keepdims=True)

            ll = fb["loglik"]
            if np.abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if fb["loglik"] > best["loglik"]:
            best = {"A": A, "B": B, "pi": pi, "loglik": fb["loglik"]}

    np.set_printoptions(precision=20, suppress=True)
    A = best["A"]
    B = best["B"]
    print("B:", B)
    pi = best["pi"]
    states = _viterbi(y, A, B, pi)

    big_AE_cat = 0
    explore_state = int(np.argmax(B[:, big_AE_cat]))
    exploit_state = 1 - explore_state
    state_labels = {explore_state: "explore", exploit_state: "exploit"}

    switch_count = int(np.sum(states[1:] != states[:-1]))
    run_lengths = _mean_run_lengths(states, n_states)
    frac_exploit = float(np.mean(states == exploit_state))

    obs = work["chosen_item"].to_numpy()
    state_obs_stats = {}
    for state_id in range(n_states):
        mask = states == state_id
        if not mask.any():
            state_obs_stats[state_id] = {"small_AE": np.nan, "big_AE": np.nan}
            continue
        subset = obs[mask]
        state_obs_stats[state_id] = {
            "small_AE": float(np.mean((subset == 1) | (subset == 0))),
            "big_AE": float(np.mean(subset == -1))
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
    save_path: str = "hmm_summary_divided_by_AE.csv"
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
    return summary_df


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
        df["chosen_item"] = df["chosen_item"].replace({-1: 0})
        # df = df[df["chosen_item"].isin([0, 1])].copy()
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

def test_interaction_alpha_diff_ooz_rate(
    all_df: pd.DataFrame,
) -> Dict[str, object]:
    """alpha_diff (= alpha_1 - alpha_0) ~ rt_cv * ooz_rate の交互作用回帰を実行する。

    Parameters
    ----------
    all_df : pd.DataFrame
        alpha_0, alpha_1, rt_cv カラムを含む DataFrame。
        ooz_rate が未計算の場合は ooz カラム（リスト）から自動計算する。

    Returns
    -------
    dict
        interaction_model, main_model, params, pvalues, tvalues,
        r_squared, n, interaction_pvalue を含む。
    """
    df = all_df.copy()

    if "alpha_diff" not in df.columns:
        df["alpha_diff"] = df["alpha_1"] - df["alpha_0"]

    if "ooz_rate" not in df.columns and "ooz" in df.columns:
        df["ooz_rate"] = df["ooz"].apply(
            lambda x: float(np.mean(x)) if isinstance(x, (list, np.ndarray)) else np.nan
        )

    df = df.dropna(subset=["alpha_diff", "rt_cv", "ooz_rate"])
    n = len(df)

    result_interaction = smf.ols("alpha_diff ~ rt_cv * ooz_rate", data=df).fit()
    result_main        = smf.ols("alpha_diff ~ rt_cv + ooz_rate", data=df).fit()

    return {
        "interaction_model":   result_interaction,
        "main_model":          result_main,
        "params":              result_interaction.params.to_dict(),
        "pvalues":             result_interaction.pvalues.to_dict(),
        "tvalues":             result_interaction.tvalues.to_dict(),
        "r_squared":           result_interaction.rsquared,
        "r_squared_adj":       result_interaction.rsquared_adj,
        "n":                   n,
        "interaction_pvalue":  result_interaction.pvalues.get("rt_cv:ooz_rate", np.nan),
    }


def evaluate_q_learning(
    method: Literal["map", "hierarchical_bayesian", "ooz_map"],
    concat_list: List[Tuple[str, pd.DataFrame]],
    results_df: pd.DataFrame,
    n_params: int = 3,
) -> pd.DataFrame:
    """推定パラメータに対する適合度指標を計算する。

    Parameters
    ----------
    method : {"map", "hierarchical_bayesian", "ooz_map"}
        "map"                  : results_df は alpha, beta, c_black カラムを含む。
        "hierarchical_bayesian": results_df は alpha_mean, beta_mean, c_black_mean カラムを含む。
        "ooz_map"              : results_df は alpha_1, beta, c_black カラムを含む。
                                 df に ooz カラムが必要。alpha_0=0 固定。n_params=3 を推奨。
    concat_list : list of (subject, DataFrame)
        行動データ。
    results_df : pd.DataFrame
        推定結果（subject, n_trials カラムが必要）。
    n_params : int
        モデルのパラメータ数（map/hierarchical_bayesian: 3, ooz_map: 4）。

    Returns
    -------
    pd.DataFrame
        subject, log_likelihood, ll_null, pseudo_r2, aic, bic,
        choice_accuracy, n_trials カラムを含む。
    """
    if method == "map":
        alpha_col, beta_col, c_col = "alpha", "beta", "c_black"
    elif method == "hierarchical_bayesian":
        alpha_col, beta_col, c_col = "alpha_mean", "beta_mean", "c_black_mean"
    elif method == "ooz_map":
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

    param_map = results_df.set_index("subject").to_dict("index")
    records = []

    for subj_id, df in concat_list:
        if subj_id not in param_map:
            continue

        row      = param_map[subj_id]
        n_trials = row["n_trials"]

        if method == "ooz_map":
            alpha_0    = row["alpha_0"]
            alpha_1    = row["alpha_1"]
            beta       = row["beta"]
            c_black    = row["c_black"]
            df_clean   = df.dropna(subset=["rt", "chosen_color", "ooz"]).copy()
            ooz_labels = df_clean["ooz"].values.astype(int)
        else:
            alpha    = row[alpha_col]
            beta     = row[beta_col]
            c_black  = row[c_col]
            df_clean = df.dropna(subset=["rt", "chosen_color"]).copy()

        choices = _choice_determine(df_clean["chosen_color"])
        states  = df_clean["target_item"].values.astype(int)
        rewards = (df_clean["reward_points"].values / 10).astype(float)

        q = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
        ll      = 0.0
        correct = 0
        for t in range(len(choices)):
            s  = int(states[t])
            dq = beta * (q[s, 1] - q[s, 0]) + c_black
            dq = np.clip(dq, -500, 500)
            if s == 0:
                p_target      = 1.0 / (1.0 + np.exp(dq))
                target_action = 0
            else:
                p_target      = 1.0 / (1.0 + np.exp(-dq))
                target_action = 1

            a = int(choices[t])
            p_chosen = p_target if a == target_action else (1.0 - p_target)
            ll += np.log(p_chosen + 1e-300)

            if (p_target > 0.5) == (a == target_action):
                correct += 1

            alpha_t = (alpha_0 if ooz_labels[t] == 0 else alpha_1) if method == "ooz_map" else alpha
            q[s, a] += alpha_t * (rewards[t] - q[s, a])

        n         = len(choices)
        ll_null   = n * np.log(0.5)
        pseudo_r2 = 1.0 - ll / ll_null
        aic       = -2 * ll + 2 * n_params
        bic       = -2 * ll + n_params * np.log(n)
        choice_accuracy = correct / n if n > 0 else np.nan

        records.append({
            "subject":         subj_id,
            "log_likelihood":  ll,
            "ll_null":         ll_null,
            "pseudo_r2":       pseudo_r2,
            "aic":             aic,
            "bic":             bic,
            "choice_accuracy": choice_accuracy,
            "n_trials":        n_trials,
        })

    return pd.DataFrame(records)


def evaluate_directional_q_learning(
    concat_list: List[Tuple[str, pd.DataFrame]],
    results_df: pd.DataFrame,
    n_params: int = 4,
    N: int = 4,
    kappa: float = 2.0,
) -> pd.DataFrame:
    """色別方向価値Q学習（OOZ依存学習率）の適合度指標を計算する。

    evaluate_q_learning と同じ指標（log_likelihood, pseudo_r2, AIC, BIC,
    choice_accuracy）を，directional_q_learning_map の尤度ロジックで算出する。
    選択は色（black/white）に対して直接定義される:

        dq = beta * (V_black(θ_black) - V_white(θ_white)) + c_black
        P(black) = sigmoid(dq)

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        df は rt, chosen_color, response_angle_css, target_direction,
        distractor_direction, target_group, reward_points, ooz を含む。
    results_df : pd.DataFrame
        fit_directional_q_learning_map の出力（subject, alpha_0, alpha_1,
        beta, c_black, n_trials カラムが必要。N, kappa があればそれを優先使用）。
    n_params : int
        モデルのパラメータ数（alpha_0, alpha_1, beta, c_black の 4）。
    N, kappa : int, float
        基底パラメータ。results_df に N/kappa 列が無い場合のフォールバック。

    Returns
    -------
    pd.DataFrame
        subject, log_likelihood, ll_null, pseudo_r2, aic, bic,
        choice_accuracy, n_trials カラムを含む。
    """
    param_map = results_df.set_index("subject").to_dict("index")
    records = []

    for subj_id, df in concat_list:
        if subj_id not in param_map:
            continue

        row     = param_map[subj_id]
        alpha_0 = row["alpha_0"]
        alpha_1 = row["alpha_1"]
        beta    = row["beta"]
        c_black = row["c_black"]
        N_subj     = int(row["N"]) if "N" in row and not pd.isna(row["N"]) else N
        kappa_subj = float(row["kappa"]) if "kappa" in row and not pd.isna(row["kappa"]) else kappa

        data = _prepare_directional_subject(df)
        if data is None:
            continue

        mus = _make_mus(N_subj)
        W_white = np.zeros(N_subj)
        W_black = np.zeros(N_subj)

        choices     = data["choices"]
        rewards     = data["rewards"]
        ooz         = data["ooz"]
        theta_white = data["theta_white"]
        theta_black = data["theta_black"]
        reported    = data["reported"]

        ll      = 0.0
        correct = 0
        for t in range(len(choices)):
            v_white = _value(theta_white[t], W_white, mus, kappa_subj)
            v_black = _value(theta_black[t], W_black, mus, kappa_subj)

            dq = np.clip(beta * (v_black - v_white) + c_black, -500, 500)
            p_black = 1.0 / (1.0 + np.exp(-dq))

            a = int(choices[t])  # 1=black, 0=white
            p_chosen = p_black if a == 1 else (1.0 - p_black)
            ll += np.log(p_chosen + 1e-300)

            if (p_black > 0.5) == (a == 1):
                correct += 1

            alpha = alpha_0 if ooz[t] == 0 else alpha_1
            phi_resp = _basis_vec(reported[t], mus, kappa_subj)
            if a == 1:
                v_resp = float(np.dot(W_black, phi_resp))
                W_black += alpha * (rewards[t] - v_resp) * phi_resp
            else:
                v_resp = float(np.dot(W_white, phi_resp))
                W_white += alpha * (rewards[t] - v_resp) * phi_resp

        n         = len(choices)
        ll_null   = n * np.log(0.5)
        pseudo_r2 = 1.0 - ll / ll_null
        aic       = -2 * ll + 2 * n_params
        bic       = -2 * ll + n_params * np.log(n)
        choice_accuracy = correct / n if n > 0 else np.nan

        records.append({
            "subject":         subj_id,
            "log_likelihood":  ll,
            "ll_null":         ll_null,
            "pseudo_r2":       pseudo_r2,
            "aic":             aic,
            "bic":             bic,
            "choice_accuracy": choice_accuracy,
            "n_trials":        row.get("n_trials", n),
        })

    return pd.DataFrame(records)


def run_q_learning(
    method: Literal["map", "bayesian", "hierarchical_bayesian"] = "map",
    fit: bool = True,
    concat_list: Optional[List[Tuple[str, pd.DataFrame]]] = None,
    result_dir: str = "results/q_learning",
    # --- MAP固有パラメータ ---
    n_alpha_grid: int = 5,
    n_beta_grid: int = 5,
    # --- Bayesian固有パラメータ ---
    nwalkers: int = 32,
    nburn: int = 1000,
    nsamples: int = 2000,
    n_rhat_chains: int = 4,
    random_seed: int = 0,
    # --- hierarchical_bayesian 固有パラメータ ---
    hier_nwalkers: int = None,
    hier_nburn: int = 2000,
    hier_nsamples: int = 4000,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, Dict]]:
    """Q学習モデルのフィッティング実行または結果読み込み。

    Parameters
    ----------
    method : {"map", "bayesian", "hierarchical_bayesian"}
        推定方法。
    fit : bool
        True: フィッティングを実行し結果を保存。
        False: result_dir から既存結果を読み込み。
    concat_list : list of (str, DataFrame), optional
        fit=True の場合に必要。各参加者の行動データ。
    result_dir : str
        結果の保存先・読み込み元ディレクトリ。
    n_alpha_grid, n_beta_grid : int
        MAP推定のグリッドサーチ分割数。
    nwalkers, nburn, nsamples, n_rhat_chains, random_seed : int
        ベイズ推定のMCMCパラメータ。
    hier_nwalkers, hier_nburn, hier_nsamples : int
        階層ベイズ推定のMCMCパラメータ。hier_nwalkers=None で自動設定。

    Returns
    -------
    method="map" の場合:
        pd.DataFrame — 推定結果
    method="bayesian" の場合:
        tuple of (pd.DataFrame, dict) — 推定結果とトレース
    method="hierarchical_bayesian" の場合:
        tuple of (group_results: pd.DataFrame, individual_results: pd.DataFrame, traces: dict)
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if fit:
        return _fit(
            method=method,
            concat_list=concat_list,
            result_dir=result_dir,
            n_alpha_grid=n_alpha_grid,
            n_beta_grid=n_beta_grid,
            nwalkers=nwalkers,
            nburn=nburn,
            nsamples=nsamples,
            n_rhat_chains=n_rhat_chains,
            random_seed=random_seed,
            hier_nwalkers=hier_nwalkers,
            hier_nburn=hier_nburn,
            hier_nsamples=hier_nsamples,
        )
    else:
        return _load(method=method, result_dir=result_dir)


def _fit(
    method: str,
    concat_list: List[Tuple[str, pd.DataFrame]],
    result_dir: Path,
    **kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """フィッティングを実行し、結果を保存する。"""
    if concat_list is None:
        raise ValueError("fit=True の場合、concat_list を指定してください。")

    if method == "map":
        results_df = fit_q_learning_map(
            concat_list,
            n_alpha_grid=kwargs["n_alpha_grid"],
            n_beta_grid=kwargs["n_beta_grid"],
            output_path=str(result_dir / "map_results.csv"),
        )
        trial_results = predict_target_choice_probs(
            concat_list,
            results_df,
            output_path=str(result_dir / "trial_results_map.csv"),
        )
        return results_df, trial_results

    elif method == "bayesian":
        results_df, traces = fit_q_learning_bayesian(
            concat_list,
            nwalkers=kwargs["nwalkers"],
            nburn=kwargs["nburn"],
            nsamples=kwargs["nsamples"],
            n_rhat_chains=kwargs["n_rhat_chains"],
            random_seed=kwargs["random_seed"],
            output_path=str(result_dir / "bayesian_results.csv"),
        )
        # トレースをnpzで保存
        import numpy as np
        np.savez(
            result_dir / "bayesian_traces.npz",
            subj_ids=traces["subject"],
            alpha_samples=traces["alpha_samples"],
            beta_samples=traces["beta_samples"],
        )
        return results_df, traces

    elif method == "hierarchical_bayesian":
        import numpy as np
        group_df, ind_df, traces = fit_q_learning_hierarchical_bayesian(
            concat_list,
            nwalkers=kwargs["hier_nwalkers"],
            nburn=kwargs["hier_nburn"],
            nsamples=kwargs["hier_nsamples"],
            n_rhat_chains=kwargs.get("n_rhat_chains", 4),
            random_seed=kwargs.get("random_seed", 0),
            output_path=str(result_dir / "hierarchical_bayesian_individual.csv"),
        )
        group_df.to_csv(result_dir / "hierarchical_bayesian_group.csv", index=False)
        np.savez(
            result_dir / "hierarchical_bayesian_traces.npz",
            subj_ids=np.array(traces["subj_ids"], dtype=object),
            alpha_samples=traces["alpha_samples"],
            beta_samples=traces["beta_samples"],
            c_samples=traces["c_samples"],
            mu_alpha_samples=traces["mu_alpha_samples"],
            sigma_alpha_samples=traces["sigma_alpha_samples"],
            mu_beta_samples=traces["mu_beta_samples"],
            sigma_beta_samples=traces["sigma_beta_samples"],
            mu_c_samples=traces["mu_c_samples"],
            sigma_c_samples=traces["sigma_c_samples"],
        )
        trial_results = predict_hier_choice_probs(
            concat_list,
            ind_df,
            output_path=str(result_dir / "trial_results_hierarchical_bayesian.csv"),
        )
        return ind_df, trial_results

    else:
        raise ValueError(f"未知の method: {method}（'map', 'bayesian', 'hierarchical_bayesian'）")


def _load(
    method: str,
    result_dir: Path,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """保存済みの結果を読み込む。"""
    if method == "map":
        individual_path = result_dir / "map_results.csv"
        trial_path     = result_dir / "trial_results_map.csv"
        if not individual_path.exists():
            raise FileNotFoundError(f"MAP結果が見つかりません: {individual_path}")
        if not trial_path.exists():
            raise FileNotFoundError(f"MAP試行結果が見つかりません: {trial_path}")
        return pd.read_csv(individual_path), pd.read_csv(trial_path)

    elif method == "bayesian":
        csv_path = result_dir / "bayesian_results.csv"
        npz_path = result_dir / "bayesian_traces.npz"

        if not csv_path.exists():
            raise FileNotFoundError(f"ベイズ推定結果が見つかりません: {csv_path}")

        results_df = pd.read_csv(csv_path)

        traces = None
        if npz_path.exists():
            import numpy as np
            data = np.load(npz_path, allow_pickle=True)
            traces = {
                "subject": data["subj_ids"].tolist(),
                "alpha_samples": data["alpha_samples"],
                "beta_samples": data["beta_samples"],
            }

        return results_df, traces

    elif method == "hierarchical_bayesian":
        ind_path   = result_dir / "hierarchical_bayesian_individual.csv"
        trial_path = result_dir / "trial_results_hierarchical_bayesian.csv"

        if not ind_path.exists():
            raise FileNotFoundError(f"階層ベイズ個人結果が見つかりません: {ind_path}")
        if not trial_path.exists():
            raise FileNotFoundError(f"階層ベイズ試行結果が見つかりません: {trial_path}")

        ind_df        = pd.read_csv(ind_path)
        trial_results = pd.read_csv(trial_path)

        return ind_df, trial_results

    else:
        raise ValueError(f"未知の method: {method}（'map', 'bayesian', 'hierarchical_bayesian'）")


def run_q_learning_ooz(
    fit: bool = True,
    concat_list: Optional[List[Tuple[str, pd.DataFrame]]] = None,
    result_dir: str = "results/q_learning_ooz",
    n_alpha0_grid: int = 5,
    n_alpha1_grid: int = 5,
    n_beta_grid: int = 5,
    n_c_grid: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """OOZ状態依存学習率 Q学習モデルの推定実行または結果読み込み。

    Parameters
    ----------
    fit : bool
        True: フィッティングを実行し結果を保存。
        False: result_dir から既存結果を読み込み。
    concat_list : list of (str, DataFrame), optional
        fit=True の場合に必要。各 DataFrame に label_if_ooz で付与した ooz カラムが必要。
    result_dir : str
        結果の保存先・読み込み元ディレクトリ。
    n_alpha0_grid, n_alpha1_grid, n_beta_grid, n_c_grid : int
        グリッドサーチの分割数（合計 n_a0 × n_a1 × n_b × n_c 点）。

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (results_df, trial_results)
        results_df   : subject, alpha_0, alpha_1, beta, c_black,
                       log_likelihood, n_trials, n_ooz, n_non_ooz
        trial_results: 試行レベルの予測確率（ooz カラムを含む）
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if fit:
        if concat_list is None:
            raise ValueError("fit=True の場合、concat_list を指定してください。")
        results_df = fit_q_learning_ooz_map(
            concat_list,
            n_alpha0_grid=n_alpha0_grid,
            n_alpha1_grid=n_alpha1_grid,
            n_beta_grid=n_beta_grid,
            n_c_grid=n_c_grid,
            output_path=str(result_dir / "ooz_map_results.csv"),
        )
        trial_results = predict_ooz_choice_probs(
            concat_list,
            results_df,
            output_path=str(result_dir / "trial_results_ooz_map.csv"),
        )
        return results_df, trial_results
    else:
        results_path = result_dir / "ooz_map_results.csv"
        trial_path   = result_dir / "trial_results_ooz_map.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"OOZ MAP結果が見つかりません: {results_path}")
        if not trial_path.exists():
            raise FileNotFoundError(f"OOZ MAP試行結果が見つかりません: {trial_path}")
        return pd.read_csv(results_path), pd.read_csv(trial_path)


def run_directional_q_learning(
    fit: bool = True,
    concat_list: Optional[List[Tuple[str, pd.DataFrame]]] = None,
    result_dir: str = "results/directional_q_learning",
    N: int = 4,
    kappa: float = 2.0,
    n_alpha0_grid: int = 4,
    n_alpha1_grid: int = 4,
    n_beta_grid: int = 4,
    n_c_grid: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """色別方向価値Q学習モデル（OOZ依存学習率）の推定実行または結果読み込み。

    Parameters
    ----------
    fit : bool
        True: フィッティングを実行し結果を保存。False: result_dir から読み込み。
    concat_list : list of (str, DataFrame), optional
        fit=True の場合に必要。各 df は rt, chosen_color, response_angle_css,
        target_direction, distractor_direction, target_group, reward_points, ooz を含む。
    result_dir : str
        結果の保存先・読み込み元ディレクトリ。
    N, kappa : int, float
        von Mises 基底のパラメータ。
    n_alpha0_grid, n_alpha1_grid, n_beta_grid, n_c_grid : int
        グリッドサーチの分割数。

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (results_df, trial_results)
        results_df   : subject, alpha_0, alpha_1, beta, c_black, log_posterior,
                       log_likelihood, n_trials, n_ooz, n_non_ooz, N, kappa
        trial_results: 試行レベルの予測確率（p_target を含む）
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if fit:
        if concat_list is None:
            raise ValueError("fit=True の場合、concat_list を指定してください。")
        results_df = fit_directional_q_learning_map(
            concat_list,
            N=N,
            kappa=kappa,
            n_alpha0_grid=n_alpha0_grid,
            n_alpha1_grid=n_alpha1_grid,
            n_beta_grid=n_beta_grid,
            n_c_grid=n_c_grid,
            output_path=str(result_dir / "directional_map_results.csv"),
        )
        trial_results = predict_directional_choice_probs(
            concat_list,
            results_df,
            N=N,
            kappa=kappa,
            output_path=str(result_dir / "trial_results_directional_map.csv"),
        )
        return results_df, trial_results
    else:
        results_path = result_dir / "directional_map_results.csv"
        trial_path   = result_dir / "trial_results_directional_map.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"方向価値Q学習結果が見つかりません: {results_path}")
        if not trial_path.exists():
            raise FileNotFoundError(f"方向価値Q学習試行結果が見つかりません: {trial_path}")
        return pd.read_csv(results_path), pd.read_csv(trial_path)


def run_rw_learning(
    representation: Literal["discrete", "continuous"] = "continuous",
    fit: bool = True,
    concat_list: Optional[List[Tuple[str, pd.DataFrame]]] = None,
    result_dir: str = "results/rw_learning",
    # --- discrete（階層ベイズ）固有パラメータ ---
    nwalkers: int = None,
    nburn: int = 2000,
    nsamples: int = 4000,
    n_rhat_chains: int = 4,
    random_seed: int = 0,
    # --- continuous（von Mises MAP）固有パラメータ ---
    N: int = 6,
    kappa: float = 2.0,
    n_alpha_grid: int = 5,
    n_beta_grid: int = 5,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, Dict],          # discrete
    Tuple[pd.DataFrame, Optional[pd.DataFrame]],      # continuous
]:
    """RW（Rescorla-Wagner）モデルの推定実行または結果読み込み。

    Parameters
    ----------
    representation : {"discrete", "continuous"}
        "discrete"  : 色カテゴリ（white/black）への連合強度を階層ベイズで推定。
        "continuous": 連続方向空間上の von Mises 基底価値学習を MAP 推定。
    fit : bool
        True: フィッティングを実行し結果を保存。
        False: result_dir から既存結果を読み込み。
    concat_list : list of (str, DataFrame), optional
        fit=True の場合に必要。
        discrete: rt, chosen_color, reward_points カラムが必要。
        continuous: rt, response_angle_css, stimulus_angle_1, stimulus_angle_2,
                    reward_points カラムが必要。target_direction / distractor_direction
                    カラムが存在すれば事後予測（trial_results）も計算する。
    result_dir : str
        結果の保存先・読み込み元ディレクトリ。representation ごとのサブディレクトリを自動作成。
    nwalkers, nburn, nsamples, n_rhat_chains, random_seed :
        discrete モード用 MCMC パラメータ。
    N, kappa, n_alpha_grid, n_beta_grid :
        continuous モード用 von Mises パラメータ。

    Returns
    -------
    representation="discrete" の場合:
        tuple of (group_results: pd.DataFrame, individual_results: pd.DataFrame, traces: dict)
    representation="continuous" の場合:
        tuple of (results: pd.DataFrame, trial_results: pd.DataFrame or None)
    """
    sub = Path(result_dir) / representation
    sub.mkdir(parents=True, exist_ok=True)

    if representation == "discrete":
        if fit:
            return _fit_rw_discrete(
                concat_list=concat_list,
                result_dir=sub,
                nwalkers=nwalkers,
                nburn=nburn,
                nsamples=nsamples,
                n_rhat_chains=n_rhat_chains,
                random_seed=random_seed,
            )
        else:
            return _load_rw_discrete(result_dir=sub)

    elif representation == "continuous":
        if fit:
            return _fit_rw_continuous(
                concat_list=concat_list,
                result_dir=sub,
                N=N,
                kappa=kappa,
                n_alpha_grid=n_alpha_grid,
                n_beta_grid=n_beta_grid,
            )
        else:
            return _load_rw_continuous(result_dir=sub)

    else:
        raise ValueError(f"未知の representation: {representation}（'discrete' or 'continuous'）")


# ---------------------------------------------------------------------------
# discrete（階層ベイズ）ヘルパー
# ---------------------------------------------------------------------------

def _fit_rw_discrete(
    concat_list: List[Tuple[str, pd.DataFrame]],
    result_dir: Path,
    nwalkers: int,
    nburn: int,
    nsamples: int,
    n_rhat_chains: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if concat_list is None:
        raise ValueError("fit=True の場合、concat_list を指定してください。")

    group_df, ind_df, traces = fit_rw_hierarchical_bayesian(
        concat_list,
        nwalkers=nwalkers,
        nburn=nburn,
        nsamples=nsamples,
        n_rhat_chains=n_rhat_chains,
        random_seed=random_seed,
        output_path=str(result_dir / "individual.csv"),
    )
    group_df.to_csv(result_dir / "group.csv", index=False)
    np.savez(
        result_dir / "traces.npz",
        subj_ids=np.array(traces["subj_ids"], dtype=object),
        alpha_samples=traces["alpha_samples"],
        beta_samples=traces["beta_samples"],
        mu_alpha_samples=traces["mu_alpha_samples"],
        sigma_alpha_samples=traces["sigma_alpha_samples"],
        mu_beta_samples=traces["mu_beta_samples"],
        sigma_beta_samples=traces["sigma_beta_samples"],
    )
    return group_df, ind_df, traces


def _load_rw_discrete(result_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    group_path = result_dir / "group.csv"
    ind_path   = result_dir / "individual.csv"
    npz_path   = result_dir / "traces.npz"

    if not group_path.exists():
        raise FileNotFoundError(f"discrete RW 集団結果が見つかりません: {group_path}")
    if not ind_path.exists():
        raise FileNotFoundError(f"discrete RW 個人結果が見つかりません: {ind_path}")

    group_df = pd.read_csv(group_path)
    ind_df   = pd.read_csv(ind_path)

    traces = None
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        traces = {
            "subj_ids":            data["subj_ids"].tolist(),
            "alpha_samples":       data["alpha_samples"],
            "beta_samples":        data["beta_samples"],
            "mu_alpha_samples":    data["mu_alpha_samples"],
            "sigma_alpha_samples": data["sigma_alpha_samples"],
            "mu_beta_samples":     data["mu_beta_samples"],
            "sigma_beta_samples":  data["sigma_beta_samples"],
        }

    return group_df, ind_df, traces


# ---------------------------------------------------------------------------
# continuous（von Mises MAP）ヘルパー
# ---------------------------------------------------------------------------

def _fit_rw_continuous(
    concat_list: List[Tuple[str, pd.DataFrame]],
    result_dir: Path,
    N: int,
    kappa: float,
    n_alpha_grid: int,
    n_beta_grid: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if concat_list is None:
        raise ValueError("fit=True の場合、concat_list を指定してください。")

    results_df = fit_von_mises_map(
        concat_list,
        N=N,
        kappa=kappa,
        n_alpha_grid=n_alpha_grid,
        n_beta_grid=n_beta_grid,
        output_path=str(result_dir / "results.csv"),
    )

    # target_direction / distractor_direction が存在すれば事後予測も実行
    has_target_cols = all(
        "target_direction" in df.columns and "distractor_direction" in df.columns
        for _, df in concat_list
    )
    trial_results = None
    if has_target_cols:
        trial_results = predict_target_choice_von_mises(
            concat_list,
            results_df,
            N=N,
            kappa=kappa,
            output_path=str(result_dir / "trial_results.csv"),
        )

    return results_df, trial_results


def _load_rw_continuous(result_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    results_path = result_dir / "results.csv"
    trial_path   = result_dir / "trial_results.csv"

    if not results_path.exists():
        raise FileNotFoundError(f"continuous RW 結果が見つかりません: {results_path}")

    results_df    = pd.read_csv(results_path)
    trial_results = pd.read_csv(trial_path) if trial_path.exists() else None

    return results_df, trial_results
