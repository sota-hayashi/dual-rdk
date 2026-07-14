"""OOZ状態依存学習率 Q学習モデル（MAP推定）

各試行のOOZラベル z_t ∈ {0, 1} に応じて学習率を切り替える:
    z_t = 0 (非OOZ): q[s,a] ← q[s,a] + alpha_0 × (r - q[s,a])
    z_t = 1 (OOZ):   q[s,a] ← q[s,a] + alpha_1 × (r - q[s,a])

OOZラベルは features.behavior.label_if_ooz で事前に付与しておく必要がある。

選択確率（MAP モデルと同一ロジック）:
    dq = beta × (Q[s,1] - Q[s,0]) + c_black
    state=0 (白=ターゲット): P(target) = sigmoid(-dq)
    state=1 (黒=ターゲット): P(target) = sigmoid(+dq)

パラメータ: alpha_0, alpha_1, beta, c_black（計4つ）

事前分布:
    alpha_0 ~ Beta(2, 2)
    alpha_1 ~ Beta(2, 2)
    beta    ~ Gamma(shape=2, scale=3)
    c_black ~ Normal(0, 2)
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist, gamma as gamma_dist, norm as norm_dist

from dualrdk.models.q_learning_map import _choice_determine


def _neg_log_posterior(
    params: np.ndarray,
    choices: np.ndarray,
    rewards: np.ndarray,
    states: np.ndarray,
    ooz_labels: np.ndarray,
) -> float:
    alpha_0, alpha_1, beta, c_black = params

    q = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    log_lik = 0.0

    for t in range(len(choices)):
        s = int(states[t])
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
        log_lik += np.log(p_chosen + 1e-300)

        alpha = alpha_0 if ooz_labels[t] == 0 else alpha_1
        q[s, a] += alpha * (rewards[t] - q[s, a])

    log_prior_a0   = beta_dist.logpdf(alpha_0, 2, 2)
    log_prior_a1   = beta_dist.logpdf(alpha_1, 2, 2)
    log_prior_beta = gamma_dist.logpdf(beta, a=2, scale=3)
    log_prior_c    = norm_dist.logpdf(c_black, 0, 2)

    log_posterior = log_lik + log_prior_a0 + log_prior_a1 + log_prior_beta + log_prior_c
    return -log_posterior


def _compute_log_likelihood(
    alpha_0: float,
    alpha_1: float,
    beta: float,
    c_black: float,
    choices: np.ndarray,
    rewards: np.ndarray,
    states: np.ndarray,
    ooz_labels: np.ndarray,
) -> float:
    q = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    log_lik = 0.0
    for t in range(len(choices)):
        s = int(states[t])
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
        log_lik += np.log(p_chosen + 1e-300)
        alpha = alpha_0 if ooz_labels[t] == 0 else alpha_1
        q[s, a] += alpha * (rewards[t] - q[s, a])
    return log_lik


def fit_q_learning_ooz_map(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_alpha0_grid: int = 5,
    n_alpha1_grid: int = 5,
    n_beta_grid: int = 5,
    n_c_grid: int = 5,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """各参加者の OOZ 状態依存 Q学習パラメータを MAP 推定する。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        df は rt, chosen_color, reward_points, target_item, ooz カラムを含む。
    n_alpha0_grid, n_alpha1_grid, n_beta_grid, n_c_grid : int
        グリッドサーチの分割数。合計 n_a0 × n_a1 × n_b × n_c 点。
    output_path : str or None
        指定した場合 CSV として保存する。

    Returns
    -------
    pd.DataFrame
        subject, alpha_0, alpha_1, beta, c_black, log_posterior,
        log_likelihood, n_trials, n_ooz, n_non_ooz カラムを含む。
    """
    records = []

    alpha0_inits = np.linspace(0.1, 0.9, n_alpha0_grid)
    alpha1_inits = np.linspace(0.1, 0.9, n_alpha1_grid)
    beta_inits   = np.linspace(0.5, 10.0, n_beta_grid)
    c_inits      = np.linspace(-2.0, 2.0, n_c_grid)

    bounds = [
        (1e-6, 1 - 1e-6),   # alpha_0
        (1e-6, 1 - 1e-6),   # alpha_1
        (1e-6, None),        # beta
        (None, None),        # c_black
    ]

    for subj_id, df in concat_list:
        if not {"rt", "chosen_color", "reward_points", "target_item", "ooz"}.issubset(df.columns):
            continue
        df_clean     = df.dropna(subset=["rt", "chosen_color", "ooz"]).copy()
        chosen_color = df_clean["chosen_color"]
        choices      = _choice_determine(chosen_color)
        rewards      = (df_clean["reward_points"].values / 10).astype(float)
        states       = df_clean["target_item"].values.astype(int)
        ooz_labels   = df_clean["ooz"].values.astype(int)
        n_trials     = len(choices)

        best_neg_lp = np.inf
        best_params = None

        for a0 in alpha0_inits:
            print(f"Subject {subj_id}: alpha_0={a0:.3f}")
            for a1 in alpha1_inits:
                for b0 in beta_inits:
                    for c0 in c_inits:
                        result = minimize(
                            _neg_log_posterior,
                            x0=[a0, a1, b0, c0],
                            args=(choices, rewards, states, ooz_labels),
                            method="L-BFGS-B",
                            bounds=bounds,
                        )
                        if result.fun < best_neg_lp:
                            best_neg_lp = result.fun
                            best_params = result.x

        a0_hat, a1_hat, beta_hat, c_hat = best_params
        log_posterior  = -best_neg_lp
        log_likelihood = _compute_log_likelihood(
            a0_hat, a1_hat, beta_hat, c_hat,
            choices, rewards, states, ooz_labels,
        )

        records.append({
            "subject":        subj_id,
            "alpha_0":        a0_hat,
            "alpha_1":        a1_hat,
            "beta":           beta_hat,
            "c_black":        c_hat,
            "log_posterior":  log_posterior,
            "log_likelihood": log_likelihood,
            "n_trials":       n_trials,
            "n_ooz":          int(ooz_labels.sum()),
            "n_non_ooz":      int((ooz_labels == 0).sum()),
        })

    results = pd.DataFrame(records)
    if output_path is not None:
        results.to_csv(output_path, index=False)
    return results


def _neg_log_posterior_group(
    params: np.ndarray,
    participants_data: list,
) -> float:
    """群全体で共有するパラメータに対する負の対数事後確率。

    各参加者ごとにQ値をリセットしつつ、対数尤度を合算する。

    Parameters
    ----------
    params : array of [alpha_0, alpha_1, beta, c_black]
    participants_data : list of (choices, rewards, states, ooz_labels) tuples
    """
    alpha_0, alpha_1, beta, c_black = params
    total_log_lik = 0.0

    for choices, rewards, states, ooz_labels in participants_data:
        q = np.array([[0.5, 0.5],
                      [0.5, 0.5]])
        for t in range(len(choices)):
            s = int(states[t])
            dq = beta * (q[s, 1] - q[s, 0]) + c_black
            dq = np.clip(dq, -500, 500)

            if s == 0:
                p_target = 1.0 / (1.0 + np.exp(dq))
                target_action = 0
            else:
                p_target = 1.0 / (1.0 + np.exp(-dq))
                target_action = 1

            a = int(choices[t])
            p_chosen = p_target if a == target_action else (1.0 - p_target)
            total_log_lik += np.log(p_chosen + 1e-300)

            alpha = alpha_0 if ooz_labels[t] == 0 else alpha_1
            q[s, a] += alpha * (rewards[t] - q[s, a])

    log_prior_a0   = beta_dist.logpdf(alpha_0, 2, 2)
    log_prior_a1   = beta_dist.logpdf(alpha_1, 2, 2)
    log_prior_beta = gamma_dist.logpdf(beta, a=2, scale=3)
    log_prior_c    = norm_dist.logpdf(c_black, 0, 2)

    log_posterior = total_log_lik + log_prior_a0 + log_prior_a1 + log_prior_beta + log_prior_c
    return -log_posterior


def fit_q_learning_ooz_map_group(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_alpha0_grid: int = 3,
    n_alpha1_grid: int = 3,
    n_beta_grid: int = 3,
    n_c_grid: int = 3,
) -> dict:
    """複数参加者のデータに対して共有パラメータをMAP推定する。

    各参加者ごとにQ値はリセットされるが、
    alpha_0, alpha_1, beta, c_black は全参加者で共通。

    Parameters
    ----------
    concat_list : list of (subject_id, DataFrame)
    n_alpha0_grid, n_alpha1_grid, n_beta_grid, n_c_grid : int
        グリッドサーチの分割数。

    Returns
    -------
    dict
        alpha_0, alpha_1, beta, c_black, alpha_diff,
        log_posterior, n_participants, n_total_trials
    """
    participants_data = []
    subject_ids = []
    for subj_id, df in concat_list:
        if not {"rt", "chosen_color", "reward_points", "target_item", "ooz"}.issubset(df.columns):
            continue
        df_clean = df.dropna(subset=["rt", "chosen_color", "ooz"]).copy()
        if df_clean.empty:
            continue
        choices = _choice_determine(df_clean["chosen_color"])
        rewards = (df_clean["reward_points"].values / 10).astype(float)
        states = df_clean["target_item"].values.astype(int)
        ooz_labels = df_clean["ooz"].values.astype(int)
        participants_data.append((choices, rewards, states, ooz_labels))
        subject_ids.append(subj_id)

    if not participants_data:
        raise ValueError("No valid participant data found.")

    alpha0_inits = np.linspace(0.1, 0.9, n_alpha0_grid)
    alpha1_inits = np.linspace(0.1, 0.9, n_alpha1_grid)
    beta_inits = np.linspace(0.5, 10.0, n_beta_grid)
    c_inits = np.linspace(-2.0, 2.0, n_c_grid)

    bounds = [
        (1e-6, 1 - 1e-6),   # alpha_0
        (1e-6, 1 - 1e-6),   # alpha_1
        (1e-6, None),        # beta
        (None, None),        # c_black
    ]

    best_neg_lp = np.inf
    best_params = None

    for a0 in alpha0_inits:
        for a1 in alpha1_inits:
            for b0 in beta_inits:
                for c0 in c_inits:
                    result = minimize(
                        _neg_log_posterior_group,
                        x0=[a0, a1, b0, c0],
                        args=(participants_data,),
                        method="L-BFGS-B",
                        bounds=bounds,
                    )
                    if result.fun < best_neg_lp:
                        best_neg_lp = result.fun
                        best_params = result.x

    a0_hat, a1_hat, beta_hat, c_hat = best_params

    # 参加者ごとの対数尤度
    per_participant = []
    for i, (choices, rewards, states, ooz_labels) in enumerate(participants_data):
        ll = _compute_log_likelihood(
            a0_hat, a1_hat, beta_hat, c_hat,
            choices, rewards, states, ooz_labels,
        )
        per_participant.append({
            "subject": subject_ids[i],
            "log_likelihood": ll,
            "n_trials": len(choices),
        })

    ll_arr = np.array([p["log_likelihood"] for p in per_participant])
    n_total_trials = sum(len(c) for c, _, _, _ in participants_data)

    return {
        "alpha_0": a0_hat,
        "alpha_1": a1_hat,
        "beta": beta_hat,
        "c_black": c_hat,
        "alpha_diff": float(np.log(a1_hat / a0_hat)),
        "log_posterior": -best_neg_lp,
        "n_participants": len(participants_data),
        "n_total_trials": n_total_trials,
        "per_participant": per_participant,
        "fit_summary": {
            "mean_ll": float(np.mean(ll_arr)),
            "std_ll": float(np.std(ll_arr, ddof=1)),
            "min_ll": float(np.min(ll_arr)),
            "max_ll": float(np.max(ll_arr)),
        },
    }


def predict_target_choice_probs(
    concat_list: List[Tuple[str, pd.DataFrame]],
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """推定パラメータで各試行のターゲット選択確率を予測する（OOZ版）

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        df は rt, chosen_color, reward_points, target_item, ooz カラムを含む。
    results_df : pd.DataFrame
        fit_q_learning_ooz_map の出力（alpha_0, alpha_1, beta, c_black カラムが必要）。
    output_path : str or None

    Returns
    -------
    pd.DataFrame
        trial-level の予測確率。ooz カラムを追加で含む。
    """
    param_map = results_df.set_index("subject")[
        ["alpha_0", "alpha_1", "beta", "c_black"]
    ].to_dict("index")
    all_rows = []

    for subject, df in concat_list:
        if subject not in param_map:
            continue
        alpha_0 = param_map[subject]["alpha_0"]
        alpha_1 = param_map[subject]["alpha_1"]
        beta    = param_map[subject]["beta"]
        c_black = param_map[subject]["c_black"]

        df_clean = df.dropna(subset=["rt", "chosen_color", "ooz"]).copy()
        required = {"rt", "chosen_color", "target_item", "reward_points", "ooz"}
        if not required.issubset(df_clean.columns):
            continue

        chosen_color = df_clean["chosen_color"]
        choices      = _choice_determine(chosen_color)
        states       = df_clean["target_item"].values.astype(int)
        rewards      = (df_clean["reward_points"].values / 10).astype(float)
        ooz_labels   = df_clean["ooz"].values.astype(int)

        q = np.array([[0.5, 0.5], [0.5, 0.5]])
        n = len(choices)
        for t in range(n):
            s     = int(states[t])
            color = chosen_color.iloc[t]
            state = "white_high" if s == 0 else "black_high"

            dq_a = beta * (q[1, 1] - q[1, 0]) + c_black
            dq_a = np.clip(dq_a, -500, 500)
            p_patternA = 1.0 / (1.0 + np.exp(-dq_a))   # P(black)

            dq_b = beta * (q[0, 1] - q[0, 0]) + c_black
            dq_b = np.clip(dq_b, -500, 500)
            p_patternB = 1.0 / (1.0 + np.exp(dq_b))    # P(white)

            all_rows.append({
                "subject":           subject,
                "trial":             t,
                "state":             state,
                "chosen_color":      color,
                "ooz":               int(ooz_labels[t]),
                "Q_patternA_white":  q[1, 0],
                "Q_patternA_black":  q[1, 1],
                "Q_patternB_white":  q[0, 0],
                "Q_patternB_black":  q[0, 1],
                "p_patternA":        p_patternA,
                "p_patternB":        p_patternB,
            })

            alpha = alpha_0 if ooz_labels[t] == 0 else alpha_1
            a = int(choices[t])
            q[s, a] += alpha * (rewards[t] - q[s, a])

    trial_results = pd.DataFrame(all_rows)
    if output_path is not None:
        trial_results.to_csv(output_path, index=False)
    return trial_results
