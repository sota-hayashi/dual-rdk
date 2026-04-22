from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist, gamma as gamma_dist


def _neg_log_posterior(params: np.ndarray, choices: np.ndarray, rewards: np.ndarray, states: np.ndarray) -> float:
    alpha, beta = params

    # Q値の初期化：2状態 × 2行動
    q = np.array([[0.5, 0.5],   # state=0(white): [target, distractor]
                  [0.5, 0.5]])  # state=1(black): [target, distractor]

    log_lik = 0.0
    for t in range(len(choices)):
        s = int(states[t])
        # softmax選択確率
        dq = beta * (q[s, 0] - q[s, 1])
        # 数値安定性のためクリップ
        dq = np.clip(dq, -500, 500)
        p_target = 1.0 / (1.0 + np.exp(-dq))

        a = int(choices[t])  # 1=target, 0=distractor
        p_chosen = p_target if a == 1 else (1.0 - p_target)
        log_lik += np.log(p_chosen + 1e-300)

        # Q値の更新（選択した行動のみ）
        r = rewards[t]
        q[s, a] += alpha * (r - q[s, a])

    # 対数事前確率: alpha ~ Beta(2,2), beta ~ Gamma(2,3) (shape=2, scale=3)
    log_prior_alpha = beta_dist.logpdf(alpha, 2, 2)
    log_prior_beta = gamma_dist.logpdf(beta, a=2, scale=3)

    log_posterior = log_lik + log_prior_alpha + log_prior_beta
    return -log_posterior


def _compute_log_likelihood(alpha: float, beta: float, choices: np.ndarray, rewards: np.ndarray, states: np.ndarray) -> float:
    q = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    log_lik = 0.0
    for t in range(len(choices)):
        s = int(states[t])  # 0 or 1
        dq = beta * (q[s, 0] - q[s, 1])
        dq = np.clip(dq, -500, 500)
        p_target = 1.0 / (1.0 + np.exp(-dq))
        a = int(choices[t])
        p_chosen = p_target if a == 1 else (1.0 - p_target)
        log_lik += np.log(p_chosen + 1e-300)
        r = rewards[t]
        q[s, a] += alpha * (r - q[s, a])
    return log_lik


def fit_q_learning_map(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_alpha_grid: int = 5,
    n_beta_grid: int = 5,
    output_path: str = None,
) -> pd.DataFrame:
    """各参加者のQ学習モデルパラメータをMAP推定で推定する。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        各タプルは (subject, df)。df は rt, chosen_item, reward_points カラムを含む。
    n_alpha_grid : int
        初期値グリッドのalpha方向の点数。
    n_beta_grid : int
        初期値グリッドのbeta方向の点数。

    Returns
    -------
    pd.DataFrame
        subject, alpha, beta, log_posterior, log_likelihood, n_trials カラムを含む。
    """
    records = []

    alpha_inits = np.linspace(0.1, 0.9, n_alpha_grid)
    beta_inits = np.linspace(0.5, 10.0, n_beta_grid)

    bounds = [(1e-6, 1 - 1e-6), (1e-6, None)]

    for subj_id, df in concat_list:
        # 前処理
        df_clean = df.dropna(subset=["rt"]).copy()
        choices = df_clean["chosen_item"].values.astype(int)
        rewards = (df_clean["reward_points"].values >= 1).astype(float)
        states = df_clean["target_item"].values.astype(int)
        n_trials = len(choices)

        best_neg_lp = np.inf
        best_params = None

        for a0 in alpha_inits:
            for b0 in beta_inits:
                result = minimize(
                    _neg_log_posterior,
                    x0=[a0, b0],
                    args=(choices, rewards, states),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if result.fun < best_neg_lp:
                    best_neg_lp = result.fun
                    best_params = result.x

        alpha_hat, beta_hat = best_params
        log_posterior = -best_neg_lp
        log_likelihood = _compute_log_likelihood(alpha_hat, beta_hat, choices, rewards, states)

        records.append({
            "subject": subj_id,
            "alpha": alpha_hat,
            "beta": beta_hat,
            "log_posterior": log_posterior,
            "log_likelihood": log_likelihood,
            "n_trials": n_trials,
        })

    results = pd.DataFrame(records)
    if output_path is not None:
        results.to_csv(output_path, index=False)
    return results
