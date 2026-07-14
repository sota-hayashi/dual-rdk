from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist, gamma as gamma_dist, norm as norm_dist

def _choice_determine(chosen_color: np.ndarray):
    """chosen_color カラムから choices を決定する。"""
    # choices: 1=black, 0=white
    chosen_color = chosen_color.values.astype(str)
    choices = np.zeros(len(chosen_color), dtype=int)
    for n in range(len(chosen_color)):
        if chosen_color[n] == "black":
            choices[n] = 1
        elif chosen_color[n] == "white":
            choices[n] = 0
        else:
            raise ValueError(f"Invalid chosen_color: {chosen_color[n]}")
    return choices


def _neg_log_posterior(params: np.ndarray, choices: np.ndarray, rewards: np.ndarray, states: np.ndarray) -> float:
    alpha, beta, c_black = params

    # Q値の初期化：2状態 × 2行動
    q = np.array([[0.5, 0.5],   # state=0(white=target:Pattern B): [white, black]
                  [0.5, 0.5]])  # state=1(black=target:Pattern A): [white, black]

    log_lik = 0.0
    for t in range(len(choices)):
        s = int(states[t])

        # 常に「黒 vs 白」のlog-oddsで定義し，色バイアスを加算
        dq = beta * (q[s, 1] - q[s, 0]) + c_black  # c_black>0 → 黒選好
        dq = np.clip(dq, -500, 500)

        if s == 0:  # white=target
            p_target = 1.0 / (1.0 + np.exp(dq))   # P(white) = 1 - sigmoid(dq)
            target_action = 0
        else:       # black=target
            p_target = 1.0 / (1.0 + np.exp(-dq))  # P(black) = sigmoid(dq)
            target_action = 1

        a = int(choices[t])  # 1=black, 0=white
        p_chosen = p_target if a == target_action else (1.0 - p_target)
        log_lik += np.log(p_chosen + 1e-300)

        # Q値の更新（選択した行動のみ）
        r = rewards[t]
        q[s, a] += alpha * (r - q[s, a])

    # 対数事前確率: alpha ~ Beta(2,2), beta ~ Gamma(2,3), c_black ~ Normal(0,2)
    log_prior_alpha = beta_dist.logpdf(alpha, 2, 2)
    log_prior_beta = gamma_dist.logpdf(beta, a=2, scale=3)
    log_prior_c = norm_dist.logpdf(c_black, 0, 2)

    log_posterior = log_lik + log_prior_alpha + log_prior_beta + log_prior_c
    return -log_posterior


def _compute_log_likelihood(alpha: float, beta: float, c_black: float, choices: np.ndarray, rewards: np.ndarray, states: np.ndarray) -> float:
    q = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    log_lik = 0.0
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
        log_lik += np.log(p_chosen + 1e-300)
        r = rewards[t]
        q[s, a] += alpha * (r - q[s, a])
    return log_lik


def fit_q_learning_map(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_alpha_grid: int = 5,
    n_beta_grid: int = 5,
    n_c_grid: int = 5,
    output_path: str = None,
) -> pd.DataFrame:
    """各参加者のQ学習モデルパラメータをMAP推定で推定する。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        各タプルは (subject, df)。df は rt, chosen_color, reward_points カラムを含む。
    n_alpha_grid : int
        初期値グリッドのalpha方向の点数。
    n_beta_grid : int
        初期値グリッドのbeta方向の点数。
    n_c_grid : int
        初期値グリッドのc_black方向の点数。

    Returns
    -------
    pd.DataFrame
        subject, alpha, beta, c_black, log_posterior, log_likelihood, n_trials カラムを含む。
    """
    records = []

    alpha_inits = np.linspace(0.1, 0.9, n_alpha_grid)
    beta_inits = np.linspace(0.5, 10.0, n_beta_grid)
    c_inits = np.linspace(-2.0, 2.0, n_c_grid)

    bounds = [(1e-6, 1 - 1e-6), (1e-6, None), (None, None)]

    for subj_id, df in concat_list:
        # 前処理: rt と chosen_color が両方有効な行のみ使用
        df_clean = df.dropna(subset=["rt", "chosen_color"]).copy()
        chosen_color = df_clean["chosen_color"]
        choices = _choice_determine(chosen_color)
        rewards = (df_clean["reward_points"].values / 10).astype(float)
        states = df_clean["target_item"].values.astype(int)
        n_trials = len(choices)

        best_neg_lp = np.inf
        best_params = None

        for a0 in alpha_inits:
            for b0 in beta_inits:
                for c0 in c_inits:
                    result = minimize(
                        _neg_log_posterior,
                        x0=[a0, b0, c0],
                        args=(choices, rewards, states),
                        method="L-BFGS-B",
                        bounds=bounds,
                    )
                    if result.fun < best_neg_lp:
                        best_neg_lp = result.fun
                        best_params = result.x

        alpha_hat, beta_hat, c_hat = best_params
        log_posterior = -best_neg_lp
        log_likelihood = _compute_log_likelihood(alpha_hat, beta_hat, c_hat, choices, rewards, states)

        records.append({
            "subject": subj_id,
            "alpha": alpha_hat,
            "beta": beta_hat,
            "c_black": c_hat,
            "log_posterior": log_posterior,
            "log_likelihood": log_likelihood,
            "n_trials": n_trials,
        })

    results = pd.DataFrame(records)
    if output_path is not None:
        results.to_csv(output_path, index=False)
    return results

def predict_target_choice_probs(
    concat_list: List[Tuple[str, pd.DataFrame]],
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    推定パラメータで各試行のターゲット選択確率を予測する
    
    Returns
    -------
    concat_list : list of (subject, DataFrame)
        df は rt, chosen_color, reward_points, target_item カラムを含む。
    results_df : pd.DataFrame
        q_learning_map の出力（subject, alpha, beta カラムが必要）。
    output_path : str or None
    """

    param_map = results_df.set_index("subject")[["alpha", "beta", "c_black"]].to_dict("index")
    all_rows = []

    for subject, df in concat_list:
        q = np.array([[0.5, 0.5],
                      [0.5, 0.5]])
        if subject not in param_map:
            continue
        alpha   = param_map[subject]["alpha"]
        beta    = param_map[subject]["beta"]
        c_black = param_map[subject]["c_black"]

        df_clean = df.dropna(subset=["rt", "chosen_color"]).copy()
        required = {"rt", "chosen_color", "target_item", "reward_points"}
        if not required.issubset(df_clean.columns):
            continue

        chosen_color = df_clean["chosen_color"]
        choices      = _choice_determine(chosen_color)
        states       = df_clean["target_item"].values.astype(int)
        rewards      = (df_clean["reward_points"].values / 10).astype(float)

        n = len(choices)
        probs_patternA = np.zeros(n)
        probs_patternB = np.zeros(n)
        q_history = np.zeros((n, 2, 2))
        for t in range(n):
            s = int(states[t])
            color = chosen_color.iloc[t]
            state = ""
            if s == 0:
                state = "white_high"
            elif s == 1:
                state = "black_high"
            
            # 選択前のQ値を記録
            q_history[t] = q.copy()
            
            # 状態sにおけるターゲット選択確率（softmax + 色バイアス）
            dq_a = beta * (q[1, 1] - q[1, 0]) + c_black  # PatternA: black=target
            dq_a = np.clip(dq_a, -500, 500)
            probs_patternA[t] = 1.0 / (1.0 + np.exp(-dq_a))  # P(black)

            dq_b = beta * (q[0, 1] - q[0, 0]) + c_black  # PatternB: white=target
            dq_b = np.clip(dq_b, -500, 500)
            probs_patternB[t] = 1.0 / (1.0 + np.exp(dq_b))   # P(white) = 1-sigmoid(dq_b)

            q_patternA_black, q_patternA_white = q[1, 1], q[1, 0]
            q_patternB_black, q_patternB_white = q[0, 1], q[0, 0]
            p_patternA = probs_patternA[t]
            p_patternB = probs_patternB[t]

            all_rows.append({
                "subject":      subject,
                "trial":        t,
                "state":        state,
                "chosen_color": color,
                "Q_patternA_white":      q_patternA_white,
                "Q_patternA_black":      q_patternA_black,
                "Q_patternB_white":      q_patternB_white,
                "Q_patternB_black":      q_patternB_black,
                "p_patternA":   p_patternA,
                "p_patternB":   p_patternB,
            })

            # Q値更新（実際の行動で）
            a = int(choices[t])
            r = rewards[t]
            q[s, a] += alpha * (r - q[s, a])

    trial_results = pd.DataFrame(all_rows)
    if output_path is not None:
        trial_results.to_csv(output_path, index=False)
    return trial_results