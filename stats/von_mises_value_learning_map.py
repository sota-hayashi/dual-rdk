"""von Mises 基底関数による方向空間価値学習モデル（MAP推定）

連続方向空間上に等間隔に配置した N 個の von Mises 基底関数で価値関数 V(θ) を表現し，
報酬予測誤差によって基底の重みを更新する。

モデルの要点:
    V(θ) = Σ_j W_j φ_j(θ),   φ_j(θ) = exp[κ cos(θ - μ_j)]
    更新:  W_j ← W_j + α δ_t φ_j(θ_reported),   δ_t = r_t - V(θ_reported)
    選択確率（softmax）: P(θ_A) = σ(β(V(θ_A) - V(θ_B)))
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist, gamma as gamma_dist


# ---------------------------------------------------------------------------
# 基底関数ユーティリティ
# ---------------------------------------------------------------------------

def _make_mus(N: int) -> np.ndarray:
    return np.array([2 * np.pi * j / N for j in range(N)])


def _basis_vec(theta: float, mus: np.ndarray, kappa: float) -> np.ndarray:
    """shape (N,) の基底ベクトルを返す。"""
    return np.exp(kappa * np.cos(theta - mus))


def _value(theta: float, W: np.ndarray, mus: np.ndarray, kappa: float) -> float:
    return float(np.dot(W, _basis_vec(theta, mus, kappa)))


def _angular_distance(a: float, b: float) -> float:
    """[0, π] に収まる円周上の距離。"""
    diff = abs(a - b) % (2 * np.pi)
    return float(min(diff, 2 * np.pi - diff))


# ---------------------------------------------------------------------------
# 対数事後確率（最適化用）
# ---------------------------------------------------------------------------

def _run_trial_loop(
    alpha: float,
    beta_inv: float,
    response_angles: np.ndarray,
    stim_angles_1: np.ndarray,
    stim_angles_2: np.ndarray,
    rewards: np.ndarray,
    mus: np.ndarray,
    kappa: float,
) -> Tuple[float, np.ndarray]:
    """試行ループを実行し（対数尤度, 最終重みベクトル）を返す。"""
    N = len(mus)
    W = np.zeros(N)
    log_lik = 0.0

    for t in range(len(response_angles)):
        theta_resp = response_angles[t]
        theta_1    = stim_angles_1[t]
        theta_2    = stim_angles_2[t]

        v1 = _value(theta_1, W, mus, kappa)
        v2 = _value(theta_2, W, mus, kappa)

        # softmax 選択確率
        dv = np.clip(beta_inv * (v1 - v2), -500, 500)
        p1 = 1.0 / (1.0 + np.exp(-dv))

        # 反応角度に近い刺激の選択確率を対数尤度に加算
        d1 = _angular_distance(theta_resp, theta_1)
        d2 = _angular_distance(theta_resp, theta_2)
        p_chosen = p1 if d1 <= d2 else (1.0 - p1)
        log_lik += np.log(p_chosen + 1e-300)

        # 重み更新（報告角度に基づく）
        v_resp = _value(theta_resp, W, mus, kappa)
        delta  = rewards[t] - v_resp
        W     += alpha * delta * _basis_vec(theta_resp, mus, kappa)

    return log_lik, W


def _neg_log_posterior(
    params: np.ndarray,
    response_angles: np.ndarray,
    stim_angles_1: np.ndarray,
    stim_angles_2: np.ndarray,
    rewards: np.ndarray,
    mus: np.ndarray,
    kappa: float,
) -> float:
    alpha, beta_inv = params
    log_lik, _ = _run_trial_loop(
        alpha, beta_inv, response_angles, stim_angles_1, stim_angles_2, rewards, mus, kappa
    )
    log_prior = beta_dist.logpdf(alpha, 2, 2) + gamma_dist.logpdf(beta_inv, a=2, scale=3)
    return -(log_lik + log_prior)


# ---------------------------------------------------------------------------
# 公開インタフェース
# ---------------------------------------------------------------------------

def fit_von_mises_map(
    concat_list: List[Tuple[str, pd.DataFrame]],
    N: int = 4,
    kappa: float = 2.0,
    n_alpha_grid: int = 5,
    n_beta_grid: int = 5,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """von Mises 基底関数モデルのパラメータを MAP 推定する。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        各タプルは (subject, df)。
        df は rt, response_angle_css, target_direction, distractor_direction,
        reward_points カラムを含む。
    N : int
        基底関数の数（デフォルト 4）。
    kappa : float
        von Mises 基底の集中度パラメータ（デフォルト 2.0）。
    n_alpha_grid, n_beta_grid : int
        グリッドサーチの初期値点数。
    output_path : str or None
        指定した場合、results を CSV として保存する。

    Returns
    -------
    pd.DataFrame
        subject, alpha, beta, log_posterior, log_likelihood, n_trials, N, kappa カラム。
    """
    mus    = _make_mus(N)
    bounds = [(1e-6, 1 - 1e-6), (1e-6, None)]

    alpha_inits = np.linspace(0.1, 0.9, n_alpha_grid)
    beta_inits  = np.linspace(0.5, 10.0, n_beta_grid)

    records = []

    for subject, df in concat_list:
        df_clean = df.dropna(subset=["rt"]).copy()
        response_angles = df_clean["response_angle_css"].values.astype(float)
        stim_angles_1   = df_clean["target_direction"].values.astype(float)
        stim_angles_2   = df_clean["distractor_direction"].values.astype(float)
        rewards         = (df_clean["reward_points"].values / 10.0).astype(float)
        n_trials        = len(response_angles)

        best_neg_lp  = np.inf
        best_params  = None

        for a0 in alpha_inits:
            for b0 in beta_inits:
                result = minimize(
                    _neg_log_posterior,
                    x0=[a0, b0],
                    args=(response_angles, stim_angles_1, stim_angles_2, rewards, mus, kappa),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if result.fun < best_neg_lp:
                    best_neg_lp = result.fun
                    best_params = result.x

        alpha_hat, beta_hat = best_params
        log_posterior = -best_neg_lp
        log_likelihood, _ = _run_trial_loop(
            alpha_hat, beta_hat,
            response_angles, stim_angles_1, stim_angles_2, rewards, mus, kappa
        )

        records.append({
            "subject":       subject,
            "alpha":         alpha_hat,
            "beta":          beta_hat,
            "log_posterior": log_posterior,
            "log_likelihood": log_likelihood,
            "n_trials":      n_trials,
            "N":             N,
            "kappa":         kappa,
        })

    results = pd.DataFrame(records)
    if output_path is not None:
        results.to_csv(output_path, index=False)
    return results


def predict_target_choice_von_mises(
    concat_list: List[Tuple[str, pd.DataFrame]],
    results_df: pd.DataFrame,
    N: int = 4,
    kappa: float = 2.0,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """推定済みパラメータで試行ごとの target 選択確率を事後計算する。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        df は rt, response_angle_css, target_direction, distractor_direction,
        reward_points カラムを含む。
    results_df : pd.DataFrame
        fit_von_mises_map の出力（subject, alpha, beta カラムが必要）。
    N, kappa : int, float
        フィット時と同じパラメータ。
    output_path : str or None

    Returns
    -------
    pd.DataFrame
        subject, trial, V_target, V_distractor, p_target カラム。
    """
    mus = _make_mus(N)
    param_map = results_df.set_index("subject")[["alpha", "beta"]].to_dict("index")
    all_rows = []

    for subject, df in concat_list:
        if subject not in param_map:
            continue
        alpha   = param_map[subject]["alpha"]
        beta_inv = param_map[subject]["beta"]

        df_clean = df.dropna(subset=["rt"]).copy()
        required = {"response_angle_css", "target_direction", "distractor_direction",
                    "reward_points"}
        if not required.issubset(df_clean.columns):
            continue

        response_angles    = df_clean["response_angle_css"].values.astype(float)
        stim_angles_1      = df_clean["target_direction"].values.astype(float)
        stim_angles_2      = df_clean["distractor_direction"].values.astype(float)
        rewards            = (df_clean["reward_points"].values / 10.0).astype(float)

        W = np.zeros(N)
        for t in range(len(response_angles)):
            theta_resp = response_angles[t]
            theta_1    = stim_angles_1[t]
            theta_2    = stim_angles_2[t]

            v_target     = _value(theta_1, W, mus, kappa)
            v_distractor = _value(theta_2, W, mus, kappa)

            dv = np.clip(beta_inv * (v_target - v_distractor), -500, 500)
            p_target = 1.0 / (1.0 + np.exp(-dv))

            all_rows.append({
                "subject":      subject,
                "trial":        t,
                "V_target":     v_target,
                "V_distractor": v_distractor,
                "p_target":     p_target,
            })

            # 重み更新（学習フェーズと同一ロジック）
            v_resp = _value(theta_resp, W, mus, kappa)
            delta  = rewards[t] - v_resp
            W     += alpha * delta * _basis_vec(theta_resp, mus, kappa)

    trial_results = pd.DataFrame(all_rows)
    if output_path is not None:
        trial_results.to_csv(output_path, index=False)
    return trial_results
