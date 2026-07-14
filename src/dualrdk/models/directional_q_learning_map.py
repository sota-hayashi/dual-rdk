"""色別方向価値Q学習モデル（OOZ状態依存学習率, MAP推定）

各色 c ∈ {white, black} ごとに，連続円環方向空間上の価値関数を
von Mises 基底の重み W_c で表現する:

    V_c(θ) = W_c · φ(θ),   φ_j(θ) = exp[κ cos(θ - μ_j)]

各試行で白・黒のドット群がそれぞれ方向 θ_white, θ_black に運動しており，
それらの現在価値を比較して色を選択する（softmax + 色バイアス c_black）:

    dq = β (V_black(θ_black) - V_white(θ_white)) + c_black
    P(black) = sigmoid(dq)

報酬を得たあと，選んだ色の価値関数のみを報告方向 θ_reported で更新する。
学習率は試行の OOZ ラベル z_t で切り替える（q_learning_ooz_map と同じ構成）:

    α_t = α_0 (z_t=0, 非OOZ) / α_1 (z_t=1, OOZ)
    δ_t = r_t - V_{c*}(θ_reported)
    W_{c*} ← W_{c*} + α_t δ_t φ(θ_reported)

パラメータ: alpha_0, alpha_1, beta, c_black（計4つ）。
基底パラメータ N, κ は固定。

事前分布:
    alpha_0 ~ Beta(2, 2)
    alpha_1 ~ Beta(2, 2)
    beta    ~ Gamma(shape=2, scale=3)
    c_black ~ Normal(0, 2)

注意: データの角度カラム（target_direction 等）は度数法 [0,360) で格納されている
ため，モデル内部ではラジアンに変換して使用する。
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist, gamma as gamma_dist, norm as norm_dist

from dualrdk.models.q_learning_map import _choice_determine
from dualrdk.models.von_mises_value_learning_map import _make_mus, _basis_vec, _value


# ---------------------------------------------------------------------------
# 前処理: 1参加者分の配列を組み立てる
# ---------------------------------------------------------------------------

_REQUIRED_COLS = {
    "rt", "chosen_color", "response_angle_css",
    "target_direction", "distractor_direction", "target_group",
    "reward_points", "ooz",
}


def _prepare_subject(df: pd.DataFrame) -> Optional[dict]:
    """1参加者の DataFrame から学習に必要な配列を取り出す。

    角度は度数法 → ラジアンに変換する。target_group を使って
    白・黒それぞれの運動方向 θ_white, θ_black を復元する。
    必要カラムが欠ける場合は None を返す。
    """
    if not _REQUIRED_COLS.issubset(df.columns):
        return None

    df_clean = df.dropna(
        subset=["rt", "chosen_color", "response_angle_css",
                "target_direction", "distractor_direction", "target_group", "ooz"]
    ).copy()
    if df_clean.empty:
        return None

    choices  = _choice_determine(df_clean["chosen_color"])          # 1=black, 0=white
    rewards  = (df_clean["reward_points"].values / 10).astype(float)
    ooz      = df_clean["ooz"].values.astype(int)

    target_dir     = np.deg2rad(df_clean["target_direction"].values.astype(float))
    distractor_dir = np.deg2rad(df_clean["distractor_direction"].values.astype(float))
    reported       = np.deg2rad(df_clean["response_angle_css"].values.astype(float))
    target_is_white = (df_clean["target_group"].values.astype(str) == "white")

    # 白がターゲットなら θ_white=target_dir, 黒がターゲットなら θ_white=distractor_dir
    theta_white = np.where(target_is_white, target_dir, distractor_dir)
    theta_black = np.where(target_is_white, distractor_dir, target_dir)

    return {
        "choices":     choices,
        "rewards":     rewards,
        "ooz":         ooz,
        "theta_white": theta_white,
        "theta_black": theta_black,
        "reported":    reported,
        "target_dir":  target_dir,
        "distractor_dir": distractor_dir,
        "target_is_white": target_is_white,
        "n_trials":    len(choices),
    }


# ---------------------------------------------------------------------------
# 試行ループ（対数尤度）
# ---------------------------------------------------------------------------

def _run_trial_loop(
    alpha_0: float,
    alpha_1: float,
    beta: float,
    c_black: float,
    data: dict,
    mus: np.ndarray,
    kappa: float,
) -> float:
    """試行ループを実行して対数尤度を返す。"""
    N = len(mus)
    W_white = np.zeros(N)
    W_black = np.zeros(N)

    choices     = data["choices"]
    rewards     = data["rewards"]
    ooz         = data["ooz"]
    theta_white = data["theta_white"]
    theta_black = data["theta_black"]
    reported    = data["reported"]

    log_lik = 0.0
    for t in range(len(choices)):
        v_white = _value(theta_white[t], W_white, mus, kappa)
        v_black = _value(theta_black[t], W_black, mus, kappa)

        dq = beta * (v_black - v_white) + c_black
        dq = np.clip(dq, -500, 500)
        p_black = 1.0 / (1.0 + np.exp(-dq))

        a = int(choices[t])  # 1=black, 0=white
        p_chosen = p_black if a == 1 else (1.0 - p_black)
        log_lik += np.log(p_chosen + 1e-300)

        # 選んだ色の価値関数のみ報告方向で更新
        alpha = alpha_0 if ooz[t] == 0 else alpha_1
        phi_resp = _basis_vec(reported[t], mus, kappa)
        if a == 1:
            v_resp = float(np.dot(W_black, phi_resp))
            W_black += alpha * (rewards[t] - v_resp) * phi_resp
        else:
            v_resp = float(np.dot(W_white, phi_resp))
            W_white += alpha * (rewards[t] - v_resp) * phi_resp

    return log_lik


def _neg_log_posterior(
    params: np.ndarray,
    data: dict,
    mus: np.ndarray,
    kappa: float,
) -> float:
    alpha_0, alpha_1, beta, c_black = params
    log_lik = _run_trial_loop(alpha_0, alpha_1, beta, c_black, data, mus, kappa)

    log_prior = (
        beta_dist.logpdf(alpha_0, 2, 2)
        + beta_dist.logpdf(alpha_1, 2, 2)
        + gamma_dist.logpdf(beta, a=2, scale=3)
        + norm_dist.logpdf(c_black, 0, 2)
    )
    return -(log_lik + log_prior)


# ---------------------------------------------------------------------------
# 公開インタフェース
# ---------------------------------------------------------------------------

def fit_directional_q_learning_map(
    concat_list: List[Tuple[str, pd.DataFrame]],
    N: int = 4,
    kappa: float = 2.0,
    n_alpha0_grid: int = 4,
    n_alpha1_grid: int = 4,
    n_beta_grid: int = 4,
    n_c_grid: int = 4,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """各参加者の色別方向価値Q学習（OOZ依存学習率）を MAP 推定する。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        df は rt, chosen_color, response_angle_css, target_direction,
        distractor_direction, target_group, reward_points, ooz を含む。
    N : int
        von Mises 基底の数（デフォルト 4）。
    kappa : float
        基底の集中度（デフォルト 2.0）。
    n_alpha0_grid, n_alpha1_grid, n_beta_grid, n_c_grid : int
        グリッドサーチの分割数。合計 n_a0 × n_a1 × n_b × n_c 点。
    output_path : str or None
        指定した場合 CSV として保存する。

    Returns
    -------
    pd.DataFrame
        subject, alpha_0, alpha_1, beta, c_black, log_posterior,
        log_likelihood, n_trials, n_ooz, n_non_ooz, N, kappa カラム。
    """
    mus = _make_mus(N)
    bounds = [
        (1e-6, 1 - 1e-6),   # alpha_0
        (1e-6, 1 - 1e-6),   # alpha_1
        (1e-6, None),        # beta
        (None, None),        # c_black
    ]

    alpha0_inits = np.linspace(0.1, 0.9, n_alpha0_grid)
    alpha1_inits = np.linspace(0.1, 0.9, n_alpha1_grid)
    beta_inits   = np.linspace(0.5, 10.0, n_beta_grid)
    c_inits      = np.linspace(-2.0, 2.0, n_c_grid)

    records = []
    for subj_id, df in concat_list:
        data = _prepare_subject(df)
        if data is None:
            continue

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
                            args=(data, mus, kappa),
                            method="L-BFGS-B",
                            bounds=bounds,
                        )
                        if result.fun < best_neg_lp:
                            best_neg_lp = result.fun
                            best_params = result.x

        a0_hat, a1_hat, beta_hat, c_hat = best_params
        log_likelihood = _run_trial_loop(a0_hat, a1_hat, beta_hat, c_hat, data, mus, kappa)
        ooz = data["ooz"]

        records.append({
            "subject":        subj_id,
            "alpha_0":        a0_hat,
            "alpha_1":        a1_hat,
            "beta":           beta_hat,
            "c_black":        c_hat,
            "log_posterior":  -best_neg_lp,
            "log_likelihood": log_likelihood,
            "n_trials":       data["n_trials"],
            "n_ooz":          int(ooz.sum()),
            "n_non_ooz":      int((ooz == 0).sum()),
            "N":              N,
            "kappa":          kappa,
        })

    results = pd.DataFrame(records)
    if output_path is not None:
        results.to_csv(output_path, index=False)
    return results


def predict_target_choice_probs(
    concat_list: List[Tuple[str, pd.DataFrame]],
    results_df: pd.DataFrame,
    N: int = 4,
    kappa: float = 2.0,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """推定パラメータで各試行のターゲット選択確率を事後復元する。

    モデルの学習にはターゲット/報酬パターンのラベルを使わないため，
    target_group はここで p_target を組み立てる目的にのみ使用する。

    Returns
    -------
    pd.DataFrame
        subject, trial, state, chosen_color, ooz, V_white, V_black,
        p_black, p_target カラム。
    """
    mus = _make_mus(N)
    param_map = results_df.set_index("subject")[
        ["alpha_0", "alpha_1", "beta", "c_black"]
    ].to_dict("index")
    all_rows = []

    for subject, df in concat_list:
        if subject not in param_map:
            continue
        data = _prepare_subject(df)
        if data is None:
            continue

        alpha_0 = param_map[subject]["alpha_0"]
        alpha_1 = param_map[subject]["alpha_1"]
        beta    = param_map[subject]["beta"]
        c_black = param_map[subject]["c_black"]

        W_white = np.zeros(N)
        W_black = np.zeros(N)
        choices         = data["choices"]
        rewards         = data["rewards"]
        ooz             = data["ooz"]
        theta_white     = data["theta_white"]
        theta_black     = data["theta_black"]
        reported        = data["reported"]
        target_is_white = data["target_is_white"]

        for t in range(len(choices)):
            v_white = _value(theta_white[t], W_white, mus, kappa)
            v_black = _value(theta_black[t], W_black, mus, kappa)

            dq = np.clip(beta * (v_black - v_white) + c_black, -500, 500)
            p_black  = 1.0 / (1.0 + np.exp(-dq))
            p_target = (1.0 - p_black) if target_is_white[t] else p_black

            all_rows.append({
                "subject":      subject,
                "trial":        t,
                "state":        "white_high" if target_is_white[t] else "black_high",
                "chosen_color": "black" if choices[t] == 1 else "white",
                "ooz":          int(ooz[t]),
                "V_white":      v_white,
                "V_black":      v_black,
                "p_black":      p_black,
                "p_target":     p_target,
            })

            alpha = alpha_0 if ooz[t] == 0 else alpha_1
            phi_resp = _basis_vec(reported[t], mus, kappa)
            if choices[t] == 1:
                v_resp = float(np.dot(W_black, phi_resp))
                W_black += alpha * (rewards[t] - v_resp) * phi_resp
            else:
                v_resp = float(np.dot(W_white, phi_resp))
                W_white += alpha * (rewards[t] - v_resp) * phi_resp

    trial_results = pd.DataFrame(all_rows)
    if output_path is not None:
        trial_results.to_csv(output_path, index=False)
    return trial_results
