"""潜在レジームをベイズフィルタで推定するQ学習モデル（MAP推定, Route B）

state_estimation_q_learning_map_spec.md の Route B を実装する。オラクル状態
（target_item = 真の白高/黒高ラベル）をモデルに渡さず、隠れレジーム
z_t ∈ {W, B} を **報酬フィードバックからの前向きベイズフィルタ** で推定し、
信念で周辺化した価値 Q(b, a) で行動を選ぶ。

    予測（時間更新）: b_pred(k) = Σ_j P(z_t=k|z_{t-1}=j) b_{t-1}(j)
    選択（補正前信念で）: dq = beta·(Q(b_pred,black) − Q(b_pred,white)) + c_black
    補正（報酬観測後）: b(k) ∝ N(r; q(k,a), σ_r) · b_pred(k)
    価値表の学習（責任重み付き TD）: q(k,a) ← q(k,a) + α·b(k)·(r − q(k,a))  ∀k

パラメータ: alpha_0, alpha_1, beta, c_black（+ 任意で hazard h）
固定: sigma_r（報酬観測ノイズ, 既定 0.3）

--------------------------------------------------------------------------
本実験デザインに対する重要な注意（ハザード率 h と c_black の識別性）
--------------------------------------------------------------------------
experiment/index.html の条件生成では、白高/黒高（target_group）の2条件を
各 24 回複製して jsPsych.randomization.shuffle で完全ランダムに並べ替える。
実データでも試行間の切替率 ≈ 0.5、平均ラン長 ≈ 2 試行であり、**ブロック構造は
存在しない**。したがって:

  * h = 1/(平均ブロック長=48) は不正当。1/48 は「セッション全体が1ブロック
    ＝状態が48試行不変」を意味し、実際（切替率0.5）と正反対。客観的な遷移率は
    h ≈ 0.5 だが、2状態対称フィルタに h=0.5 を入れると予測ステップが常に
    b_pred=(0.5,0.5) にリセットされ、信念が履歴情報を運ばず**フィルタが不活性化**
    する（z_t を予測する信号は報酬履歴でなくその試行の運動方向にある）。

  * よって h は客観値に固定せず、「参加者が仮定する持続性」を表す**自由パラメータ
    として推定する**（既定, hazard=None）。4パラメータで q_learning_ooz_map と
    公平に BIC 比較したい場合は hazard に固定値を渡す。

  * c_black は信念状態に吸収され得る: 報酬が色間で対称なため q(W,·)≈q(B,·) に
    収束しやすく、信念が平坦だと Q(b,black)−Q(b,white) が定数化し、定数バイアス
    c_black と部分的に交絡する。緩和のため対称初期化（q=0.5, b0=0.5/0.5）を用い、
    余計な色対称性の破れを持ち込まない。パラメータリカバリでは c_black と h・価値
    非対称性の trade-off に注意する。
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist, gamma as gamma_dist, norm as norm_dist

from dualrdk.models.q_learning_map import _choice_determine

DEFAULT_SIGMA_R = 0.3


def _unpack(params: np.ndarray, hazard: Optional[float]) -> Tuple[float, float, float, float, float]:
    """パラメータベクトルを展開する。hazard=None なら末尾を h とみなす。"""
    if hazard is None:
        alpha_0, alpha_1, beta, c_black, h = params
    else:
        alpha_0, alpha_1, beta, c_black = params
        h = hazard
    return alpha_0, alpha_1, beta, c_black, h


def _run_filter(
    alpha_0: float,
    alpha_1: float,
    beta: float,
    c_black: float,
    h: float,
    choices: np.ndarray,
    rewards: np.ndarray,
    ooz_labels: np.ndarray,
    sigma_r: float,
) -> float:
    """前向きベイズフィルタ＋責任重み付き TD を1参加者分回し、対数尤度を返す。"""
    # q[regime, action]  regime 0=W, 1=B ; action 0=white, 1=black
    q = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    b = np.array([0.5, 0.5])            # 信念 b[W], b[B]
    inv_two_var = 1.0 / (2.0 * sigma_r * sigma_r)
    log_lik = 0.0

    for t in range(len(choices)):
        # --- 予測（時間更新） ---
        b_pred_W = (1.0 - h) * b[0] + h * b[1]
        b_pred_B = (1.0 - h) * b[1] + h * b[0]

        # --- 選択（補正前信念 b_pred で。因果: 今試行の報酬は未観測） ---
        Q_white = b_pred_W * q[0, 0] + b_pred_B * q[1, 0]
        Q_black = b_pred_W * q[0, 1] + b_pred_B * q[1, 1]
        dq = np.clip(beta * (Q_black - Q_white) + c_black, -500, 500)
        p_black = 1.0 / (1.0 + np.exp(-dq))

        a = int(choices[t])
        p_chosen = p_black if a == 1 else (1.0 - p_black)
        log_lik += np.log(p_chosen + 1e-300)

        # --- 補正（報酬観測後の信念更新, 選んだ行動 a で条件づけ） ---
        r = rewards[t]
        lik_W = np.exp(-((r - q[0, a]) ** 2) * inv_two_var)
        lik_B = np.exp(-((r - q[1, a]) ** 2) * inv_two_var)
        denom = lik_W * b_pred_W + lik_B * b_pred_B
        if denom > 0.0:
            b_post_W = lik_W * b_pred_W / denom
        else:                              # 数値的アンダーフロー時は予測信念を維持
            b_post_W = b_pred_W
        b_post_B = 1.0 - b_post_W
        b = np.array([b_post_W, b_post_B])

        # --- 価値表の学習（責任 b(k) で重み付けた RW 更新, 選んだ列のみ） ---
        alpha = alpha_0 if ooz_labels[t] == 0 else alpha_1
        q[0, a] += alpha * b_post_W * (r - q[0, a])
        q[1, a] += alpha * b_post_B * (r - q[1, a])

    return log_lik


def _log_prior(alpha_0: float, alpha_1: float, beta: float, c_black: float,
               h: Optional[float], fit_hazard: bool) -> float:
    lp = (
        beta_dist.logpdf(alpha_0, 2, 2)
        + beta_dist.logpdf(alpha_1, 2, 2)
        + gamma_dist.logpdf(beta, a=2, scale=3)
        + norm_dist.logpdf(c_black, 0, 2)
    )
    if fit_hazard:
        # 端に張り付かない程度の弱い事前（ほぼ一様）
        lp += beta_dist.logpdf(h, 1.2, 1.2)
    return lp


def _neg_log_posterior(
    params: np.ndarray,
    choices: np.ndarray,
    rewards: np.ndarray,
    ooz_labels: np.ndarray,
    hazard: Optional[float],
    sigma_r: float,
) -> float:
    alpha_0, alpha_1, beta, c_black, h = _unpack(params, hazard)
    log_lik = _run_filter(
        alpha_0, alpha_1, beta, c_black, h,
        choices, rewards, ooz_labels, sigma_r,
    )
    log_posterior = log_lik + _log_prior(
        alpha_0, alpha_1, beta, c_black, h, fit_hazard=hazard is None
    )
    return -log_posterior


def _compute_log_likelihood(
    alpha_0: float, alpha_1: float, beta: float, c_black: float, h: float,
    choices: np.ndarray, rewards: np.ndarray, ooz_labels: np.ndarray, sigma_r: float,
) -> float:
    return _run_filter(
        alpha_0, alpha_1, beta, c_black, h,
        choices, rewards, ooz_labels, sigma_r,
    )


def _prepare_subject(df: pd.DataFrame):
    """必要カラムを検証し (choices, rewards, ooz_labels, chosen_color, df_clean) を返す。"""
    required = {"rt", "chosen_color", "reward_points", "ooz"}
    if not required.issubset(df.columns):
        return None
    df_clean = df.dropna(subset=["rt", "chosen_color", "ooz"]).copy()
    if df_clean.empty:
        return None
    chosen_color = df_clean["chosen_color"]
    choices = _choice_determine(chosen_color)
    rewards = (df_clean["reward_points"].values / 10).astype(float)
    ooz_labels = df_clean["ooz"].values.astype(int)
    return choices, rewards, ooz_labels, chosen_color, df_clean


def fit_state_estimation_q_learning_map(
    concat_list: List[Tuple[str, pd.DataFrame]],
    hazard: Optional[float] = None,
    sigma_r: float = DEFAULT_SIGMA_R,
    n_alpha0_grid: int = 4,
    n_alpha1_grid: int = 4,
    n_beta_grid: int = 4,
    n_c_grid: int = 4,
    n_h_grid: int = 3,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """各参加者の状態推定 Q学習パラメータを MAP 推定する（Route B）。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        df は rt, chosen_color, reward_points, ooz カラムを含む（target_item は不要）。
    hazard : float or None
        レジーム切替のハザード率 h。None（既定）なら h も自由パラメータとして推定
        （末尾に付加、計5パラメータ）。固定したい場合は 0<h<1 を渡す（計4パラメータ、
        q_learning_ooz_map と同数）。**モジュール docstring の注意（1/48 は不正当）参照。**
    sigma_r : float
        報酬観測ノイズ幅（固定）。
    n_*_grid : int
        グリッド初期値の分割数。n_h_grid は hazard=None のときのみ使用。
    output_path : str or None
        指定時 CSV 保存。

    Returns
    -------
    pd.DataFrame
        subject, alpha_0, alpha_1, beta, c_black, hazard, sigma_r,
        log_posterior, log_likelihood, n_params, n_trials, n_ooz, n_non_ooz。
    """
    fit_hazard = hazard is None
    records = []

    alpha0_inits = np.linspace(0.1, 0.9, n_alpha0_grid)
    alpha1_inits = np.linspace(0.1, 0.9, n_alpha1_grid)
    beta_inits = np.linspace(0.5, 10.0, n_beta_grid)
    c_inits = np.linspace(-2.0, 2.0, n_c_grid)
    h_inits = np.linspace(0.05, 0.5, n_h_grid) if fit_hazard else [None]

    bounds = [
        (1e-6, 1 - 1e-6),   # alpha_0
        (1e-6, 1 - 1e-6),   # alpha_1
        (1e-6, None),       # beta
        (None, None),       # c_black
    ]
    if fit_hazard:
        bounds.append((1e-3, 1 - 1e-3))   # hazard

    for subj_id, df in concat_list:
        prepared = _prepare_subject(df)
        if prepared is None:
            continue
        choices, rewards, ooz_labels, _, _ = prepared
        n_trials = len(choices)

        best_neg_lp = np.inf
        best_params = None
        for a0 in alpha0_inits:
            print(f"Subject {subj_id}: alpha_0={a0:.3f}")
            for a1 in alpha1_inits:
                for b0 in beta_inits:
                    for c0 in c_inits:
                        for h0 in h_inits:
                            x0 = [a0, a1, b0, c0] + ([h0] if fit_hazard else [])
                            result = minimize(
                                _neg_log_posterior,
                                x0=x0,
                                args=(choices, rewards, ooz_labels, hazard, sigma_r),
                                method="L-BFGS-B",
                                bounds=bounds,
                            )
                            if result.fun < best_neg_lp:
                                best_neg_lp = result.fun
                                best_params = result.x

        a0_hat, a1_hat, beta_hat, c_hat, h_hat = _unpack(best_params, hazard)
        log_posterior = -best_neg_lp
        log_likelihood = _compute_log_likelihood(
            a0_hat, a1_hat, beta_hat, c_hat, h_hat,
            choices, rewards, ooz_labels, sigma_r,
        )

        records.append({
            "subject":        subj_id,
            "alpha_0":        a0_hat,
            "alpha_1":        a1_hat,
            "beta":           beta_hat,
            "c_black":        c_hat,
            "hazard":         h_hat,
            "sigma_r":        sigma_r,
            "log_posterior":  log_posterior,
            "log_likelihood": log_likelihood,
            "n_params":       5 if fit_hazard else 4,
            "n_trials":       n_trials,
            "n_ooz":          int(ooz_labels.sum()),
            "n_non_ooz":      int((ooz_labels == 0).sum()),
        })

    results = pd.DataFrame(records)
    if output_path is not None:
        results.to_csv(output_path, index=False)
    return results


def predict_target_choice_probs(
    concat_list: List[Tuple[str, pd.DataFrame]],
    results_df: pd.DataFrame,
    sigma_r: float = DEFAULT_SIGMA_R,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """推定パラメータで各試行のターゲット選択確率と信念軌跡を復元する（Route B）。

    フィルタは fitting と同一（オラクル target_item は使わない）。target_item /
    target_group は **事後の突き合わせ専用**で、ターゲット選択確率 p_target の
    再構成と、信念 b_W の妥当性チェックにのみ用いる（学習・選択には渡さない）。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        df は rt, chosen_color, reward_points, ooz を含む。target_item /
        target_group があれば p_target と state の付与に使う。
    results_df : pd.DataFrame
        fit_state_estimation_q_learning_map の出力
        （alpha_0, alpha_1, beta, c_black, hazard, sigma_r が必要）。
    sigma_r : float
        results_df に sigma_r 列が無い場合のフォールバック。
    output_path : str or None

    Returns
    -------
    pd.DataFrame
        subject, trial, ooz, state（あれば）, chosen_color,
        b_pred_W, b_post_W（補正前/後の白高信念）,
        Q_white, Q_black, p_black, p_target（あれば）,
        q_W_white, q_W_black, q_B_white, q_B_black。
    """
    needed = ["alpha_0", "alpha_1", "beta", "c_black", "hazard"]
    param_map = results_df.set_index("subject")[
        needed + (["sigma_r"] if "sigma_r" in results_df.columns else [])
    ].to_dict("index")
    all_rows = []

    for subject, df in concat_list:
        if subject not in param_map:
            continue
        prepared = _prepare_subject(df)
        if prepared is None:
            continue
        choices, rewards, ooz_labels, chosen_color, df_clean = prepared

        p = param_map[subject]
        alpha_0, alpha_1, beta, c_black, h = (
            p["alpha_0"], p["alpha_1"], p["beta"], p["c_black"], p["hazard"]
        )
        s_r = p.get("sigma_r", sigma_r) if isinstance(p, dict) else sigma_r

        has_state = "target_item" in df_clean.columns
        states = df_clean["target_item"].values.astype(int) if has_state else None
        has_group = "target_group" in df_clean.columns
        groups = df_clean["target_group"].astype(str).values if has_group else None

        # フィルタを再実行し、各試行の q テーブルも記録する
        q = np.array([[0.5, 0.5], [0.5, 0.5]])
        b = np.array([0.5, 0.5])
        inv_two_var = 1.0 / (2.0 * s_r * s_r)

        for t in range(len(choices)):
            b_pred_W = (1.0 - h) * b[0] + h * b[1]
            b_pred_B = (1.0 - h) * b[1] + h * b[0]

            Q_white = b_pred_W * q[0, 0] + b_pred_B * q[1, 0]
            Q_black = b_pred_W * q[0, 1] + b_pred_B * q[1, 1]
            dq = np.clip(beta * (Q_black - Q_white) + c_black, -500, 500)
            p_black = 1.0 / (1.0 + np.exp(-dq))

            row = {
                "subject":      subject,
                "trial":        t,
                "ooz":          int(ooz_labels[t]),
                "chosen_color": chosen_color.iloc[t],
                "b_pred_W":     b_pred_W,
                "Q_white":      Q_white,
                "Q_black":      Q_black,
                "p_black":      p_black,
                "q_W_white":    q[0, 0],
                "q_W_black":    q[0, 1],
                "q_B_white":    q[1, 0],
                "q_B_black":    q[1, 1],
            }
            if has_state:
                s = int(states[t])
                row["state"] = "white_high" if s == 0 else "black_high"
            if has_group:
                # ターゲット選択確率: 白がターゲットなら P(white)=1-p_black
                row["p_target"] = (1.0 - p_black) if groups[t] == "white" else p_black

            # --- 補正（信念更新） ---
            a = int(choices[t])
            r = rewards[t]
            lik_W = np.exp(-((r - q[0, a]) ** 2) * inv_two_var)
            lik_B = np.exp(-((r - q[1, a]) ** 2) * inv_two_var)
            denom = lik_W * b_pred_W + lik_B * b_pred_B
            b_post_W = lik_W * b_pred_W / denom if denom > 0.0 else b_pred_W
            b_post_B = 1.0 - b_post_W
            b = np.array([b_post_W, b_post_B])
            row["b_post_W"] = b_post_W

            # --- 価値表の学習 ---
            alpha = alpha_0 if ooz_labels[t] == 0 else alpha_1
            q[0, a] += alpha * b_post_W * (r - q[0, a])
            q[1, a] += alpha * b_post_B * (r - q[1, a])

            all_rows.append(row)

    trial_results = pd.DataFrame(all_rows)
    if output_path is not None:
        trial_results.to_csv(output_path, index=False)
    return trial_results
