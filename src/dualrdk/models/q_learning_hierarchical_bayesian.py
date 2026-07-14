"""階層ベイズQ学習モデル（emcee によるMCMC推定）

Non-centered parameterization:
    logit(alpha_i)  = mu_alpha + sigma_alpha * z_alpha_i,  z_alpha_i ~ N(0,1)
    log(beta_i)     = mu_beta  + sigma_beta  * z_beta_i,   z_beta_i  ~ N(0,1)
    c_black_i       = mu_c     + sigma_c     * z_c_i,      z_c_i     ~ N(0,1)

パラメータベクトルのレイアウト（長さ 6 + 3*N）:
    [0]           mu_alpha
    [1]           log_sigma_alpha   (sigma_alpha = exp(...) > 0 を保証)
    [2]           mu_beta
    [3]           log_sigma_beta
    [4]           mu_c
    [5]           log_sigma_c
    [6..6+N)      z_alpha[i]
    [6+N..6+2N)   z_beta[i]
    [6+2N..6+3N)  z_c[i]

選択確率:
    dq = beta_i * (Q[s,1] - Q[s,0]) + c_black_i   (1=黒, 0=白)
    state=0 (白=ターゲット): P(target) = P(白) = sigmoid(-dq)
    state=1 (黒=ターゲット): P(target) = P(黒) = sigmoid(+dq)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import emcee
from scipy.special import expit  # sigmoid

from dualrdk.models.q_learning_map import _choice_determine


# ---------------------------------------------------------------------------
# 数値ユーティリティ
# ---------------------------------------------------------------------------

def _hdi(samples: np.ndarray, credible_mass: float = 0.95) -> Tuple[float, float]:
    n = len(samples)
    sorted_s = np.sort(samples)
    width = int(np.floor(credible_mass * n))
    diffs = sorted_s[width:] - sorted_s[: n - width]
    idx = np.argmin(diffs)
    return float(sorted_s[idx]), float(sorted_s[idx + width])


def _r_hat(chains: np.ndarray) -> float:
    """Gelman-Rubin R-hat。chains: shape (n_chains, n_samples)"""
    m, n = chains.shape
    B = n * np.var(chains.mean(axis=1), ddof=1)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    if W == 0:
        return np.nan
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


# ---------------------------------------------------------------------------
# 対数尤度（MAP モデルと同一ロジック）
# ---------------------------------------------------------------------------

def _log_likelihood_single(
    alpha: float, beta: float, c_black: float,
    choices: np.ndarray, rewards: np.ndarray, states: np.ndarray
) -> float:
    """1参加者分の対数尤度を計算する。

    choices : 0=白, 1=黒  (chosen_color から変換済み)
    states  : 0=白ターゲット(PatternB), 1=黒ターゲット(PatternA)  (target_item)
    rewards : reward_points / 10
    """
    q = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    ll = 0.0
    for t in range(len(choices)):
        s = int(states[t])
        # 「黒 vs 白」log-odds + 色バイアス
        dq = beta * (q[s, 1] - q[s, 0]) + c_black
        dq = np.clip(dq, -500, 500)
        if s == 0:  # 白=ターゲット
            p_target      = 1.0 / (1.0 + np.exp(dq))   # P(白)
            target_action = 0
        else:       # 黒=ターゲット
            p_target      = 1.0 / (1.0 + np.exp(-dq))  # P(黒)
            target_action = 1
        a = int(choices[t])
        p_chosen = p_target if a == target_action else (1.0 - p_target)
        ll += np.log(p_chosen + 1e-300)
        q[s, a] += alpha * (rewards[t] - q[s, a])
    return ll


# ---------------------------------------------------------------------------
# 対数事後確率（クロージャ）
# ---------------------------------------------------------------------------

def _make_log_posterior(
    choices_list: List[np.ndarray],
    rewards_list: List[np.ndarray],
    states_list:  List[np.ndarray],
):
    N = len(choices_list)

    def log_posterior(params: np.ndarray) -> float:
        mu_alpha  = params[0]
        log_sig_a = params[1]
        mu_beta   = params[2]
        log_sig_b = params[3]
        mu_c      = params[4]
        log_sig_c = params[5]
        z_alpha   = params[6         : 6 + N]
        z_beta    = params[6 + N     : 6 + 2 * N]
        z_c       = params[6 + 2 * N : 6 + 3 * N]

        sigma_alpha = np.exp(log_sig_a)
        sigma_beta  = np.exp(log_sig_b)
        sigma_c     = np.exp(log_sig_c)

        # ハイパー事前分布
        # mu_alpha ~ N(0, 1.5^2)
        lp  = -0.5 * (mu_alpha / 1.5) ** 2
        # sigma_alpha ~ Half-Cauchy(0,1)
        lp += np.log(2.0 / np.pi) - np.log(1.0 + sigma_alpha ** 2) + log_sig_a
        # mu_beta ~ N(0.5, 1.5^2)
        lp += -0.5 * ((mu_beta - 0.5) / 1.5) ** 2
        # sigma_beta ~ Half-Cauchy(0,1)
        lp += np.log(2.0 / np.pi) - np.log(1.0 + sigma_beta ** 2) + log_sig_b
        # mu_c ~ N(0, 1.5^2)
        lp += -0.5 * (mu_c / 1.5) ** 2
        # sigma_c ~ Half-Cauchy(0,1)
        lp += np.log(2.0 / np.pi) - np.log(1.0 + sigma_c ** 2) + log_sig_c

        # z 事前分布（標準正規）
        lp += -0.5 * np.sum(z_alpha ** 2)
        lp += -0.5 * np.sum(z_beta  ** 2)
        lp += -0.5 * np.sum(z_c     ** 2)

        # 尤度
        for i in range(N):
            logit_ai = mu_alpha + sigma_alpha * z_alpha[i]
            log_bi   = mu_beta  + sigma_beta  * z_beta[i]
            c_i      = mu_c     + sigma_c     * z_c[i]
            alpha_i  = float(expit(logit_ai))
            beta_i   = float(np.exp(np.clip(log_bi, -10, 10)))
            lp += _log_likelihood_single(
                alpha_i, beta_i, c_i,
                choices_list[i], rewards_list[i], states_list[i]
            )

        return lp

    return log_posterior


# ---------------------------------------------------------------------------
# 公開インタフェース
# ---------------------------------------------------------------------------

def fit_q_learning_hierarchical_bayesian(
    concat_list: List[Tuple[str, pd.DataFrame]],
    nwalkers: int = None,
    nburn: int = 2000,
    nsamples: int = 4000,
    n_rhat_chains: int = 4,
    random_seed: int = 0,
    output_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """全参加者のQ学習パラメータを階層ベイズモデルで同時推定する。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        各タプルは (subject, df)。df は rt, chosen_color, reward_points,
        target_item カラムを含む。
    nwalkers : int or None
        emcee のウォーカー数。None の場合は max(64, 4*(6+3*N)) の偶数に自動設定。
    nburn : int
        バーンインステップ数（デフォルト 2000）。
    nsamples : int
        バーンイン後のサンプリングステップ数（デフォルト 4000）。
    n_rhat_chains : int
        R-hat 計算のためにウォーカーを分割するグループ数。
    random_seed : int
        乱数シード。
    output_path : str or None
        指定した場合、individual_results を CSV として保存する。

    Returns
    -------
    group_results : pd.DataFrame (1行)
        集団レベルのハイパーパラメータの事後統計量。
    individual_results : pd.DataFrame (N行)
        個人レベルのパラメータ（alpha, beta, c_black）の事後統計量。
    traces : dict
        'subj_ids', 'alpha_samples', 'beta_samples', 'c_samples',
        'mu_alpha_samples', 'sigma_alpha_samples',
        'mu_beta_samples', 'sigma_beta_samples',
        'mu_c_samples', 'sigma_c_samples' を含む。
    """
    rng = np.random.default_rng(random_seed)

    # 前処理（MAP モデルと同一: rt と chosen_color が両方有効な行のみ）
    subjects      = []
    choices_list  = []
    rewards_list  = []
    states_list   = []
    n_trials_list = []

    for subject, df in concat_list:
        df_clean = df.dropna(subset=["rt", "chosen_color"]).copy()
        choices  = _choice_determine(df_clean["chosen_color"])
        rewards  = (df_clean["reward_points"].values / 10).astype(float)
        states   = df_clean["target_item"].values.astype(int)
        subjects.append(subject)
        choices_list.append(choices)
        rewards_list.append(rewards)
        states_list.append(states)
        n_trials_list.append(len(choices))

    N    = len(subjects)
    ndim = 6 + 3 * N   # mu_alpha, log_sig_a, mu_beta, log_sig_b, mu_c, log_sig_c + 3*N

    if nwalkers is None:
        nwalkers = max(64, 4 * ndim)
        if nwalkers % 2 != 0:
            nwalkers += 1

    assert nwalkers > 2 * ndim, (
        f"nwalkers ({nwalkers}) must be > 2*ndim ({2*ndim}). Increase nwalkers."
    )
    assert nwalkers % 2 == 0, "nwalkers must be even."

    log_posterior = _make_log_posterior(choices_list, rewards_list, states_list)

    # ウォーカーの初期位置
    p0 = np.zeros((nwalkers, ndim))
    p0[:, 0] = rng.normal(0.0, 0.05, size=nwalkers)   # mu_alpha
    p0[:, 1] = rng.normal(0.0, 0.05, size=nwalkers)   # log_sigma_alpha
    p0[:, 2] = rng.normal(0.5, 0.05, size=nwalkers)   # mu_beta
    p0[:, 3] = rng.normal(0.0, 0.05, size=nwalkers)   # log_sigma_beta
    p0[:, 4] = rng.normal(0.0, 0.05, size=nwalkers)   # mu_c
    p0[:, 5] = rng.normal(0.0, 0.05, size=nwalkers)   # log_sigma_c
    p0[:, 6       : 6 + N]     = rng.normal(0.0, 0.05, size=(nwalkers, N))  # z_alpha
    p0[:, 6 + N   : 6 + 2 * N] = rng.normal(0.0, 0.05, size=(nwalkers, N))  # z_beta
    p0[:, 6 + 2*N : 6 + 3 * N] = rng.normal(0.0, 0.05, size=(nwalkers, N))  # z_c

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    # バーンイン
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()

    # 本サンプリング
    sampler.run_mcmc(state, nsamples, progress=True)

    # サンプル取得: (nwalkers, nsamples, ndim)
    raw = sampler.get_chain(flat=False)   # (nsamples, nwalkers, ndim)
    raw = raw.transpose(1, 0, 2)          # (nwalkers, nsamples, ndim)

    # ------------------------------------------------------------------
    # ハイパーパラメータのサンプル（フラット）
    # ------------------------------------------------------------------
    mu_alpha_samp    = raw[:, :, 0].flatten()
    sigma_alpha_samp = np.exp(raw[:, :, 1].flatten())
    mu_beta_samp     = raw[:, :, 2].flatten()
    sigma_beta_samp  = np.exp(raw[:, :, 3].flatten())
    mu_c_samp        = raw[:, :, 4].flatten()
    sigma_c_samp     = np.exp(raw[:, :, 5].flatten())

    # ------------------------------------------------------------------
    # 個人パラメータのサンプル
    # ------------------------------------------------------------------
    mu_a_w  = raw[:, :, 0]
    sig_a_w = np.exp(raw[:, :, 1])
    mu_b_w  = raw[:, :, 2]
    sig_b_w = np.exp(raw[:, :, 3])
    mu_c_w  = raw[:, :, 4]
    sig_c_w = np.exp(raw[:, :, 5])

    all_alpha_samples = []
    all_beta_samples  = []
    all_c_samples     = []
    for i in range(N):
        za = raw[:, :, 6 + i]
        zb = raw[:, :, 6 + N + i]
        zc = raw[:, :, 6 + 2 * N + i]
        alpha_i_samp = expit(mu_a_w + sig_a_w * za).flatten()
        beta_i_samp  = np.exp(np.clip(mu_b_w + sig_b_w * zb, -10, 10)).flatten()
        c_i_samp     = (mu_c_w + sig_c_w * zc).flatten()
        all_alpha_samples.append(alpha_i_samp)
        all_beta_samples.append(beta_i_samp)
        all_c_samples.append(c_i_samp)

    # 集団レベルの事後平均（shrinkage 基準）
    group_alpha_mean = float(np.mean(expit(mu_alpha_samp)))
    group_beta_mean  = float(np.mean(np.exp(np.clip(mu_beta_samp, -10, 10))))
    group_c_mean     = float(np.mean(mu_c_samp))

    # ------------------------------------------------------------------
    # R-hat（ウォーカーを n_rhat_chains グループに分割）
    # ------------------------------------------------------------------
    wpc = nwalkers // n_rhat_chains  # walkers per chain

    # ------------------------------------------------------------------
    # 集団レベル結果
    # ------------------------------------------------------------------
    group_record = {}
    for name, samp in [
        ("mu_alpha",    mu_alpha_samp),
        ("sigma_alpha", sigma_alpha_samp),
        ("mu_beta",     mu_beta_samp),
        ("sigma_beta",  sigma_beta_samp),
        ("mu_c",        mu_c_samp),
        ("sigma_c",     sigma_c_samp),
    ]:
        lo, hi = _hdi(samp)
        group_record[f"{name}_mean"]     = float(np.mean(samp))
        group_record[f"{name}_hdi_low"]  = lo
        group_record[f"{name}_hdi_high"] = hi

    group_results = pd.DataFrame([group_record])

    # ------------------------------------------------------------------
    # 個人レベル結果
    # ------------------------------------------------------------------
    ind_records = []
    for i, subject in enumerate(subjects):
        a_samp = all_alpha_samples[i]
        b_samp = all_beta_samples[i]
        c_samp = all_c_samples[i]

        a_lo, a_hi = _hdi(a_samp)
        b_lo, b_hi = _hdi(b_samp)
        c_lo, c_hi = _hdi(c_samp)

        # R-hat: 変換後スケールで計算
        za_chains = raw[:wpc * n_rhat_chains, :, 6 + i].reshape(n_rhat_chains, -1)
        zb_chains = raw[:wpc * n_rhat_chains, :, 6 + N + i].reshape(n_rhat_chains, -1)
        zc_chains = raw[:wpc * n_rhat_chains, :, 6 + 2 * N + i].reshape(n_rhat_chains, -1)

        alpha_chains = expit(
            raw[:wpc * n_rhat_chains, :, 0].reshape(n_rhat_chains, -1)
            + np.exp(raw[:wpc * n_rhat_chains, :, 1].reshape(n_rhat_chains, -1)) * za_chains
        )
        beta_chains = np.exp(np.clip(
            raw[:wpc * n_rhat_chains, :, 2].reshape(n_rhat_chains, -1)
            + np.exp(raw[:wpc * n_rhat_chains, :, 3].reshape(n_rhat_chains, -1)) * zb_chains,
            -10, 10
        ))
        c_chains = (
            raw[:wpc * n_rhat_chains, :, 4].reshape(n_rhat_chains, -1)
            + np.exp(raw[:wpc * n_rhat_chains, :, 5].reshape(n_rhat_chains, -1)) * zc_chains
        )

        ind_records.append({
            "subject":         subject,
            "alpha_mean":      float(np.mean(a_samp)),
            "alpha_median":    float(np.median(a_samp)),
            "alpha_hdi_low":   a_lo,
            "alpha_hdi_high":  a_hi,
            "beta_mean":       float(np.mean(b_samp)),
            "beta_median":     float(np.median(b_samp)),
            "beta_hdi_low":    b_lo,
            "beta_hdi_high":   b_hi,
            "c_black_mean":    float(np.mean(c_samp)),
            "c_black_median":  float(np.median(c_samp)),
            "c_black_hdi_low":  c_lo,
            "c_black_hdi_high": c_hi,
            "r_hat_alpha":     _r_hat(alpha_chains),
            "r_hat_beta":      _r_hat(beta_chains),
            "r_hat_c":         _r_hat(c_chains),
            "shrinkage_alpha": float(np.mean(a_samp)) - group_alpha_mean,
            "shrinkage_beta":  float(np.mean(b_samp)) - group_beta_mean,
            "shrinkage_c":     float(np.mean(c_samp)) - group_c_mean,
            "n_trials":        n_trials_list[i],
        })

    individual_results = pd.DataFrame(ind_records)

    if output_path is not None:
        individual_results.to_csv(output_path, index=False)

    traces = {
        "subj_ids":            subjects,
        "alpha_samples":       np.stack(all_alpha_samples, axis=0),  # (N, n_total)
        "beta_samples":        np.stack(all_beta_samples,  axis=0),
        "c_samples":           np.stack(all_c_samples,     axis=0),
        "mu_alpha_samples":    mu_alpha_samp,
        "sigma_alpha_samples": sigma_alpha_samp,
        "mu_beta_samples":     mu_beta_samp,
        "sigma_beta_samples":  sigma_beta_samp,
        "mu_c_samples":        mu_c_samp,
        "sigma_c_samples":     sigma_c_samp,
    }

    return group_results, individual_results, traces


def predict_target_choice_probs(
    concat_list: List[Tuple[str, "pd.DataFrame"]],
    results_df: "pd.DataFrame",
    output_path: Optional[str] = None,
) -> "pd.DataFrame":
    """推定パラメータで各試行のターゲット選択確率を予測する（階層ベイズ版）

    results_df には alpha_mean, beta_mean, c_black_mean カラムが必要。
    """
    param_map = results_df.set_index("subject")[["alpha_mean", "beta_mean", "c_black_mean"]].to_dict("index")
    all_rows = []

    for subject, df in concat_list:
        if subject not in param_map:
            continue
        alpha   = param_map[subject]["alpha_mean"]
        beta    = param_map[subject]["beta_mean"]
        c_black = param_map[subject]["c_black_mean"]

        df_clean = df.dropna(subset=["rt", "chosen_color"]).copy()
        required = {"rt", "chosen_color", "target_item", "reward_points"}
        if not required.issubset(df_clean.columns):
            continue

        chosen_color = df_clean["chosen_color"]
        choices      = _choice_determine(chosen_color)
        states       = df_clean["target_item"].values.astype(int)
        rewards      = (df_clean["reward_points"].values / 10).astype(float)

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
                "Q_patternA_white":  q[1, 0],
                "Q_patternA_black":  q[1, 1],
                "Q_patternB_white":  q[0, 0],
                "Q_patternB_black":  q[0, 1],
                "p_patternA":        p_patternA,
                "p_patternB":        p_patternB,
            })

            a = int(choices[t])
            r = rewards[t]
            q[s, a] += alpha * (r - q[s, a])

    trial_results = pd.DataFrame(all_rows)
    if output_path is not None:
        trial_results.to_csv(output_path, index=False)
    return trial_results
