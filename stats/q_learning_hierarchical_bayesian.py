"""階層ベイズQ学習モデル（emcee によるMCMC推定）

Non-centered parameterization:
    logit(alpha_i) = mu_alpha + sigma_alpha * z_alpha_i,  z_alpha_i ~ N(0,1)
    log(beta_i)    = mu_beta  + sigma_beta  * z_beta_i,   z_beta_i  ~ N(0,1)

パラメータベクトルのレイアウト（長さ 4 + 2*N）:
    [0]       mu_alpha
    [1]       log_sigma_alpha   (sigma_alpha = exp(...) > 0 を保証)
    [2]       mu_beta
    [3]       log_sigma_beta
    [4..4+N)  z_alpha[i]
    [4+N..4+2N)  z_beta[i]
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import emcee
from scipy.special import expit  # sigmoid


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
# 対数事後確率
# ---------------------------------------------------------------------------

def _log_likelihood_single(alpha: float, beta: float,
                            choices: np.ndarray, rewards: np.ndarray, states: np.ndarray) -> float:
    q = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    ll = 0.0
    for t in range(len(choices)):
        s = int(states[t])
        dq = np.clip(beta * (q[s, 0] - q[s, 1]), -500, 500)
        p_target = expit(dq)
        a = int(choices[t])
        ll += np.log((p_target if a == 1 else 1.0 - p_target) + 1e-300)
        q[s, a] += alpha * (rewards[t] - q[s, a])
    return ll


def _make_log_posterior(choices_list: List[np.ndarray], rewards_list: List[np.ndarray], states_list: List[np.ndarray]):
    """クロージャとして対数事後確率関数を生成する。"""
    N = len(choices_list)

    def log_posterior(params: np.ndarray) -> float:
        mu_alpha     = params[0]
        log_sig_a    = params[1]
        mu_beta      = params[2]
        log_sig_b    = params[3]
        z_alpha      = params[4     : 4 + N]
        z_beta       = params[4 + N : 4 + 2 * N]

        sigma_alpha = np.exp(log_sig_a)
        sigma_beta  = np.exp(log_sig_b)

        # ハイパー事前分布
        # mu_alpha ~ N(0, 1.5^2)
        lp  = -0.5 * (mu_alpha / 1.5) ** 2
        # sigma_alpha ~ Half-Cauchy(0,1)、log_sigma_alpha で再パラメータ化
        lp += np.log(2.0 / np.pi) - np.log(1.0 + sigma_alpha ** 2) + log_sig_a
        # mu_beta ~ N(0.5, 1.5^2)
        lp += -0.5 * ((mu_beta - 0.5) / 1.5) ** 2
        # sigma_beta ~ Half-Cauchy(0,1)
        lp += np.log(2.0 / np.pi) - np.log(1.0 + sigma_beta ** 2) + log_sig_b

        # z 事前分布（標準正規）
        lp += -0.5 * np.sum(z_alpha ** 2)
        lp += -0.5 * np.sum(z_beta  ** 2)

        # 尤度
        for i in range(N):
            logit_ai = mu_alpha + sigma_alpha * z_alpha[i]
            log_bi   = mu_beta  + sigma_beta  * z_beta[i]
            alpha_i  = float(expit(logit_ai))
            beta_i   = float(np.exp(np.clip(log_bi, -10, 10)))
            lp += _log_likelihood_single(alpha_i, beta_i, choices_list[i], rewards_list[i], states_list[i])

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
        各タプルは (subject, df)。df は rt, chosen_item, reward_points, target_item カラムを含む。
    nwalkers : int or None
        emcee のウォーカー数。None の場合は max(64, 4*(4+2*N)) の偶数に自動設定。
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
        個人レベルのパラメータの事後統計量。
    traces : dict
        'subj_ids', 'alpha_samples', 'beta_samples',
        'mu_alpha_samples', 'sigma_alpha_samples',
        'mu_beta_samples', 'sigma_beta_samples' を含む。
    """
    rng = np.random.default_rng(random_seed)

    # 前処理
    subjects = []
    choices_list = []
    rewards_list = []
    states_list = []
    n_trials_list = []

    for subject, df in concat_list:
        df_clean = df.dropna(subset=["rt"]).copy()
        choices = df_clean["chosen_item"].values.astype(int)
        rewards = (df_clean["reward_points"].values >= 1).astype(float)
        states = df_clean["target_item"].values.astype(int)
        subjects.append(subject)
        choices_list.append(choices)
        rewards_list.append(rewards)
        states_list.append(states)
        n_trials_list.append(len(choices))

    N = len(subjects)
    ndim = 4 + 2 * N

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
    p0[:, 0] = rng.normal(0.0,  0.05, size=nwalkers)   # mu_alpha
    p0[:, 1] = rng.normal(0.0,  0.05, size=nwalkers)   # log_sigma_alpha
    p0[:, 2] = rng.normal(0.5,  0.05, size=nwalkers)   # mu_beta
    p0[:, 3] = rng.normal(0.0,  0.05, size=nwalkers)   # log_sigma_beta
    p0[:, 4:4+N]     = rng.normal(0.0, 0.05, size=(nwalkers, N))  # z_alpha
    p0[:, 4+N:4+2*N] = rng.normal(0.0, 0.05, size=(nwalkers, N))  # z_beta

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    # バーンイン
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()

    # 本サンプリング
    sampler.run_mcmc(state, nsamples, progress=True)

    # サンプル取得: (nwalkers, nsamples, ndim) に整形
    raw = sampler.get_chain(flat=False)          # (nsamples, nwalkers, ndim)
    raw = raw.transpose(1, 0, 2)                  # (nwalkers, nsamples, ndim)

    # ------------------------------------------------------------------
    # ハイパーパラメータのサンプル（フラット）
    # ------------------------------------------------------------------
    mu_alpha_samp     = raw[:, :, 0].flatten()
    sigma_alpha_samp  = np.exp(raw[:, :, 1].flatten())
    mu_beta_samp      = raw[:, :, 2].flatten()
    sigma_beta_samp   = np.exp(raw[:, :, 3].flatten())

    # ------------------------------------------------------------------
    # 個人パラメータのサンプル
    # ------------------------------------------------------------------
    # z_alpha[i]: raw[:, :, 4+i],  z_beta[i]: raw[:, :, 4+N+i]
    # alpha_i = sigmoid(mu_alpha + sigma_alpha * z_alpha_i)
    # beta_i  = exp(mu_beta  + sigma_beta  * z_beta_i)
    mu_a_w    = raw[:, :, 0]          # (nwalkers, nsamples)
    sig_a_w   = np.exp(raw[:, :, 1])
    mu_b_w    = raw[:, :, 2]
    sig_b_w   = np.exp(raw[:, :, 3])

    all_alpha_samples = []
    all_beta_samples  = []
    for i in range(N):
        za = raw[:, :, 4 + i]
        zb = raw[:, :, 4 + N + i]
        alpha_i_samp = expit(mu_a_w + sig_a_w * za).flatten()
        beta_i_samp  = np.exp(np.clip(mu_b_w + sig_b_w * zb, -10, 10)).flatten()
        all_alpha_samples.append(alpha_i_samp)
        all_beta_samples.append(beta_i_samp)

    # 集団レベルの事後平均 alpha・beta（shrinkage 基準）
    group_alpha_mean = float(np.mean(expit(mu_alpha_samp)))
    group_beta_mean  = float(np.mean(np.exp(np.clip(mu_beta_samp, -10, 10))))

    # ------------------------------------------------------------------
    # R-hat（ウォーカーを n_rhat_chains グループに分割）
    # ------------------------------------------------------------------
    wpc = nwalkers // n_rhat_chains  # walkers per chain

    def _rhat_param(walker_idx: int) -> float:
        chains = raw[:wpc * n_rhat_chains, :, walker_idx].reshape(n_rhat_chains, -1)
        return _r_hat(chains)

    # ------------------------------------------------------------------
    # 集団レベル結果
    # ------------------------------------------------------------------
    group_record = {}
    for name, samp in [
        ("mu_alpha",    mu_alpha_samp),
        ("sigma_alpha", sigma_alpha_samp),
        ("mu_beta",     mu_beta_samp),
        ("sigma_beta",  sigma_beta_samp),
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

        a_lo, a_hi = _hdi(a_samp)
        b_lo, b_hi = _hdi(b_samp)

        # R-hat: z_alpha[i] チェイン
        za_chains = raw[:wpc * n_rhat_chains, :, 4 + i].reshape(n_rhat_chains, -1)
        zb_chains = raw[:wpc * n_rhat_chains, :, 4 + N + i].reshape(n_rhat_chains, -1)
        # R-hat は変換後のスケールで計算（alpha, beta）
        alpha_chains = expit(
            raw[:wpc * n_rhat_chains, :, 0].reshape(n_rhat_chains, -1)
            + np.exp(raw[:wpc * n_rhat_chains, :, 1].reshape(n_rhat_chains, -1)) * za_chains
        )
        beta_chains = np.exp(np.clip(
            raw[:wpc * n_rhat_chains, :, 2].reshape(n_rhat_chains, -1)
            + np.exp(raw[:wpc * n_rhat_chains, :, 3].reshape(n_rhat_chains, -1)) * zb_chains,
            -10, 10
        ))

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
            "r_hat_alpha":     _r_hat(alpha_chains),
            "r_hat_beta":      _r_hat(beta_chains),
            "shrinkage_alpha": float(np.mean(a_samp)) - group_alpha_mean,
            "shrinkage_beta":  float(np.mean(b_samp)) - group_beta_mean,
            "n_trials":        n_trials_list[i],
        })

    individual_results = pd.DataFrame(ind_records)

    if output_path is not None:
        individual_results.to_csv(output_path, index=False)

    traces = {
        "subj_ids":           subjects,
        "alpha_samples":      np.stack(all_alpha_samples, axis=0),   # (N, n_total_samples)
        "beta_samples":       np.stack(all_beta_samples,  axis=0),
        "mu_alpha_samples":   mu_alpha_samp,
        "sigma_alpha_samples": sigma_alpha_samp,
        "mu_beta_samples":    mu_beta_samp,
        "sigma_beta_samples": sigma_beta_samp,
    }

    return group_results, individual_results, traces
