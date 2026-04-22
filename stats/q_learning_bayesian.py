from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import emcee
from scipy.stats import beta as beta_dist, gamma as gamma_dist


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _log_likelihood(params: np.ndarray, choices: np.ndarray, rewards: np.ndarray, states: np.ndarray) -> float:
    alpha, beta = params
    q = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    log_lik = 0.0
    for t in range(len(choices)):
        s = int(states[t])
        dq = np.clip(beta * (q[s, 0] - q[s, 1]), -500, 500)
        p_target = 1.0 / (1.0 + np.exp(-dq))
        a = int(choices[t])
        p_chosen = p_target if a == 1 else (1.0 - p_target)
        log_lik += np.log(p_chosen + 1e-300)
        q[s, a] += alpha * (rewards[t] - q[s, a])
    return log_lik


def _log_prior(params: np.ndarray) -> float:
    alpha, beta = params
    if not (0.0 < alpha < 1.0) or not (beta > 0.0):
        return -np.inf
    return beta_dist.logpdf(alpha, 2, 2) + gamma_dist.logpdf(beta, a=2, scale=3)


def _log_posterior(params: np.ndarray, choices: np.ndarray, rewards: np.ndarray, states: np.ndarray) -> float:
    lp = _log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(params, choices, rewards, states)


def _hdi(samples: np.ndarray, credible_mass: float = 0.95) -> Tuple[float, float]:
    """最短区間 HDI を計算する。"""
    n = len(samples)
    sorted_samples = np.sort(samples)
    interval_width = int(np.floor(credible_mass * n))
    widths = sorted_samples[interval_width:] - sorted_samples[: n - interval_width]
    min_idx = np.argmin(widths)
    return float(sorted_samples[min_idx]), float(sorted_samples[min_idx + interval_width])


def _r_hat(chains: np.ndarray) -> float:
    """Gelman-Rubin R-hat を計算する。

    Parameters
    ----------
    chains : np.ndarray, shape (n_chains, n_samples)
    """
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    grand_mean = chains.mean()
    # between-chain variance
    B = n * np.var(chain_means, ddof=1)
    # within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


# ---------------------------------------------------------------------------
# 公開インタフェース
# ---------------------------------------------------------------------------

def fit_q_learning_bayesian(
    concat_list: List[Tuple[str, pd.DataFrame]],
    nwalkers: int = 32,
    nburn: int = 1000,
    nsamples: int = 2000,
    n_rhat_chains: int = 4,
    random_seed: int = 0,
    output_path: str = None,
) -> Tuple[pd.DataFrame, Dict]:
    """各参加者のQ学習モデルパラメータをMCMCで推定する。

    Parameters
    ----------
    concat_list : list of (subject, DataFrame)
        各タプルは (subject, df)。df は rt, chosen_item, reward_points カラムを含む。
    nwalkers : int
        emcee のウォーカー数（偶数かつ >= 4 であること）。
    nburn : int
        バーンインステップ数。
    nsamples : int
        バーンイン後のサンプリングステップ数（ウォーカーごと）。
    n_rhat_chains : int
        R-hat 計算のためにウォーカーを分割するグループ数。nwalkers の約数であること。
    random_seed : int
        再現性のための乱数シード。

    Returns
    -------
    results : pd.DataFrame
        subject, alpha_mean, alpha_median, alpha_hdi_low, alpha_hdi_high,
        beta_mean, beta_median, beta_hdi_low, beta_hdi_high,
        r_hat_alpha, r_hat_beta, log_likelihood, n_trials カラムを含む。
    traces : dict
        'subject'       : 参加者IDのリスト
        'alpha_samples' : shape (n_subjects, nwalkers * nsamples)
        'beta_samples'  : shape (n_subjects, nwalkers * nsamples)
    """
    rng = np.random.default_rng(random_seed)
    records = []
    all_alpha_samples = []
    all_beta_samples = []
    subject_ids = []

    ndim = 2

    for subject, df in concat_list:
        # 前処理
        df_clean = df.dropna(subset=["rt"]).copy()
        choices = df_clean["chosen_item"].values.astype(int)
        rewards = (df_clean["reward_points"].values >= 1).astype(float)
        states = df_clean["target_item"].values.astype(int)
        n_trials = len(choices)

        # ウォーカーの初期位置（事前分布の妥当な範囲内でランダム初期化）
        alpha0 = rng.uniform(0.1, 0.9, size=nwalkers)
        beta0 = rng.uniform(0.5, 5.0, size=nwalkers)
        p0 = np.stack([alpha0, beta0], axis=1)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, _log_posterior, args=(choices, rewards, states)
        )

        # バーンイン
        state = sampler.run_mcmc(p0, nburn, progress=False)
        sampler.reset()

        # 本サンプリング
        sampler.run_mcmc(state, nsamples, progress=False)

        # shape: (nwalkers, nsamples, ndim)
        flat_samples = sampler.get_chain(flat=False)  # (nsamples, nwalkers, ndim)
        flat_samples = flat_samples.transpose(1, 0, 2)  # (nwalkers, nsamples, ndim)

        alpha_all = flat_samples[:, :, 0].flatten()  # (nwalkers * nsamples,)
        beta_all = flat_samples[:, :, 1].flatten()

        # R-hat: ウォーカーを n_rhat_chains グループに分割
        walkers_per_chain = nwalkers // n_rhat_chains
        alpha_chains = flat_samples[:walkers_per_chain * n_rhat_chains, :, 0].reshape(n_rhat_chains, -1)
        beta_chains = flat_samples[:walkers_per_chain * n_rhat_chains, :, 1].reshape(n_rhat_chains, -1)

        r_hat_alpha = _r_hat(alpha_chains)
        r_hat_beta = _r_hat(beta_chains)

        # 事後統計量
        alpha_mean = float(np.mean(alpha_all))
        alpha_median = float(np.median(alpha_all))
        alpha_hdi_low, alpha_hdi_high = _hdi(alpha_all)

        beta_mean = float(np.mean(beta_all))
        beta_median = float(np.median(beta_all))
        beta_hdi_low, beta_hdi_high = _hdi(beta_all)

        log_lik = _log_likelihood(np.array([alpha_mean, beta_mean]), choices, rewards, states)

        records.append({
            "subject": subject,
            "alpha_mean": alpha_mean,
            "alpha_median": alpha_median,
            "alpha_hdi_low": alpha_hdi_low,
            "alpha_hdi_high": alpha_hdi_high,
            "beta_mean": beta_mean,
            "beta_median": beta_median,
            "beta_hdi_low": beta_hdi_low,
            "beta_hdi_high": beta_hdi_high,
            "r_hat_alpha": r_hat_alpha,
            "r_hat_beta": r_hat_beta,
            "log_likelihood": log_lik,
            "n_trials": n_trials,
        })

        subject_ids.append(subject)
        all_alpha_samples.append(alpha_all)
        all_beta_samples.append(beta_all)

    results = pd.DataFrame(records)
    if output_path:
        results.to_csv(output_path, index=False)
    traces = {
        "subject": subject_ids,
        "alpha_samples": np.stack(all_alpha_samples, axis=0),
        "beta_samples": np.stack(all_beta_samples, axis=0),
    }
    return results, traces
