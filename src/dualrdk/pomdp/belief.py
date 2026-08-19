"""潜在報酬軸 psi についての信念（§1.4, §2）。

信念は 1 度刻み 360 点のグリッド上の **対数確率質量** として保持する
（密度ではない）。不変条件は logsumexp(log_b) == 0（FR-2.1）。

遷移核は恒等写像 T(psi'|psi, a) = delta(psi' - psi) なので、ベイズフィルタの
予測ステップは恒等作用素であり実装しない（FR-1.1）。離散化すれば遷移行列は
単位行列である。したがって更新は尤度の掛け算（対数では加算）だけになる。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from dualrdk.pomdp.circular import DEG, circ_dist

GRID_N = 360
# 格子点 psi_j = j 度（ラジアン）。刺激角が整数度なので 1 度刻みが自然。
PSI_GRID = jnp.arange(GRID_N) * DEG
PSI_GRID_NP = np.arange(GRID_N) * (np.pi / 180.0)


def uniform_log_belief(grid_n: int = GRID_N):
    """初期信念：円周上一様（§2.1）。"""
    return jnp.full(grid_n, -jnp.log(float(grid_n)))


def _index_distances(idx):
    """格子点と方向インデックスの円距離を **整数度** で返す。

    刺激角も報告角も格子も整数度なので、距離を整数演算で計算できる。
    浮動小数点の atan2 で比較すると二等分線上のタイが数値誤差で解けてしまい、
    (a) 一様信念で q が厳密に 0.5 にならない、(b) 全角度を回転したときに
    マスクが 1 点ずれて対数尤度が不変にならない、という問題が起きる。
    整数演算なら回転はインデックスの巡回シフトになり、どちらも厳密に成立する。
    """
    j = jnp.arange(GRID_N)
    diff = (j - idx) % GRID_N
    return jnp.minimum(diff, GRID_N - diff)


def white_is_target_mask(iw, ib):
    """各格子点 psi について「白が高報酬側」か否か（bool）。

    psi が theta_w の方に近ければ白が高報酬側（§0.3 の argmin ルール）。
    真になる領域は theta_w と theta_b の垂直二等分線が切る 180 度の弧である。
    尤度で theta_star(psi) を選ぶのに使う。引数は格子インデックス（整数度）。
    """
    return _index_distances(iw) < _index_distances(ib)


def white_weight(iw, ib):
    """q の計算用の重み。二等分線ちょうどの格子点には 0.5 を割り当てる。

    刺激角が整数度なので二等分線は格子点に乗りうる。そこでは「どちらが近いか」が
    定義できず、bool マスクだと 180 点にならない（179 または 181 になる）。
    連続極限ではタイの集合は測度ゼロなので、離散化では質量を半分ずつ分けるのが
    正しい。対称性 j -> (iw + ib - j) により重みの総和は厳密に 180 になり、
    一様信念では厳密に q = 0.5 が得られる。
    """
    d_w = _index_distances(iw)
    d_b = _index_distances(ib)
    return jnp.where(d_w < d_b, 1.0, jnp.where(d_w > d_b, 0.0, 0.5))


def belief_to_q(log_b, iw, ib):
    """信念をスカラー q_t（白が高報酬側である確率）に縮約する（FR-1.4）。

    信念は 360 次元だが、その試行の意思決定に対してはこのスカラー 1 個に
    圧縮される。
    """
    return jnp.sum(jnp.exp(log_b) * white_weight(iw, ib))


def belief_update(log_b, log_L):
    """ベイズ更新（§2.3）。

    予測ステップは恒等（FR-1.1）なので更新ステップだけを行う。

    Returns
    -------
    log_b_new : 正規化済み対数信念
    log_evidence : logsumexp(log_b + log_L)。その試行の予測尤度
                   p(r_t | 履歴) であり、診断用に記録する（FR-2.4）。
    """
    log_unnorm = log_b + log_L
    log_evidence = logsumexp(log_unnorm)
    return log_unnorm - log_evidence, log_evidence


def belief_update_checked(log_b, log_L):
    """numpy 版のベイズ更新。事後の全域アンダーフローで例外を送出する（FR-2.3）。

    汚染混合 eps > 0 の下では log_L >= log(eps) > -inf なので到達しないが、
    eps を 0 にした場合や実装ミスを検出するための安全網。黙って一様分布に
    リセットしてはならない。
    """
    log_b = np.asarray(log_b, dtype=float)
    log_L = np.asarray(log_L, dtype=float)
    log_unnorm = log_b + log_L
    m = np.max(log_unnorm)
    if not np.isfinite(m):
        raise ValueError(
            "信念が全域でゼロになった（logsumexp = -inf）。"
            "汚染混合 eps を確認すること（FR-2.2, FR-2.3）。"
        )
    log_evidence = m + np.log(np.sum(np.exp(log_unnorm - m)))
    return log_unnorm - log_evidence, log_evidence


def log_belief_by_counting(consistent, log_eps, grid_n: int = GRID_N):
    """矛盾回数の数え上げによる閉形式（§2.4、単体テスト用）。

    予測ステップが恒等なので b_t は尤度の単純な積になり、順序尤度では

        log b_t(psi) = log b_0(psi) + n_t(psi) log eps

    となる（n_t は試行 1..t のうち psi と矛盾した回数）。逐次フィルタ実装が
    これと一致することを FR-2.5 のテストで確認する。

    Parameters
    ----------
    consistent : (T, grid_n) の bool 配列
    log_eps : スカラーまたは (T,) 配列（M3z では試行ごとに eps が変わる）

    Returns
    -------
    (T, grid_n) の正規化済み対数信念（各 t の更新 *後*）
    """
    consistent = np.asarray(consistent, dtype=bool)
    log_eps = np.broadcast_to(np.asarray(log_eps, dtype=float), (consistent.shape[0],))
    contrib = np.where(consistent, 0.0, log_eps[:, None])
    log_b = -np.log(float(grid_n)) + np.cumsum(contrib, axis=0)
    m = np.max(log_b, axis=1, keepdims=True)
    log_b = log_b - (m + np.log(np.sum(np.exp(log_b - m), axis=1, keepdims=True)))
    return log_b
