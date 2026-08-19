"""エージェントの内部報酬モデル R_hat（§4）。

環境の報酬関数 R（`task_reward.py`）とは意図的に別モジュールに置き、関数を
共用しない（FR-4.1）。R はデータを生成した客観的事実、R_hat は「参加者が
報酬の決まり方についてどう思っているか」のモデルであり、別物である。

R_hat が入るのは信念更新の尤度 L_t(psi) だけで、M3 / M3z のみが使う。
M2 / M2z は観測した r をそのまま予測誤差に使うので R_hat を必要としない。

既定は `ordinal`（FR-4.2）：「点が入ったか否か」の 1 ビットのみを使うため、
R の傾き・量子化・最大値にほぼ非依存で、参加者の知識状態についての仮定が
最も弱い。`smooth` は感度分析用（FR-4.3）。

各関数は trial 辞書を受け取り、事前計算（likelihood.precompute_trial_features）
があればそれを使う。無ければその場で計算する（シミュレーション時）。
"""
from __future__ import annotations

import jax.numpy as jnp

from dualrdk.pomdp.circular import DEG, circ_dist

# 報酬 > 0 の境界（§0.3）。R_hat 側の定数であり task_reward の値とは独立に持つ。
REWARD_RADIUS = 45.0 * DEG

# smooth バリアントの報酬観測ノイズ（固定）。w のみを自由パラメータにする。
SMOOTH_SIGMA = 0.15


def _dist_star(tr, mask):
    """各格子点 psi の下での高報酬側の方向までの角度距離 [rad]。"""
    if "dist_star" in tr:
        return tr["dist_star"]
    theta_star = jnp.where(mask, tr["theta_w"], tr["theta_b"])
    return circ_dist(tr["a"], theta_star)


def ordinal_log_lik(tr, mask, eps, w=None):
    """順序尤度（既定, FR-4.2）。

    consistent(psi) = ( [d(a, theta_star(psi)) <= 45deg] == [r > 0] )

    汚染混合 L = (1 - eps) * 1[consistent] + eps を代入すると

        consistent  -> L = (1-eps) + eps = 1     -> log L = 0
        矛盾        -> L = 0 + eps      = eps    -> log L = log eps

    となり実装は 1 行になる。eps がないと log L = -inf が伝播し、1 試行の
    逸脱で事後が不可逆に潰れる（FR-2.2）。

    consistent 行列は eps 以外パラメータに依存しないので事前計算できる。
    """
    if "consistent" in tr:
        consistent = tr["consistent"]
    else:
        hit = _dist_star(tr, mask) <= REWARD_RADIUS
        consistent = hit == tr["reward_positive"]
    return jnp.where(consistent, 0.0, jnp.log(eps))


def smooth_log_lik(tr, mask, eps, w, sigma=SMOOTH_SIGMA):
    """平滑バリアント（FR-4.3、感度分析用）。

    R_hat(x) = max(0, 1 - |x| / w) を主観的な報酬形として仮定し、観測報酬との
    二乗誤差をガウス尤度にする。w は自由パラメータ、sigma は固定。
    """
    pred = jnp.maximum(0.0, 1.0 - _dist_star(tr, mask) / w)
    ll = -0.5 * ((tr["r_norm"] - pred) / sigma) ** 2
    ll = ll - jnp.max(ll)  # 最大を 0 に正規化してから汚染混合
    return jnp.log((1.0 - eps) * jnp.exp(ll) + eps)


VARIANTS = {"ordinal": ordinal_log_lik, "smooth": smooth_log_lik}


def get_variant(name: str):
    if name not in VARIANTS:
        raise ValueError(f"未知の R_hat バリアント: {name!r}（{sorted(VARIANTS)} のいずれか）")
    return VARIANTS[name]
