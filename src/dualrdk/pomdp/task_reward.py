"""環境の報酬関数 R（§0.3）。

これは実験プログラムに実装された **客観的事実** であり、参加者の知識状態とは
無関係に存在する。エージェントの内部報酬モデル R_hat（`agent_reward.py`）とは
別モジュールに置き、関数を共用しない（FR-4.1）。

    r = max(0, floor(10 - |e| / 5)),   e = wrap(a - theta_star)  [度]

性質：整数 0..10 の量子化、|e| > 45 度で厳密に 0、ノイズなし。
シミュレーション（recovery）とデータ検証にのみ使用する。
"""
from __future__ import annotations

import numpy as np

REWARD_MAX = 10
REWARD_RADIUS_DEG = 45.0


def wrap_deg(x):
    """角度を (-180, 180] に折り返す。"""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def env_reward(a_deg, theta_star_deg):
    """環境の報酬関数。整数点を返す。"""
    e = np.abs(wrap_deg(np.asarray(a_deg, dtype=float) - np.asarray(theta_star_deg, dtype=float)))
    return np.maximum(0, np.floor(REWARD_MAX - e / 5.0)).astype(int)


def env_reward_rad_jax(a_rad, theta_star_rad):
    """jax 版（シミュレーション用）。入力はラジアン、出力は整数点（float）。

    微分不可能（floor）だが、シミュレーションでは勾配を取らないので問題ない。
    """
    import jax.numpy as jnp

    e = jnp.abs(jnp.arctan2(jnp.sin(a_rad - theta_star_rad), jnp.cos(a_rad - theta_star_rad)))
    e_deg = e * (180.0 / jnp.pi)
    return jnp.maximum(0.0, jnp.floor(REWARD_MAX - e_deg / 5.0))
