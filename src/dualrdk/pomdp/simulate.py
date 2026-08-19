"""モデルからのデータ生成（§5.6 のリカバリ用）。

刺激（theta_W, theta_B, theta_star）と注意状態 z_t は **実データからそのまま
借りる**。課題の生成モデルを再実装せずに済み、かつ設計と一致したシミュレーション
になる（design-matched simulation）。モデルが生成するのは反応 a_t だけで、
報酬は環境の報酬関数 R（`task_reward`）で決まる。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from dualrdk.pomdp.belief import GRID_N
from dualrdk.pomdp.circular import DEG, TWO_PI
from dualrdk.pomdp.likelihood import derive_choice_fields
from dualrdk.pomdp.models import BINARY_MODELS
from dualrdk.pomdp.policy import p_white, sample_action
from dualrdk.pomdp.task_reward import REWARD_MAX, env_reward_rad_jax


def simulate_subject(model, params, stim, key):
    """1 参加者分をシミュレートする。

    stim : 各値が (T,) の辞書（theta_w, theta_b, theta_star, it, zone, valid）
    Returns : trials 辞書（likelihood.scan_subject にそのまま渡せる形）

    M1 系（2 値行動）は応答角を生成しない。選択した雲の方向をそのまま a_t と
    して置く。M1 の尤度は角度を見ないので影響はなく、環境の報酬関数を共通に
    使えるという利点だけが残る。
    """
    state0 = model.init_state(params)
    binary = model.name in BINARY_MODELS

    def body(carry, x):
        state, k = carry
        k, k_act = jax.random.split(k)
        d = model.decision(state, x, params)
        if binary:
            p_w = p_white(d, params["beta"], params.get("c_black", jnp.asarray(0.0)))
            pick_white = jax.random.uniform(k_act) < p_w
            a = jnp.where(pick_white, x["theta_w"], x["theta_b"])
        else:
            a = sample_action(
                k_act, x["theta_w"], x["theta_b"], d,
                params["beta"], params["kappa"], params["lam"],
                params.get("c_black", jnp.asarray(0.0)),
            )
        r = env_reward_rad_jax(a, x["theta_star"])
        tr = {
            "theta_w": x["theta_w"],
            "theta_b": x["theta_b"],
            "a": a,
            "iw": x["iw"],
            "ib": x["ib"],
            "ia": (jnp.round(a / DEG).astype(jnp.int32)) % GRID_N,
            "it": x["it"],
            "r_norm": r / float(REWARD_MAX),
            "reward_positive": r > 0,
            "zone": x["zone"],
            "valid": x["valid"],
        }
        tr.update(derive_choice_fields(tr))
        new_state, _ = model.step(state, tr, params)
        return (new_state, k), tr

    _, trials = lax.scan(body, (state0, key), stim)
    return trials


def simulate_batch(model, params_batched, stim_batched, key):
    """参加者方向に vmap したシミュレーション。"""
    n_subj = jax.tree_util.tree_leaves(stim_batched)[0].shape[0]
    keys = jax.random.split(key, n_subj)
    return jax.vmap(lambda p, s, k: simulate_subject(model, p, s, k))(
        params_batched, stim_batched, keys
    )


def stim_arrays(theta_white_deg, theta_black_deg, target_deg, zone, valid):
    """度数の配列から stim 辞書を組む（(n_subj, T) でも (T,) でも可）。"""
    import numpy as np

    tw = jnp.asarray(np.asarray(theta_white_deg, dtype=float) * float(DEG))
    tb = jnp.asarray(np.asarray(theta_black_deg, dtype=float) * float(DEG))
    ts = jnp.asarray(np.asarray(target_deg, dtype=float) * float(DEG))
    return {
        "theta_w": tw,
        "theta_b": tb,
        "theta_star": ts,
        "iw": jnp.asarray(np.rint(np.asarray(theta_white_deg)).astype(int) % GRID_N),
        "ib": jnp.asarray(np.rint(np.asarray(theta_black_deg)).astype(int) % GRID_N),
        "it": jnp.asarray(np.rint(np.asarray(target_deg)).astype(int) % GRID_N),
        "zone": jnp.asarray(np.asarray(zone, dtype=int)),
        "valid": jnp.asarray(np.asarray(valid, dtype=bool)),
    }
