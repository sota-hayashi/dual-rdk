"""逐次スキャンによる対数尤度（§5.1）。

試行はパラメータ条件付きでも独立ではなく、V_t / b_t を通じて逐次依存している。
したがってベクトル化して独立試行として扱ってはならず、lax.scan で 1 試行ずつ
状態を運ぶ（FR-5.1）。

無効試行（rt が NaN 等）は尤度に寄与させず、状態更新も行わない。モデル化
しない試行がモデルの内部状態を動かすのは筋が通らないため。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from dualrdk.pomdp.belief import GRID_N, PSI_GRID
from dualrdk.pomdp.circular import DEG

TRIAL_FIELDS = (
    "theta_w",
    "theta_b",
    "a",
    "iw",
    "ib",
    "ia",
    "r_norm",
    "reward_positive",
    "zone",
    "valid",
)

# precompute_trial_features が付与する、パラメータに依存しない量
PRECOMPUTED_FIELDS = (
    "cos_grid_a",  # (..., T, 360) cos(psi_j - a_t)        M2 の汎化カーネル用
    "w_white",     # (..., T, 360) 白側の重み（タイは 0.5） q 用
    "mask",        # (..., T, 360) 白が高報酬側か           theta_star の選択用
    "dist_star",   # (..., T, 360) d(a_t, theta_star(psi))  R_hat 用 [rad]
    "consistent",  # (..., T, 360) 順序尤度の一致フラグ      M3 用
    "cos_aw",      # (..., T)      cos(a_t - theta_w)       方策
    "cos_ab",      # (..., T)      cos(a_t - theta_b)       方策
    "chose_white", # (..., T)      応答角が白側の雲に近いか  M1 と共通評価
    "s_state",     # (..., T)      0=白が高報酬 / 1=黒       M1 の状態（要 "it"）
)

REWARD_RADIUS_DEG = 45.0


def _index_distance(x, y):
    """整数度の巡回距離。x, y は格子インデックス。"""
    diff = (x - y) % GRID_N
    return jnp.minimum(diff, GRID_N - diff)


def precompute_trial_features(trials: dict) -> dict:
    """パラメータに依存しない量を先に計算しておく。

    尤度の内側ループで毎回計算していた三角関数と巡回距離は、実は刺激角・
    報告角というデータだけで決まる。NUTS は勾配評価のたびに尤度を呼ぶので、
    これらを事前計算しておくと内側ループから三角関数が完全に消える。

    - M2 系: カーネル K = exp(kappa_gen (cos(psi - a) - 1)) の cos 部分
    - M3 系: 順序尤度の consistent 行列（eps 以外パラメータに依存しない）
    - 方策 : cos(a - theta_w), cos(a - theta_b)

    先頭次元は任意（(T, ...) でも (n_subj, T, ...) でも可）。
    シミュレーションでは a_t を実行時にサンプルするため事前計算できない。
    各モデルは事前計算があればそれを使い、無ければその場で計算する。
    """
    iw, ib, ia = trials["iw"], trials["ib"], trials["ia"]
    j = jnp.arange(GRID_N)

    d_w = _index_distance(j, iw[..., None])  # (..., T, 360)
    d_b = _index_distance(j, ib[..., None])
    mask = d_w < d_b
    w_white = jnp.where(d_w < d_b, 1.0, jnp.where(d_w > d_b, 0.0, 0.5))

    # theta_star(psi) までの距離。a も theta も整数度なので整数演算で厳密に出る
    d_aw_deg = _index_distance(ia, iw)  # (..., T)
    d_ab_deg = _index_distance(ia, ib)
    dist_star_deg = jnp.where(mask, d_aw_deg[..., None], d_ab_deg[..., None])

    hit = dist_star_deg <= REWARD_RADIUS_DEG
    consistent = hit == trials["reward_positive"][..., None]

    out = dict(trials)
    out["cos_grid_a"] = jnp.cos(PSI_GRID - trials["a"][..., None])
    out["w_white"] = w_white
    out["mask"] = mask
    out["dist_star"] = dist_star_deg * DEG
    out["consistent"] = consistent
    out["cos_aw"] = jnp.cos(trials["a"] - trials["theta_w"])
    out["cos_ab"] = jnp.cos(trials["a"] - trials["theta_b"])
    out.update(derive_choice_fields(trials))
    return out


def derive_choice_fields(trials: dict) -> dict:
    """離散選択の場（M1 系と共通評価が使う）。

    chose_white : 応答角がどちらの雲に近いか。等距離のときは白としない
                  （実データでは 2584 試行中 1 試行だけ）。
    s_state     : 0 = 白が高報酬 / 1 = 黒が高報酬。既存 MAP 実装の
                  `target_item` と同じ規約。オラクル情報なので M1 系のみが使う。
                  trials に "it"（ターゲット方向の格子インデックス）が
                  無ければ付与しない。
    """
    out = {
        "chose_white": _index_distance(trials["ia"], trials["iw"])
        < _index_distance(trials["ia"], trials["ib"])
    }
    if "it" in trials:
        out["s_state"] = jnp.where(trials["it"] == trials["iw"], 0, 1)
    return out


def scan_subject(model, params, trials):
    """1 参加者分をスキャンする。

    Parameters
    ----------
    model : ModelSpec
    params : 制約付きスケールのパラメータ辞書（スカラー）
    trials : 各値が (T,) の辞書。TRIAL_FIELDS を持つ

    Returns
    -------
    total_log_lik : スカラー
    outs : 各値が (T,) の辞書（log_pi, d, p_w, q, log_evidence）
    """
    state0 = model.init_state(params)

    def body(state, tr):
        return model.step(state, tr, params)

    _, outs = lax.scan(body, state0, trials)
    total = jnp.sum(jnp.where(trials["valid"], outs["log_pi"], 0.0))
    return total, outs


def subject_log_lik(model, params, trials):
    return scan_subject(model, params, trials)[0]


def batched_log_lik(model, params_batched, trials_batched):
    """参加者方向に vmap した対数尤度。

    params_batched : 各値が (n_subj,)
    trials_batched : 各値が (n_subj, T)

    Returns
    -------
    (n_subj,) の対数尤度。PSIS-LOO の観測単位は参加者である（FR-5.7）。
    """
    return jax.vmap(lambda p, t: subject_log_lik(model, p, t))(params_batched, trials_batched)


def batched_latents(model, params_batched, trials_batched):
    """潜在変数の軌跡（FR-6.1）。各値が (n_subj, T)。"""
    return jax.vmap(lambda p, t: scan_subject(model, p, t)[1])(params_batched, trials_batched)
