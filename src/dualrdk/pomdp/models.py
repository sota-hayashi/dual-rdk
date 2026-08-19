"""モデル族 M0 / M1 / M1z / M2 / M2z / M3 / M3z（§3.3）。

決定変数 d_t の生成規則だけが異なり、方策（§3.1）は M0 / M2 / M3 系で共有する。
**M2z が主モデル**である。

    M0   d_t = c（定数）                        beta は 1 に固定
    M1   d_t = Q_t[s_t, W] - Q_t[s_t, B]        単一の alpha。状態は 2 値
    M1z  同上                                   alpha が注意状態 z_t 依存
    M2   d_t = V_t(th_W) - V_t(th_B)            単一の alpha
    M2z  同上                                   alpha が注意状態 z_t 依存
    M3   d_t = 2 q_t - 1                        単一の eps
    M3z  同上                                   eps が注意状態 z_t 依存

観測空間が 2 つある。M1 系は 2 値選択（確率質量）、M0 / M2 / M3 系は円環上の
応答角（確率密度）を予測する。**この 2 つの log_pi は単位が違うので直接比較
してはならない**。族をまたぐ比較には `policy.log_choice_lik`（選択のみの対数
尤度）を使う（`evaluate.py`）。

方策は「選択則」であって更新式を持たない。更新されるのは V_t（M2 系）または
b_t（M3 系）の方で、学習速度パラメータ alpha / eps はそこにある。

割引率は gamma = 0（myopic）。遷移核が恒等写像なので行動は将来の環境状態を
変えられず、環境由来の時間的クレジット割当が存在しない（FR-3.5）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import jax
import jax.numpy as jnp

from dualrdk.pomdp import agent_reward
from dualrdk.pomdp.belief import (
    PSI_GRID,
    belief_update,
    uniform_log_belief,
    white_is_target_mask,
    white_weight,
)
from dualrdk.pomdp.policy import choice_logit, log_policy_from_cos, p_white

# 無制約 -> 制約 の変換（FR-5.2）
_TRANSFORMS: Mapping[str, Callable] = {
    "log": jnp.exp,
    "logit": jax.nn.sigmoid,
    "identity": lambda x: x,
}


@dataclass(frozen=True)
class ParamSpec:
    """1 パラメータの定義。事前分布は **無制約スケール** で与える（§5.3）。

    仕様書の事前分布との対応:
        log beta      ~ N(log 5, 1.0)
        log kappa     ~ N(log 15, 0.7)
        log kappa_gen ~ N(log 2, 0.8)
        logit alpha   ~ N(-1.0, 1.2)
        eps, lam      ~ Beta(1, 20)  ->  logit スケールの N(-3.2, 1.2) で近似
                                        （非中心化階層は無制約空間で組む必要が
                                          あるため。中央値 0.039、95%区間
                                          [0.004, 0.30]）
    """

    name: str
    transform: str
    prior_loc: float
    prior_scale: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: tuple[ParamSpec, ...]
    init_state: Callable
    decision: Callable  # (state, trial, params) -> d_t
    step: Callable  # (state, trial, params) -> (new_state, outputs)
    fixed: dict = field(default_factory=dict)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def constrain(self, unc: Mapping[str, jnp.ndarray]) -> dict:
        """無制約空間のパラメータを制約付きスケールに戻し、固定値を足す。"""
        out = {p.name: _TRANSFORMS[p.transform](unc[p.name]) for p in self.params}
        out.update(self.fixed)
        return out


# --------------------------------------------------------------------------
# 共通の事前分布（無制約スケール）
# --------------------------------------------------------------------------
_BETA = ParamSpec("beta", "log", float(jnp.log(5.0)), 1.0)
_KAPPA = ParamSpec("kappa", "log", float(jnp.log(15.0)), 0.7)
_KAPPA_GEN = ParamSpec("kappa_gen", "log", float(jnp.log(2.0)), 0.8)
_LAM = ParamSpec("lam", "logit", -3.2, 1.2)
_C = ParamSpec("c", "identity", 0.0, 2.0)
# 色バイアス。既存 2 値実装の c_black ~ Normal(0, 2) に対応するが、階層モデルでは
# 個人差 sigma が別に乗るので群平均の事前は狭めに置く。
_C_BLACK = ParamSpec("c_black", "identity", 0.0, 1.0)


def _alpha_spec(name: str) -> ParamSpec:
    return ParamSpec(name, "logit", -1.0, 1.2)


def _eps_spec(name: str) -> ParamSpec:
    return ParamSpec(name, "logit", -3.2, 1.2)


def _by_zone(p, base: str, zone):
    """z_t に応じて学習速度を切り替える。単一パラメータのモデルではそのまま返す。"""
    if base in p:
        return p[base]
    return jnp.where(zone == 1, p[f"{base}_out"], p[f"{base}_in"])


def _nan():
    return jnp.asarray(jnp.nan)


# --- 事前計算（likelihood.precompute_trial_features）があれば使う ------------
# シミュレーション時は a_t を実行時にサンプルするため事前計算できないので、
# 無ければその場で計算するフォールバックを持つ。


def _cos_grid_a(tr):
    if "cos_grid_a" in tr:
        return tr["cos_grid_a"]
    return jnp.cos(PSI_GRID - tr["a"])


def _w_white(tr):
    if "w_white" in tr:
        return tr["w_white"]
    return white_weight(tr["iw"], tr["ib"])


def _mask(tr):
    if "mask" in tr:
        return tr["mask"]
    return white_is_target_mask(tr["iw"], tr["ib"])


def _c_black(p):
    return p.get("c_black", jnp.asarray(0.0))


def _log_policy(tr, d, p):
    if "cos_aw" in tr:
        cos_aw, cos_ab = tr["cos_aw"], tr["cos_ab"]
    else:
        cos_aw = jnp.cos(tr["a"] - tr["theta_w"])
        cos_ab = jnp.cos(tr["a"] - tr["theta_b"])
    return log_policy_from_cos(
        cos_aw, cos_ab, d, p["beta"], p["kappa"], p["lam"], _c_black(p)
    )


# --------------------------------------------------------------------------
# M0: 学習なしベースライン
# --------------------------------------------------------------------------
def _m0_init(p):
    return jnp.zeros(())


def _m0_decision(state, tr, p):
    return p["c"]


def _m0_step(state, tr, p):
    d = _m0_decision(state, tr, p)
    log_pi = _log_policy(tr, d, p)
    out = {
        "log_pi": log_pi,
        "d": d,
        "p_w": p_white(d, p["beta"], _c_black(p)),
        "q": _nan(),
        "log_evidence": _nan(),
    }
    return state, out


# --------------------------------------------------------------------------
# M1 / M1z: 2 値状態・2 値行動の表形式 Q 学習（既存 MAP 実装の移植）
# --------------------------------------------------------------------------
# 状態 s_t は「どちらの雲が高報酬側か」の 2 値で、データから直接与える。
#
# これは **仮定**である。参加者は試行 1 の時点で刺激が 2 群に分かれることを
# 知らないし、どちらの群が報酬をもたらすかも知らない。M1 系はカテゴリ化の
# コストをゼロと置き、「正しい 2 分割を最初から誤差なく持っている」参加者を
# モデル化している。M2 / M3 系はこの仮定を置かず、V や b_t から自力で導く。
# M1 と M2 の比較は、この仮定の当否を測るものである。
#
# 行動も 2 値なので、書き込んだセルは次に同じ状態が来たとき必ず読まれる。
# したがって M2 系のような汎化カーネルを必要としない。応答角の情報は捨てる
# ので kappa / lam も持たない。
def _m1_init(p):
    """Q_0 = 0.5（2 状態 x 2 行動）。既存実装と同じ初期値。"""
    return jnp.full((2, 2), 0.5)


def _m1_decision(state, tr, p):
    s = tr["s_state"]
    return state[s, 0] - state[s, 1]  # 白 - 黒。符号を M2 系と揃える


def _m1_step(state, tr, p):
    Q = state
    s = tr["s_state"]
    d = _m1_decision(Q, tr, p)

    # log_pi は 2 値選択の確率質量。M2 / M3 系の密度とは単位が違う（冒頭参照）。
    z = choice_logit(d, p["beta"], _c_black(p))
    log_pi = jnp.where(tr["chose_white"], jax.nn.log_sigmoid(z), jax.nn.log_sigmoid(-z))

    alpha = _by_zone(p, "alpha", tr["zone"])
    a_bin = jnp.where(tr["chose_white"], 0, 1)
    delta = tr["r_norm"] - Q[s, a_bin]
    Q_new = jnp.where(tr["valid"], Q.at[s, a_bin].add(alpha * delta), Q)

    out = {
        "log_pi": log_pi,
        "d": d,
        "p_w": jax.nn.sigmoid(z),
        "q": _nan(),
        "log_evidence": _nan(),
    }
    return Q_new, out


# --------------------------------------------------------------------------
# M2 / M2z: 方向ベース Q 学習
# --------------------------------------------------------------------------
def _m2_init(p):
    """V_0 = 0（360 次元のグリッドベクトル）。"""
    return jnp.zeros(PSI_GRID.shape[0])


def _m2_decision(state, tr, p):
    return state[tr["iw"]] - state[tr["ib"]]


def _m2_step(state, tr, p):
    V = state
    d = _m2_decision(V, tr, p)
    log_pi = _log_policy(tr, d, p)

    # 学習則。予測誤差は「実際に報告した方向 a_t」で評価する（th_W / th_B ではない）。
    # 更新は von Mises 汎化カーネルで全方向に広がる。これは von Mises 基底による
    # 線形関数近似の semi-gradient TD 更新と等価。
    alpha = _by_zone(p, "alpha", tr["zone"])
    K = jnp.exp(p["kappa_gen"] * (_cos_grid_a(tr) - 1.0))
    delta = tr["r_norm"] - V[tr["ia"]]
    V_new = jnp.where(tr["valid"], V + alpha * K * delta, V)

    out = {
        "log_pi": log_pi,
        "d": d,
        "p_w": p_white(d, p["beta"], _c_black(p)),
        "q": _nan(),
        "log_evidence": _nan(),
    }
    return V_new, out


# --------------------------------------------------------------------------
# M3 / M3z: 信念ベース
# --------------------------------------------------------------------------
def _m3_q(log_b, tr):
    # q はタイを 0.5 で分ける重みで計算し、尤度の theta_star は bool マスクで選ぶ。
    return jnp.sum(jnp.exp(log_b) * _w_white(tr)), _mask(tr)


def _m3_decision(state, tr, p):
    q, _ = _m3_q(state, tr)
    return 2.0 * q - 1.0


def _make_m3_step(log_lik_fn):
    def _m3_step(state, tr, p):
        log_b = state
        q, mask = _m3_q(log_b, tr)
        d = 2.0 * q - 1.0
        log_pi = _log_policy(tr, d, p)

        eps = _by_zone(p, "eps", tr["zone"])
        log_L = log_lik_fn(tr, mask, eps, p.get("w", jnp.asarray(1.0)))
        log_b_upd, log_evidence = belief_update(log_b, log_L)
        log_b_new = jnp.where(tr["valid"], log_b_upd, log_b)

        out = {
            "log_pi": log_pi,
            "d": d,
            "p_w": p_white(d, p["beta"], _c_black(p)),
            "q": q,
            "log_evidence": log_evidence,
        }
        return log_b_new, out

    return _m3_step


def _m3_init(p):
    return uniform_log_belief()


# --------------------------------------------------------------------------
# レジストリ
# --------------------------------------------------------------------------
def build_model(
    name: str,
    agent_reward_variant: str = "ordinal",
    fixed: Mapping[str, float] | None = None,
) -> ModelSpec:
    """モデル仕様を構築する。

    Parameters
    ----------
    name : "M0" | "M1" | "M1z" | "M2" | "M2z" | "M3" | "M3z"
    agent_reward_variant : "ordinal"（既定, FR-4.2）| "smooth"（FR-4.3）
        M3 系にのみ影響する。M2 系は R_hat を必要としない。
    fixed : パラメータ名 -> 値。指定したパラメータは推定対象から外し、定数に
        する（2 段階推定で kappa / lam を固定する用途）。M1 系のように元から
        そのパラメータを持たないモデルでは無視される。
    """
    spec = _build_model_free(name, agent_reward_variant)
    if not fixed:
        return spec
    keep = tuple(p for p in spec.params if p.name not in fixed)
    frozen = {k: jnp.asarray(v) for k, v in fixed.items() if k in spec.param_names}
    unknown = set(fixed) - set(spec.param_names)
    if unknown:
        raise ValueError(f"{name} は {sorted(unknown)} を持たないので固定できない")
    return ModelSpec(
        name=spec.name,
        params=keep,
        init_state=spec.init_state,
        decision=spec.decision,
        step=spec.step,
        fixed={**spec.fixed, **frozen},
    )


def _build_model_free(name: str, agent_reward_variant: str) -> ModelSpec:
    if name == "M0":
        return ModelSpec(
            name="M0",
            params=(_C, _KAPPA, _LAM),
            init_state=_m0_init,
            decision=_m0_decision,
            step=_m0_step,
            fixed={"beta": jnp.asarray(1.0)},  # beta と c が縮退するため固定（§3.3）
        )

    if name in ("M1", "M1z"):
        alphas = (_alpha_spec("alpha"),) if name == "M1" else (
            _alpha_spec("alpha_in"),
            _alpha_spec("alpha_out"),
        )
        return ModelSpec(
            name=name,
            params=(_BETA, *alphas, _C_BLACK),
            init_state=_m1_init,
            decision=_m1_decision,
            step=_m1_step,
        )

    if name in ("M2", "M2z"):
        alphas = (_alpha_spec("alpha"),) if name == "M2" else (
            _alpha_spec("alpha_in"),
            _alpha_spec("alpha_out"),
        )
        return ModelSpec(
            name=name,
            params=(_BETA, *alphas, _KAPPA_GEN, _KAPPA, _LAM, _C_BLACK),
            init_state=_m2_init,
            decision=_m2_decision,
            step=_m2_step,
        )

    if name in ("M3", "M3z"):
        epss = (_eps_spec("eps"),) if name == "M3" else (
            _eps_spec("eps_in"),
            _eps_spec("eps_out"),
        )
        extra = ()
        if agent_reward_variant == "smooth":
            # R_hat の主観的幅 w（FR-4.3）。中央値 45 度 ~ 0.785 rad
            extra = (ParamSpec("w", "log", float(jnp.log(0.785)), 0.5),)
        return ModelSpec(
            name=name,
            params=(_BETA, *epss, _KAPPA, _LAM, _C_BLACK, *extra),
            init_state=_m3_init,
            decision=_m3_decision,
            step=_make_m3_step(agent_reward.get_variant(agent_reward_variant)),
        )

    raise ValueError(f"未知のモデル: {name!r}（{MODEL_NAMES} のいずれか）")


MODEL_NAMES = ("M0", "M1", "M1z", "M2", "M2z", "M3", "M3z")

# 応答角を予測するモデル（log_pi が円環上の密度）と、2 値選択のみを予測する
# モデル（log_pi が確率質量）。ELPD や尤度を族をまたいで比べてよいかの判定に使う。
ANGLE_MODELS = ("M0", "M2", "M2z", "M3", "M3z")
BINARY_MODELS = ("M1", "M1z")
