"""2 峰混合方策（§3.1）。全モデル共通。

報酬の山は幅 +-45 度で、2 方向は Delta >= 90 度離れているため互いに素である。
したがって最適行動は必ず theta_W か theta_B のいずれかで、中間を報告する
利得はない。反応分布は 2 つの von Mises の混合 + 一様 lapse とする。

    p_W = sigmoid(beta * d - c_black)
    pi(a) = (1-lam) [ p_W vM(a; th_W, kap) + (1-p_W) vM(a; th_B, kap) ] + lam/(2pi)

p_W は 2 アーム softmax を書き換えたものと厳密に同じである
（分子分母を exp(beta V_W) で割ると sigmoid(beta (V_W - V_B)) になる）。

c_black は価値と無関係な色バイアス（c_black > 0 で黒寄り）。既存の 2 値 Q 学習
実装（models/q_learning_map.py）が持っていた項に対応する。これが無いと、価値差
d では説明できない系統的な色の偏りが d に押し付けられる。

観測されるのは連続値 a_t であって「どちらを選んだか」ではないので、選択は
潜在変数として周辺化される。これが混合分布になる理由。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from dualrdk.pomdp.circular import TWO_PI, log_vonmises, log_vonmises_from_cos


def choice_logit(d, beta, c_black=0.0):
    """白を選ぶ側のロジット。方策と離散選択尤度の唯一の接点。"""
    return beta * d - c_black


def log_policy_from_cos(cos_aw, cos_ab, d, beta, kappa, lam, c_black=0.0):
    """log pi_t(a_t)。cos(a - theta) を事前計算済みの高速版。

    cos(a - theta_w) と cos(a - theta_b) はどちらもデータのみに依存するので、
    勾配評価のたびに計算し直す必要はない。数値安定のため全て log 空間で組む。
    """
    z = choice_logit(d, beta, c_black)
    lw = jax.nn.log_sigmoid(z) + log_vonmises_from_cos(cos_aw, kappa)
    lb = jax.nn.log_sigmoid(-z) + log_vonmises_from_cos(cos_ab, kappa)
    log_mix = jnp.logaddexp(lw, lb)
    log_lapse = jnp.log(lam) - jnp.log(TWO_PI)
    return jnp.logaddexp(jnp.log1p(-lam) + log_mix, log_lapse)


def log_policy(a, theta_w, theta_b, d, beta, kappa, lam, c_black=0.0):
    """log pi_t(a_t)。角度から直接計算する版（テスト・可読性用）。"""
    return log_policy_from_cos(
        jnp.cos(a - theta_w), jnp.cos(a - theta_b), d, beta, kappa, lam, c_black
    )


def policy_pdf(a, theta_w, theta_b, d, beta, kappa, lam, c_black=0.0):
    return jnp.exp(log_policy(a, theta_w, theta_b, d, beta, kappa, lam, c_black))


def p_white(d, beta, c_black=0.0):
    return jax.nn.sigmoid(choice_logit(d, beta, c_black))


def log_choice_lik(chose_white, d, beta, c_black=0.0, lam=0.0):
    """離散選択のみの対数尤度（FR-6.5）。

    M1 系（2 値行動）と M2 / M3 系（連続行動）を同じ土俵に乗せるための共通通貨。
    角度成分を落として「どちらの雲を選んだか」だけを見る。密度ではなく確率質量
    なので、モデル族をまたいで足したり比べたりできる。

    lam > 0 のときはラプスを 2 値側にも反映する（ラプス時は白黒 50:50）。
    """
    z = choice_logit(d, beta, c_black)
    log_pw = jnp.where(chose_white, jax.nn.log_sigmoid(z), jax.nn.log_sigmoid(-z))
    if lam is None:
        return log_pw
    return jnp.logaddexp(jnp.log1p(-lam) + log_pw, jnp.log(lam) - jnp.log(2.0))


def sample_action(key, theta_w, theta_b, d, beta, kappa, lam, c_black=0.0):
    """方策から行動をサンプルする（recovery 用のシミュレーション）。

    jax.random に vonmises が無いため numpyro の VonMises（Best-Fisher の
    棄却法）を使う。ローカル import にして、コア数値モジュールが numpyro に
    依存しないようにしている。
    """
    from numpyro.distributions import VonMises

    k_lapse, k_choice, k_noise, k_unif = jax.random.split(key, 4)
    is_lapse = jax.random.uniform(k_lapse) < lam
    pick_white = jax.random.uniform(k_choice) < p_white(d, beta, c_black)
    mu = jnp.where(pick_white, theta_w, theta_b)
    a_model = VonMises(loc=mu, concentration=kappa).sample(k_noise)
    a_lapse = jax.random.uniform(k_unif, minval=0.0, maxval=TWO_PI)
    return jnp.where(is_lapse, a_lapse, a_model) % TWO_PI
