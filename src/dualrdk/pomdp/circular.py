"""円周量のユーティリティ（FR-3.1）。

角度は内部で全てラジアン。度数への変換は I/O 境界でのみ行う。
生の [0, 2pi) スカラーを線形演算に投入してはならない（0 と 2pi-eps が最遠点になる）。
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import i0e

DEG = jnp.pi / 180.0
TWO_PI = 2.0 * jnp.pi


def deg2rad(x):
    return jnp.asarray(x) * DEG


def rad2deg(x):
    return jnp.asarray(x) / DEG


def wrap(x):
    """角度を (-pi, pi] に折り返す。"""
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def circ_dist(a, b):
    """円距離 d(a, b) = |atan2(sin(a-b), cos(a-b))| in [0, pi]。"""
    return jnp.abs(wrap(a - b))


def circ_mean(x, axis=None):
    """円周平均。算術平均を使ってはならない。"""
    return jnp.arctan2(jnp.mean(jnp.sin(x), axis=axis), jnp.mean(jnp.cos(x), axis=axis))


def log_vonmises_from_cos(cos_x, kappa):
    """cos(x - mu) を事前計算済みの場合の von Mises 対数密度。

    cos(x - mu) は x も mu もデータなのでパラメータに依存しない。勾配評価の
    たびに三角関数を計算し直さないためのエントリポイント。
    """
    return kappa * (cos_x - 1.0) - jnp.log(TWO_PI) - jnp.log(i0e(kappa))


def log_vonmises(x, mu, kappa):
    """von Mises の対数密度 [rad^-1]。

    f(x | mu, k) = exp(k cos(x - mu)) / (2 pi I0(k))

    素の I0 は kappa >~ 700 でオーバーフローするため、指数スケール済み
    i0e(k) = exp(-k) I0(k) を用いて

        log f = k (cos(x - mu) - 1) - log(2 pi) - log i0e(k)

    と書く。第1項は常に <= 0 なので exp がオーバーフローしない（FR-3.1）。
    """
    return log_vonmises_from_cos(jnp.cos(x - mu), kappa)
