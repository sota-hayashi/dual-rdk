"""§6.4 の円周ユーティリティのテスト。"""
import jax.numpy as jnp
import numpy as np

from dualrdk.pomdp.circular import DEG, circ_dist, circ_mean, log_vonmises, wrap


def test_circ_dist_wraps_around_zero():
    """circ_dist(0, 359) は 1 度。0 と 359 を最遠点にしてはならない。"""
    d = circ_dist(jnp.asarray(0.0), jnp.asarray(359.0 * DEG))
    assert np.isclose(float(d) / float(DEG), 1.0, atol=1e-9)


def test_wrap_range():
    x = jnp.asarray([0.0, 3.0, 3.5, 6.0, -3.5])
    w = np.asarray(wrap(x))
    assert np.all(w > -np.pi - 1e-9) and np.all(w <= np.pi + 1e-9)


def test_circ_mean_across_zero():
    """359 度と 1 度の円周平均は 0 度。算術平均（180 度）ではない。"""
    x = jnp.asarray([359.0, 1.0]) * DEG
    m = float(circ_mean(x)) / float(DEG)
    # 0 度近傍なので剰余ではなく円距離で比較する（-1e-14 % 360 は 360 に近い値になる）
    assert abs((m + 180.0) % 360.0 - 180.0) < 1e-6


def test_log_vonmises_finite_at_large_kappa():
    """kappa = 1000 でも有限値（素の I0 はオーバーフローする）。"""
    for kappa in (1.0, 15.0, 1000.0):
        v = float(log_vonmises(jnp.asarray(0.3), jnp.asarray(0.1), jnp.asarray(kappa)))
        assert np.isfinite(v)


def test_log_vonmises_integrates_to_one():
    grid = jnp.linspace(0.0, 2 * jnp.pi, 200_001)
    for kappa in (0.5, 15.0, 200.0):
        dens = np.asarray(jnp.exp(log_vonmises(grid, jnp.asarray(1.0), jnp.asarray(kappa))))
        integral = np.trapezoid(dens, np.asarray(grid))
        assert np.isclose(integral, 1.0, atol=1e-6)
