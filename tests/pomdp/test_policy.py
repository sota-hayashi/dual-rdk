"""§6.4 の方策のテスト。"""
import jax.numpy as jnp
import numpy as np

from dualrdk.pomdp.circular import DEG, TWO_PI
from dualrdk.pomdp.policy import log_policy, p_white, policy_pdf

TW = jnp.asarray(30.0 * DEG)
TB = jnp.asarray(130.0 * DEG)


def _integrate(d, beta, kappa, lam, n=200_001):
    a = jnp.linspace(0.0, TWO_PI, n)
    dens = np.asarray(policy_pdf(a, TW, TB, jnp.asarray(d), jnp.asarray(beta),
                                 jnp.asarray(kappa), jnp.asarray(lam)))
    return float(np.trapezoid(dens, np.asarray(a)))


def test_policy_is_normalized():
    """混合密度は円周上で 1 に積分される。"""
    for d, beta, kappa, lam in [(0.6, 5.0, 15.0, 0.05), (-1.0, 1.0, 3.0, 0.2), (0.0, 8.0, 100.0, 0.01)]:
        assert np.isclose(_integrate(d, beta, kappa, lam), 1.0, atol=1e-6)


def test_lapse_sets_a_floor():
    """2 峰から遠い反応の密度は lapse の床 lam/(2pi) に一致する。

    lam がないとここが 0 になり log pi = -inf で尤度が壊れる。
    """
    lam = 0.05
    far = TW + jnp.asarray(180.0 * DEG)
    dens = float(policy_pdf(far, TW, TB, jnp.asarray(0.6), jnp.asarray(5.0),
                            jnp.asarray(15.0), jnp.asarray(lam)))
    # 反対側の峰の裾がわずかに漏れるので rtol は 1e-4
    assert np.isclose(dens, lam / float(TWO_PI), rtol=1e-4)


def test_p_white_matches_two_arm_softmax():
    """sigmoid(beta d) は 2 アーム softmax の書き換えと厳密に一致する。"""
    beta, v_w, v_b = 3.0, 0.8, 0.2
    soft = np.exp(beta * v_w) / (np.exp(beta * v_w) + np.exp(beta * v_b))
    assert np.isclose(float(p_white(jnp.asarray(v_w - v_b), jnp.asarray(beta))), soft, atol=1e-12)


def test_peak_ratio_tracks_choice_probability():
    """2 峰の密度比は p_W/(1-p_W) にほぼ一致する（lapse の床の分だけずれる）。"""
    d, beta, kappa, lam = 0.6, 5.0, 15.0, 0.0
    dw = float(policy_pdf(TW, TW, TB, jnp.asarray(d), jnp.asarray(beta), jnp.asarray(kappa), jnp.asarray(lam)))
    db = float(policy_pdf(TB, TW, TB, jnp.asarray(d), jnp.asarray(beta), jnp.asarray(kappa), jnp.asarray(lam)))
    pw = float(p_white(jnp.asarray(d), jnp.asarray(beta)))
    assert np.isclose(dw / db, pw / (1 - pw), rtol=1e-3)


def test_log_policy_finite_at_extreme_params():
    for beta, kappa, lam in [(50.0, 1000.0, 1e-4), (0.01, 0.1, 0.5)]:
        v = float(log_policy(jnp.asarray(2.0), TW, TB, jnp.asarray(1.0),
                             jnp.asarray(beta), jnp.asarray(kappa), jnp.asarray(lam)))
        assert np.isfinite(v)
