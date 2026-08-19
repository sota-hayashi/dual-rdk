"""§6.4 の尤度・モデルのテスト（回転同変性、M2z == M2 の縮退）。"""
import jax.numpy as jnp
import numpy as np
import pytest

from dualrdk.pomdp.data import build_trials
from dualrdk.pomdp.likelihood import subject_log_lik
from dualrdk.pomdp.models import MODEL_NAMES, build_model
from dualrdk.pomdp.task_reward import env_reward

N_TRIALS = 48


def _fake_subject(seed=0, rotate_deg=0.0):
    """刺激・反応を作る。報酬は環境の報酬関数から決定論的に決める。"""
    rng = np.random.default_rng(seed)
    theta_w = rng.integers(0, 360, N_TRIALS).astype(float)
    delta = rng.integers(90, 135, N_TRIALS).astype(float)
    theta_b = (theta_w + delta) % 360.0
    # 半分の試行で白、半分で黒を「正解」にする
    target_is_white = rng.random(N_TRIALS) < 0.5
    target = np.where(target_is_white, theta_w, theta_b)
    a = (np.where(rng.random(N_TRIALS) < 0.7, target, np.where(target_is_white, theta_b, theta_w))
         + rng.normal(0, 8, N_TRIALS))
    a = np.rint(a) % 360.0
    r = env_reward(a, target)
    zone = (rng.random(N_TRIALS) < 0.3).astype(int)

    rot = rotate_deg
    return build_trials(
        (theta_w + rot) % 360.0,
        (theta_b + rot) % 360.0,
        (a + rot) % 360.0,
        r,
        zone,
        target_deg=(target + rot) % 360.0,
    )


def _params(model):
    base = {
        "beta": 5.0, "kappa": 15.0, "lam": 0.05, "kappa_gen": 2.0,
        "alpha": 0.25, "alpha_in": 0.25, "alpha_out": 0.25,
        "eps": 0.02, "eps_in": 0.02, "eps_out": 0.02, "c": 0.4, "w": 0.785,
        "c_black": 0.3,
    }
    p = {k: jnp.asarray(base[k]) for k in model.param_names}
    p.update(model.fixed)
    return p


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_log_lik_is_finite(name):
    model = build_model(name)
    ll = float(subject_log_lik(model, _params(model), _fake_subject()))
    assert np.isfinite(ll)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_rotational_equivariance(name):
    """全角度を delta 回転しても対数尤度が不変（課題は回転同変）。"""
    model = build_model(name)
    p = _params(model)
    ll0 = float(subject_log_lik(model, p, _fake_subject(seed=1, rotate_deg=0.0)))
    ll1 = float(subject_log_lik(model, p, _fake_subject(seed=1, rotate_deg=137.0)))
    assert abs(ll0 - ll1) < 1e-8


def test_m2z_reduces_to_m2_when_alphas_equal():
    """alpha_in = alpha_out の M2z は M2 と対数尤度が一致する。"""
    tr = _fake_subject(seed=2)
    m2, m2z = build_model("M2"), build_model("M2z")
    p2 = _params(m2)
    p2z = _params(m2z)
    ll2 = float(subject_log_lik(m2, p2, tr))
    ll2z = float(subject_log_lik(m2z, p2z, tr))
    assert abs(ll2 - ll2z) < 1e-10


def test_m3z_reduces_to_m3_when_eps_equal():
    tr = _fake_subject(seed=3)
    m3, m3z = build_model("M3"), build_model("M3z")
    ll3 = float(subject_log_lik(m3, _params(m3), tr))
    ll3z = float(subject_log_lik(m3z, _params(m3z), tr))
    assert abs(ll3 - ll3z) < 1e-10


def test_zone_dependent_alpha_changes_likelihood():
    """alpha_in != alpha_out なら尤度が変わる（zone が実際に効いている）。"""
    tr = _fake_subject(seed=4)
    m2z = build_model("M2z")
    p = _params(m2z)
    p_diff = dict(p)
    p_diff["alpha_out"] = jnp.asarray(0.9)
    assert abs(float(subject_log_lik(m2z, p, tr)) - float(subject_log_lik(m2z, p_diff, tr))) > 1e-6


def test_invalid_trials_are_skipped():
    """無効試行は尤度に寄与せず、状態も更新しない。"""
    rng = np.random.default_rng(5)
    theta_w = rng.integers(0, 360, N_TRIALS).astype(float)
    theta_b = (theta_w + 100.0) % 360.0
    a = (theta_w + rng.normal(0, 5, N_TRIALS)) % 360.0
    r = env_reward(a, theta_w)
    zone = np.zeros(N_TRIALS, dtype=int)
    valid = np.ones(N_TRIALS, dtype=bool)
    valid[10:20] = False

    model = build_model("M2z")
    p = _params(model)
    full = build_trials(theta_w, theta_b, a, r, zone, valid)
    ll_masked = float(subject_log_lik(model, p, full))

    keep = valid
    dropped = build_trials(theta_w[keep], theta_b[keep], a[keep], r[keep], zone[keep])
    ll_dropped = float(subject_log_lik(model, p, dropped))
    assert abs(ll_masked - ll_dropped) < 1e-8


@pytest.mark.parametrize("name", ["M3", "M3z"])
def test_gradients_are_finite(name):
    """NUTS に必要な勾配が有限であること。"""
    import jax

    model = build_model(name)
    tr = _fake_subject(seed=6)
    p = _params(model)

    def f(beta):
        q = dict(p)
        q["beta"] = beta
        return subject_log_lik(model, q, tr)

    g = float(jax.grad(f)(jnp.asarray(5.0)))
    assert np.isfinite(g)
