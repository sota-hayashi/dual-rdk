"""M1 / M1z（2 値状態・2 値行動）と 2 段階推定のテスト。"""
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from dualrdk.pomdp.data import build_trials
from dualrdk.pomdp.likelihood import scan_subject, subject_log_lik
from dualrdk.pomdp.models import BINARY_MODELS, build_model
from dualrdk.pomdp.perceptual import fit_perceptual, load_fixed
from dualrdk.pomdp.policy import log_choice_lik, p_white
from dualrdk.pomdp.task_reward import env_reward

N_TRIALS = 48


def _subject(seed=0, p_target=0.75):
    rng = np.random.default_rng(seed)
    theta_w = rng.integers(0, 360, N_TRIALS).astype(float)
    theta_b = (theta_w + rng.integers(90, 135, N_TRIALS)) % 360.0
    white_target = rng.random(N_TRIALS) < 0.5
    target = np.where(white_target, theta_w, theta_b)
    other = np.where(white_target, theta_b, theta_w)
    chosen = np.where(rng.random(N_TRIALS) < p_target, target, other)
    a = np.rint(chosen + rng.normal(0, 8, N_TRIALS)) % 360.0
    r = env_reward(a, target)
    zone = (rng.random(N_TRIALS) < 0.3).astype(int)
    return build_trials(theta_w, theta_b, a, r, zone, target_deg=target)


def _p(model, **over):
    base = {
        "beta": 5.0, "alpha": 0.25, "alpha_in": 0.25, "alpha_out": 0.25,
        "c_black": 0.0, "kappa": 15.0, "kappa_gen": 2.0, "lam": 0.05, "c": 0.4,
        "eps": 0.02, "eps_in": 0.02, "eps_out": 0.02, "w": 0.785,
    }
    base.update(over)
    p = {k: jnp.asarray(base[k]) for k in model.param_names}
    p.update(model.fixed)
    return p


def test_m1_state_and_choice_fields_exist():
    tr = _subject()
    assert "s_state" in tr and "chose_white" in tr
    # s_state = 0 は白が高報酬側
    s = np.asarray(tr["s_state"])
    assert set(np.unique(s)).issubset({0, 1})


def test_m1_log_pi_is_a_probability_mass():
    """M1 の log_pi は 2 値の確率質量なので、必ず 0 以下になる。

    M2 / M3 系の log_pi は円環上の密度で、正の値も取りうる。この違いが
    「族をまたいで尤度を比べてはならない」ことの実体である。
    """
    model = build_model("M1z")
    _, outs = scan_subject(model, _p(model), _subject())
    assert np.all(np.asarray(outs["log_pi"]) <= 0.0)

    m2 = build_model("M2z")
    _, outs2 = scan_subject(m2, _p(m2), _subject())
    assert np.asarray(outs2["log_pi"]).max() > 0.0


def test_m1_has_no_perceptual_parameters():
    for name in BINARY_MODELS:
        names = build_model(name).param_names
        assert "kappa" not in names and "lam" not in names and "kappa_gen" not in names


def test_m1z_reduces_to_m1_when_alphas_equal():
    tr = _subject()
    m1, m1z = build_model("M1"), build_model("M1z")
    ll1 = float(subject_log_lik(m1, _p(m1, alpha=0.3), tr))
    ll2 = float(subject_log_lik(m1z, _p(m1z, alpha_in=0.3, alpha_out=0.3), tr))
    assert ll1 == pytest.approx(ll2, rel=1e-10)


def test_m1_is_rotation_invariant():
    """M1 の状態も選択も角度の絶対値に依存しない。"""
    model = build_model("M1z")
    rng = np.random.default_rng(3)
    theta_w = rng.integers(0, 360, N_TRIALS).astype(float)
    theta_b = (theta_w + rng.integers(90, 135, N_TRIALS)) % 360.0
    white_target = rng.random(N_TRIALS) < 0.5
    target = np.where(white_target, theta_w, theta_b)
    a = np.rint(target + rng.normal(0, 8, N_TRIALS)) % 360.0
    r = env_reward(a, target)
    zone = np.zeros(N_TRIALS, dtype=int)

    lls = []
    for rot in (0.0, 137.0):
        tr = build_trials(
            (theta_w + rot) % 360, (theta_b + rot) % 360, (a + rot) % 360,
            r, zone, target_deg=(target + rot) % 360,
        )
        lls.append(float(subject_log_lik(model, _p(model), tr)))
    assert lls[0] == pytest.approx(lls[1], abs=1e-10)


def test_c_black_shifts_choice_toward_black():
    model = build_model("M1z")
    tr = _subject()
    _, a = scan_subject(model, _p(model, c_black=0.0), tr)
    _, b = scan_subject(model, _p(model, c_black=2.0), tr)
    assert np.asarray(b["p_w"]).mean() < np.asarray(a["p_w"]).mean()


def test_c_black_shifts_m2_policy_too():
    model = build_model("M2z")
    tr = _subject()
    _, a = scan_subject(model, _p(model, c_black=0.0), tr)
    _, b = scan_subject(model, _p(model, c_black=2.0), tr)
    assert np.asarray(b["p_w"]).mean() < np.asarray(a["p_w"]).mean()


def test_log_choice_lik_matches_m1_log_pi():
    """共通通貨（log_choice_lik）が M1 の尤度と一致する。"""
    model = build_model("M1")
    tr = _subject()
    p = _p(model)
    _, outs = scan_subject(model, p, tr)
    direct = log_choice_lik(tr["chose_white"], outs["d"], p["beta"], p["c_black"], lam=None)
    assert np.allclose(np.asarray(outs["log_pi"]), np.asarray(direct), atol=1e-12)


def test_p_white_consistent_with_choice_logit():
    d = jnp.linspace(-1.0, 1.0, 11)
    assert np.allclose(
        np.asarray(p_white(d, 3.0, 0.5)),
        1.0 / (1.0 + np.exp(-(3.0 * np.asarray(d) - 0.5))),
    )


# --------------------------------------------------------------------------
# 2 段階推定
# --------------------------------------------------------------------------
def _tidy_from_trials(trials_list):
    rows = []
    for i, (tw, tb, a, target) in enumerate(trials_list):
        n = len(tw)
        rows.append(
            pd.DataFrame(
                {
                    "subject_id": f"s{i}", "trial": np.arange(1, n + 1), "phi_deg": np.nan,
                    "theta_white_deg": tw, "theta_black_deg": tb, "response_deg": a,
                    "target_deg": target, "reward": env_reward(a, target).astype(float),
                    "zone": 0, "valid": True,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _make_tidy(kappa_true, n_subj=3, seed=0):
    rng = np.random.default_rng(seed)
    sd_deg = np.rad2deg(1.0 / np.sqrt(kappa_true))
    out = []
    for _ in range(n_subj):
        tw = rng.integers(0, 360, 200).astype(float)
        tb = (tw + rng.integers(90, 135, 200)) % 360.0
        pick_white = rng.random(200) < 0.5
        center = np.where(pick_white, tw, tb)
        a = np.rint(center + rng.normal(0, sd_deg, 200)) % 360.0
        target = np.where(rng.random(200) < 0.5, tw, tb)
        out.append((tw, tb, a, target))
    return _tidy_from_trials(out)


def test_perceptual_recovers_kappa():
    """段階1が既知の kappa を復元する。"""
    for kappa_true in (10.0, 40.0):
        df = fit_perceptual(_make_tidy(kappa_true, seed=int(kappa_true)))
        assert df["converged"].all()
        # 応答ノイズしか入れていないので、そこそこ近い値が出るはず
        assert df["kappa"].median() == pytest.approx(kappa_true, rel=0.45)


def test_fixed_perceptual_removes_parameters(tmp_path):
    df = fit_perceptual(_make_tidy(15.0, n_subj=2))
    path = tmp_path / "perceptual.csv"
    df.to_csv(path, index=False)

    fixed = load_fixed(path, ["s0", "s1"])
    model = build_model("M2z", fixed=fixed)
    assert "kappa" not in model.param_names
    assert "lam" not in model.param_names
    assert len(model.param_names) == 5  # beta, alpha_in, alpha_out, kappa_gen, c_black
    assert np.asarray(model.fixed["kappa"]).shape == (2,)


def test_load_fixed_rejects_missing_subject(tmp_path):
    df = fit_perceptual(_make_tidy(15.0, n_subj=2))
    path = tmp_path / "perceptual.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="無い参加者"):
        load_fixed(path, ["s0", "s1", "s2"])


def test_build_model_rejects_unknown_fixed():
    with pytest.raises(ValueError, match="固定できない"):
        build_model("M1z", fixed={"kappa": 10.0})
