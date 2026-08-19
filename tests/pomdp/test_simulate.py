"""シミュレーション（§5.6 のリカバリ基盤）のテスト。"""
import jax
import jax.numpy as jnp
import numpy as np

from dualrdk.pomdp.circular import DEG, circ_dist
from dualrdk.pomdp.likelihood import subject_log_lik
from dualrdk.pomdp.models import build_model
from dualrdk.pomdp.policy import sample_action
from dualrdk.pomdp.simulate import simulate_subject, stim_arrays
from dualrdk.pomdp.task_reward import env_reward

N_TRIALS = 48


def test_sample_action_matches_kappa():
    """反応ノイズの円周標準偏差が kappa と整合する。"""
    kappa = 15.0
    keys = jax.random.split(jax.random.PRNGKey(0), 20_000)
    f = jax.jit(
        jax.vmap(
            lambda k: sample_action(
                k, jnp.asarray(30.0 * DEG), jnp.asarray(130.0 * DEG),
                jnp.asarray(10.0), jnp.asarray(1.0), jnp.asarray(kappa), jnp.asarray(0.0),
            )
        )
    )
    a = np.asarray(f(keys))
    r = np.abs(np.mean(np.exp(1j * (a - 30 * np.pi / 180))))
    sd_deg = np.degrees(np.sqrt(-2 * np.log(r)))
    assert 13.0 < sd_deg < 17.0


def test_env_reward_matches_spec_table():
    """§0.3 の報酬関数（整数量子化、45 度で 0）。"""
    got = [int(env_reward(e, 0.0)) for e in (0, 5, 25, 44, 45, 46, 50, 90)]
    assert got == [10, 9, 5, 1, 1, 0, 0, 0]


def _stim(seed=0, phi=0.0):
    """実験の生成規則に従って刺激を作る（学習可能な構造を持たせる）。

    条件 A: 白 in phi+[0,45), 黒 in phi+[270,315), 正解=白
    条件 B: 白 in phi+[180,225), 黒 in phi+[90,135), 正解=黒
    オフセットは u >= v（角度差 >= 90 度の棄却条件と等価）。
    結果として正解は常に psi = phi + 67.5 に近い方になる。
    """
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 45, N_TRIALS).astype(float)
    v = rng.integers(0, 45, N_TRIALS).astype(float)
    u, v = np.maximum(u, v), np.minimum(u, v)
    is_a = rng.random(N_TRIALS) < 0.5
    tw = np.where(is_a, phi + u, phi + 180.0 + u) % 360.0
    tb = np.where(is_a, phi + 270.0 + v, phi + 90.0 + v) % 360.0
    target = np.where(is_a, tw, tb)
    zone = (rng.random(N_TRIALS) < 0.3).astype(int)
    return stim_arrays(tw, tb, target, zone, np.ones(N_TRIALS, dtype=bool))


def test_stim_follows_argmin_rule():
    """テスト用の刺激生成が psi = phi + 67.5 の argmin ルールを満たす。"""
    stim = _stim(0)
    d_star = np.asarray(circ_dist(stim["theta_star"], jnp.asarray(67.5 * DEG)))
    other = np.where(
        np.isclose(np.asarray(stim["theta_star"]), np.asarray(stim["theta_w"])),
        np.asarray(stim["theta_b"]),
        np.asarray(stim["theta_w"]),
    )
    d_other = np.asarray(circ_dist(jnp.asarray(other), jnp.asarray(67.5 * DEG)))
    assert np.all(d_star < d_other)


def test_simulated_trials_feed_the_likelihood():
    """シミュレーション出力がそのまま尤度計算に渡せる形になっている。"""
    model = build_model("M2z")
    p = {
        "beta": jnp.asarray(5.0), "alpha_in": jnp.asarray(0.3), "alpha_out": jnp.asarray(0.3),
        "kappa_gen": jnp.asarray(2.0), "kappa": jnp.asarray(15.0), "lam": jnp.asarray(0.05),
    }
    tr = simulate_subject(model, p, _stim(1), jax.random.PRNGKey(0))
    assert np.isfinite(float(subject_log_lik(model, p, tr)))
    r = np.asarray(tr["r_norm"]) * 10.0
    assert np.all((r >= 0) & (r <= 10))
    # 報酬 > 0 は「いずれかのクラウドから 45 度以内」と等価
    near = np.asarray(circ_dist(tr["a"], _stim(1)["theta_star"])) / float(DEG)
    assert np.array_equal(np.asarray(tr["reward_positive"]), near <= 45.0)


def test_higher_beta_gives_better_performance():
    """beta を上げると高報酬側を選ぶ割合が上がる（シミュレータが機能している）。"""
    model = build_model("M2z")
    stim = _stim(2)

    def mean_reward(beta):
        p = {
            "beta": jnp.asarray(beta), "alpha_in": jnp.asarray(0.4), "alpha_out": jnp.asarray(0.4),
            "kappa_gen": jnp.asarray(2.0), "kappa": jnp.asarray(15.0), "lam": jnp.asarray(0.02),
        }
        vals = [
            float(np.mean(np.asarray(simulate_subject(model, p, stim, jax.random.PRNGKey(s))["r_norm"])))
            for s in range(30)
        ]
        return float(np.mean(vals))

    assert mean_reward(20.0) > mean_reward(0.05)
