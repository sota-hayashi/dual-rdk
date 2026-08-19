"""§6.4 の信念表現のテスト。"""
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

from dualrdk.pomdp.agent_reward import ordinal_log_lik
from dualrdk.pomdp.belief import (
    GRID_N,
    belief_to_q,
    belief_update,
    belief_update_checked,
    log_belief_by_counting,
    uniform_log_belief,
    white_is_target_mask,
    white_weight,
)
from dualrdk.pomdp.circular import DEG

IW, IB = 30, 130                      # 格子インデックス（整数度）
THETA_W = IW * DEG
THETA_B = IB * DEG


def test_uniform_belief_gives_q_half():
    """一様信念では q = 0.5。"""
    q = float(belief_to_q(uniform_log_belief(), IW, IB))
    assert np.isclose(q, 0.5, atol=1e-10)


def test_white_weight_splits_circle_in_half():
    """重みの総和はちょうど半円分（タイは 0.5 で分割される）。"""
    w = np.asarray(white_weight(IW, IB))
    assert np.isclose(w.sum(), GRID_N / 2, atol=1e-10)
    # bool マスクは二等分線上の格子点を落とすので 1 点ずれうる
    mask = np.asarray(white_is_target_mask(IW, IB))
    assert abs(int(mask.sum()) - GRID_N // 2) <= 1


@pytest.mark.parametrize("psi_deg,expect_white", [(0.0, True), (170.0, False)])
def test_point_belief_matches_argmin_rule(psi_deg, expect_white):
    """psi に集中した信念では、q が argmin ルール（どちらが近いか）と一致する。"""
    log_b = np.full(GRID_N, -np.inf)
    log_b[int(psi_deg) % GRID_N] = 0.0
    q = float(belief_to_q(jnp.asarray(log_b), IW, IB))
    assert np.isclose(q, 1.0 if expect_white else 0.0, atol=1e-10)


def test_update_keeps_normalization():
    """更新後も logsumexp(log_b) == 0（FR-2.1）。"""
    log_b = uniform_log_belief()
    rng = np.random.default_rng(0)
    for _ in range(20):
        log_L = jnp.asarray(np.where(rng.random(GRID_N) < 0.5, 0.0, np.log(0.02)))
        log_b, _ = belief_update(log_b, log_L)
        assert np.isclose(float(logsumexp(log_b)), 0.0, atol=1e-10)


def test_update_checked_raises_on_total_collapse():
    """事後が全域ゼロになったら例外（FR-2.3）。黙ってリセットしてはならない。"""
    log_b = np.asarray(uniform_log_belief())
    with pytest.raises(ValueError):
        belief_update_checked(log_b, np.full(GRID_N, -np.inf))


def test_sequential_filter_matches_counting_closed_form():
    """逐次フィルタと矛盾カウンタの閉形式が一致する（FR-2.5, §2.4）。"""
    rng = np.random.default_rng(1)
    eps = 0.02
    n_trials = 48
    iw = rng.integers(0, 360, n_trials)
    ib = (iw + rng.integers(90, 135, n_trials)) % 360
    theta_w = iw * float(DEG)
    theta_b = ib * float(DEG)
    a = rng.integers(0, 360, n_trials) * float(DEG)
    reward_positive = rng.random(n_trials) < 0.5

    log_b = uniform_log_belief()
    consistent = np.zeros((n_trials, GRID_N), dtype=bool)
    seq = np.zeros((n_trials, GRID_N))
    for t in range(n_trials):
        mask = white_is_target_mask(int(iw[t]), int(ib[t]))
        tr = {
            "theta_w": jnp.asarray(theta_w[t]),
            "theta_b": jnp.asarray(theta_b[t]),
            "a": jnp.asarray(a[t]),
            "reward_positive": jnp.asarray(bool(reward_positive[t])),
        }
        log_L = ordinal_log_lik(tr, mask, jnp.asarray(eps))
        consistent[t] = np.asarray(log_L) == 0.0
        log_b, _ = belief_update(log_b, log_L)
        seq[t] = np.asarray(log_b)

    closed = log_belief_by_counting(consistent, np.log(eps))
    assert np.max(np.abs(seq - closed)) < 1e-10
