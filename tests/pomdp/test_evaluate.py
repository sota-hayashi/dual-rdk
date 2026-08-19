"""共通評価指標のテスト（FR-6.5）。

ここで守りたい性質は 2 つ。
- 一致率・選択のみ対数尤度が、推定法にもモデル族にも依存しない定義であること
- 「ターゲット選択確率」ではなく「参加者の実際の選択」を基準にしていること
"""
import numpy as np
import pandas as pd
import pytest

from dualrdk.pomdp.evaluate import (
    add_observed_choice,
    calibration_table,
    evaluate,
    learning_curve,
    merge_latent,
)

N = 12


def _tidy(seed=0):
    rng = np.random.default_rng(seed)
    theta_w = rng.integers(0, 360, N).astype(float)
    theta_b = (theta_w + rng.integers(90, 135, N)) % 360.0
    white_target = rng.random(N) < 0.5
    target = np.where(white_target, theta_w, theta_b)
    # 8 割はターゲット側の雲、2 割は反対側を選ぶ
    pick_target = rng.random(N) < 0.8
    chosen = np.where(pick_target, target, np.where(white_target, theta_b, theta_w))
    resp = np.rint(chosen + rng.normal(0, 5, N)) % 360.0
    return pd.DataFrame(
        {
            "subject_id": ["s0"] * N,
            "trial": np.arange(1, N + 1),
            "phi_deg": np.nan,
            "theta_white_deg": theta_w,
            "theta_black_deg": theta_b,
            "response_deg": resp,
            "target_deg": target,
            "reward": 0.0,
            "zone": 0,
            "valid": True,
        }
    )


def _latent(tidy, p_w):
    return pd.DataFrame(
        {"subject_id": tidy["subject_id"], "trial": tidy["trial"], "p_w": p_w}
    )


def test_observed_choice_matches_nearest_cloud():
    tidy = _tidy()
    obs = add_observed_choice(tidy)
    d_w = np.abs((obs["response_deg"] - obs["theta_white_deg"]) % 360.0)
    d_w = np.minimum(d_w, 360.0 - d_w)
    d_b = np.abs((obs["response_deg"] - obs["theta_black_deg"]) % 360.0)
    d_b = np.minimum(d_b, 360.0 - d_b)
    assert (obs["chose_white"] == (d_w < d_b)).all()


def test_perfect_model_gets_agreement_one():
    """観測選択をそのまま予測すれば一致率 1、対数尤度 0 に近づく。"""
    tidy = _tidy()
    obs = add_observed_choice(tidy)
    p_w = np.where(obs["chose_white"], 0.999, 0.001)
    out = evaluate(_latent(tidy, p_w), tidy)["overall"]
    assert out["agreement"] == pytest.approx(1.0)
    assert out["mean_p_observed"] == pytest.approx(0.999)
    assert out["choice_log_lik"] == pytest.approx(N * np.log(0.999), rel=1e-6)


def test_chance_model_gives_zero_pseudo_r2():
    tidy = _tidy()
    out = evaluate(_latent(tidy, np.full(N, 0.5)), tidy)["overall"]
    assert out["mcfadden_r2"] == pytest.approx(0.0, abs=1e-9)
    assert out["mean_p_observed"] == pytest.approx(0.5)


def test_anti_model_falls_below_chance():
    """観測と逆を予測すると一致率 0。ターゲット基準の指標では見抜けない差。"""
    tidy = _tidy()
    obs = add_observed_choice(tidy)
    p_w = np.where(obs["chose_white"], 0.1, 0.9)
    out = evaluate(_latent(tidy, p_w), tidy)["overall"]
    assert out["agreement"] == pytest.approx(0.0)
    assert out["mcfadden_r2"] < 0.0


def test_predicted_correct_rate_detects_overprediction():
    """常にターゲット側を高確率で予測するモデルは、参加者の正解率を上回る。

    これが「ターゲット選択確率の平均 > 70%」を妥当性の証拠にしてはいけない
    理由である。参加者の実測正解率と並べて初めて意味を持つ。
    """
    tidy = _tidy()
    obs = add_observed_choice(tidy)
    p_w = np.where(obs["white_target"], 0.95, 0.05)
    out = evaluate(_latent(tidy, p_w), tidy)["overall"]
    assert out["predicted_correct_rate"] == pytest.approx(0.95)
    assert out["predicted_correct_rate"] > out["participant_correct_rate"]


def test_invalid_trials_are_dropped():
    tidy = _tidy()
    tidy.loc[0, "valid"] = False
    df = merge_latent(_latent(tidy, np.full(N, 0.5)), tidy)
    assert len(df) == N - 1


def test_calibration_and_learning_curve_shapes():
    tidy = _tidy()
    df = merge_latent(_latent(tidy, np.linspace(0.05, 0.95, N)), tidy)
    cal = calibration_table(df)
    assert {"n", "predicted", "observed", "z"}.issubset(cal.columns)
    assert cal["n"].sum() == len(df)
    curve = learning_curve(df, block=4)
    assert len(curve) == N // 4
    assert curve["n"].sum() == len(df)


def test_holdout_uses_only_late_trials():
    tidy = _tidy()
    obs = add_observed_choice(tidy)
    # 前半は完璧、後半は反転
    p_w = np.where(obs["chose_white"], 0.9, 0.1)
    p_w[N // 2:] = 1.0 - p_w[N // 2:]
    out = evaluate(_latent(tidy, p_w), tidy, holdout_from=N // 2 + 1)
    assert out["overall"]["agreement"] == pytest.approx(0.5)
    assert out["holdout"]["agreement"] == pytest.approx(0.0)
