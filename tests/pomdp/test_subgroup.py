"""参加者層別（副次解析）のテスト。"""
import numpy as np
import pandas as pd
import pytest

from dualrdk.pomdp.subgroup import classify, random_split, write_subject_lists

N = 24


def _tidy(subjects=("s0", "s1", "s2", "s3")):
    rng = np.random.default_rng(0)
    rows = []
    for s in subjects:
        tw = rng.integers(0, 360, N).astype(float)
        tb = (tw + rng.integers(90, 135, N)) % 360.0
        white_target = rng.random(N) < 0.5
        target = np.where(white_target, tw, tb)
        resp = np.rint(target + rng.normal(0, 5, N)) % 360.0
        rows.append(
            pd.DataFrame(
                {
                    "subject_id": s, "trial": np.arange(1, N + 1), "phi_deg": np.nan,
                    "theta_white_deg": tw, "theta_black_deg": tb, "response_deg": resp,
                    "target_deg": target, "reward": 0.0, "zone": 0, "valid": True,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _write_fit(tmp_path, name, tidy, p_w):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"subject_id": tidy["subject_id"], "trial": tidy["trial"], "p_w": p_w}
    ).to_csv(d / "latent.csv", index=False)
    return d


def _chose_white(tidy):
    from dualrdk.pomdp.evaluate import add_observed_choice

    return add_observed_choice(tidy)["chose_white"].to_numpy()


def test_classify_splits_by_advantage(tmp_path):
    """学習モデルが良い参加者だけが learner になる。"""
    tidy = _tidy()
    cw = _chose_white(tidy)
    base = _write_fit(tmp_path, "M0", tidy, np.full(len(tidy), 0.5))
    # s0, s1 は完璧に予測、s2, s3 は偶然以下
    good = tidy["subject_id"].isin(["s0", "s1"]).to_numpy()
    p_w = np.where(good, np.where(cw, 0.9, 0.1), np.where(cw, 0.3, 0.7))
    learn = _write_fit(tmp_path, "M2z", tidy, p_w)

    a = classify(base, learn, tidy)
    g = a.set_index("subject_id")["group"].to_dict()
    assert g["s0"] == "learner" and g["s1"] == "learner"
    assert g["s2"] == "nonlearner" and g["s3"] == "nonlearner"
    assert (a.loc[a.group == "learner", "advantage"] > 0).all()


def test_classify_agreement_metric_agrees_on_clear_cases(tmp_path):
    tidy = _tidy()
    base = _write_fit(tmp_path, "M0", tidy, np.full(len(tidy), 0.5))
    learn = _write_fit(tmp_path, "M2z", tidy, np.full(len(tidy), 0.5))
    a = classify(base, learn, tidy, metric="agreement")
    # 完全に同じ予測なら advantage は 0 で、全員 nonlearner 側に落ちる
    assert np.allclose(a["advantage"], 0.0)
    assert (a["group"] == "nonlearner").all()


def test_margin_moves_borderline_subjects(tmp_path):
    """僅差の参加者は margin を上げると learner から外れる。"""
    tidy = _tidy()
    obs = _chose_white(tidy)
    base = _write_fit(tmp_path, "M0", tidy, np.full(len(tidy), 0.5))
    # 観測選択の向きにわずかに寄せただけ（全員がぎりぎり learner になる）
    learn = _write_fit(tmp_path, "M2z", tidy, np.where(obs, 0.55, 0.45))

    assert (classify(base, learn, tidy)["group"] == "learner").all()
    strict = classify(base, learn, tidy, margin=50.0)
    assert (strict["group"] == "nonlearner").all()


def test_random_split_preserves_group_sizes():
    a = pd.DataFrame(
        {"subject_id": [f"s{i}" for i in range(10)],
         "group": ["learner"] * 7 + ["nonlearner"] * 3,
         "advantage": np.arange(10.0)}
    )
    b = random_split(a, seed=1)
    assert b["group"].value_counts().to_dict() == a["group"].value_counts().to_dict()
    assert set(b["subject_id"]) == set(a["subject_id"])
    assert not (b["group"] == a["group"]).all()  # seed=1 では実際に入れ替わる


def test_write_subject_lists_roundtrip(tmp_path):
    from dualrdk.pomdp.fit import read_subject_list

    a = pd.DataFrame(
        {"subject_id": ["a", "b", "c"], "group": ["learner", "nonlearner", "learner"],
         "advantage": [1.0, -1.0, 2.0]}
    )
    paths = write_subject_lists(a, tmp_path)
    assert read_subject_list(paths["learner"]) == ["a", "c"]
    assert read_subject_list(paths["nonlearner"]) == ["b"]


def test_classify_rejects_unknown_metric(tmp_path):
    tidy = _tidy()
    d = _write_fit(tmp_path, "M0", tidy, np.full(len(tidy), 0.5))
    with pytest.raises(ValueError, match="metric"):
        classify(d, d, tidy, metric="accuracy")
