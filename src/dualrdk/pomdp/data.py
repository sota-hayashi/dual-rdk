"""入力データの変換と検証（§0.2, §0.3）。

入力は前処理済みの `concat_list: List[Tuple[str, pd.DataFrame]]`。参加者除外
（4 基準・26 名）と OOZ ラベルは前処理済みとして受け取り、ここでは再計算しない。

learning ステージ 48 試行のみを扱う。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd

from dualrdk.pomdp.belief import GRID_N
from dualrdk.pomdp.task_reward import REWARD_MAX, env_reward, wrap_deg

N_TRIALS = 48
PSI_OFFSET_DEG = 67.5  # psi = phi + 67.5（§0.3）
DELTA_RANGE_DEG = (90.0, 134.0)

REQUIRED_COLUMNS = (
    "num_trial",
    "rt",
    "response_angle_rdk",
    "target_direction",
    "distractor_direction",
    "target_group",
    "reward_points",
    "ooz",
)

# phi は尤度計算には不要（M2 系は theta_W/theta_B/a/r だけ、M3 系も信念を一様から
# 始めるので phi を知らない）。必要になるのは検証（argmin ルールの照合）と、
# 参加者間で V(theta) を比較するための座標回転だけ。したがって任意項目とする。
OPTIONAL_COLUMNS = ("session_rotation",)

TIDY_COLUMNS = (
    "subject_id",
    "trial",
    "phi_deg",
    "theta_white_deg",
    "theta_black_deg",
    "response_deg",
    "target_deg",
    "reward",
    "zone",
    "valid",
)


# --------------------------------------------------------------------------
# concat_list -> tidy
# --------------------------------------------------------------------------
def _restore_colors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """target/distractor と target_group から白・黒の運動方向を復元する（§0.2）。"""
    tgt = df["target_direction"].to_numpy(dtype=float)
    dis = df["distractor_direction"].to_numpy(dtype=float)
    is_white = df["target_group"].astype(str).str.lower().to_numpy() == "white"
    theta_white = np.where(is_white, tgt, dis)
    theta_black = np.where(is_white, dis, tgt)
    return theta_white, theta_black


def tidy_from_concat_list(
    concat_list: Iterable[Tuple[str, pd.DataFrame]],
    n_trials: int = N_TRIALS,
) -> pd.DataFrame:
    """concat_list を tidy テーブルに変換する。"""
    rows: List[pd.DataFrame] = []
    for subj_id, df in concat_list:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{subj_id}: 必要な列が無い: {missing}")

        # num_trial は 0 始まりの場合も 1 始まりの場合もある
        work = df.sort_values("num_trial").reset_index(drop=True)
        if len(work) != n_trials:
            raise ValueError(
                f"{subj_id}: learning ステージの試行数が {len(work)}（{n_trials} を期待）"
            )

        theta_white, theta_black = _restore_colors(work)
        if "session_rotation" in work.columns:
            phi = work["session_rotation"].to_numpy(dtype=float)
            finite = np.isfinite(phi)
            if finite.any() and not np.allclose(phi[finite], phi[finite][0], atol=1e-9):
                raise ValueError(f"{subj_id}: session_rotation が参加者内で一定でない")
        else:
            phi = np.full(len(work), np.nan)

        resp = work["response_angle_rdk"].to_numpy(dtype=float)
        valid = (
            np.isfinite(work["rt"].to_numpy(dtype=float))
            & np.isfinite(resp)
            & np.isfinite(theta_white)
            & np.isfinite(theta_black)
            & np.isfinite(work["reward_points"].to_numpy(dtype=float))
        )

        rows.append(
            pd.DataFrame(
                {
                    "subject_id": subj_id,
                    "trial": np.arange(1, n_trials + 1),
                    "phi_deg": phi,
                    "theta_white_deg": theta_white,
                    "theta_black_deg": theta_black,
                    "response_deg": resp,
                    "target_deg": work["target_direction"].to_numpy(dtype=float),
                    "reward": work["reward_points"].to_numpy(dtype=float),
                    "zone": work["ooz"].to_numpy(dtype=float),
                    "valid": valid,
                }
            )
        )

    tidy = pd.concat(rows, ignore_index=True)
    # 無効試行の数値列は下流で使わないが、NaN のまま配列化すると計算が汚染される。
    # phi_deg は意図的に NaN のまま残す（欠測を「0 度」と誤認させないため）。
    for col in ("response_deg", "theta_white_deg", "theta_black_deg", "target_deg", "reward", "zone"):
        tidy[col] = tidy[col].fillna(0.0)
    tidy["zone"] = tidy["zone"].astype(int)
    return tidy[list(TIDY_COLUMNS)]


# --------------------------------------------------------------------------
# 検証
# --------------------------------------------------------------------------
def validate_tidy(tidy: pd.DataFrame, strict: bool = False) -> dict:
    """§0.3 の課題定数に照らしてデータを検証する。

    - 2 方向の角度差 Delta が [90, 134] 度に入るか
    - reward が環境の報酬関数と一致するか
    - target_direction が psi = phi + 67.5 に近い方か（argmin ルール）
    """
    v = tidy[tidy["valid"]].copy()
    d_wb = np.abs(wrap_deg(v["theta_white_deg"] - v["theta_black_deg"]))
    delta_ok = (d_wb >= DELTA_RANGE_DEG[0] - 1e-6) & (d_wb <= DELTA_RANGE_DEG[1] + 1e-6)

    r_expected = env_reward(v["response_deg"].to_numpy(), v["target_deg"].to_numpy())
    reward_ok = r_expected == v["reward"].to_numpy().astype(int)

    # argmin ルール（theta_star は psi = phi + 67.5 に近い方）の照合。
    # phi が記録されていない場合は検証不能なのでスキップする。
    phi = v["phi_deg"].to_numpy(dtype=float)
    has_phi = np.isfinite(phi)
    if has_phi.any():
        psi = phi[has_phi] + PSI_OFFSET_DEG
        d_w = np.abs(wrap_deg(v["theta_white_deg"].to_numpy()[has_phi] - psi))
        d_b = np.abs(wrap_deg(v["theta_black_deg"].to_numpy()[has_phi] - psi))
        target_is_white = np.isclose(
            v["theta_white_deg"].to_numpy()[has_phi], v["target_deg"].to_numpy()[has_phi]
        )
        argmin_violations = int(((d_w < d_b) != target_is_white).sum())
    else:
        argmin_violations = None

    report = {
        "n_subjects": int(tidy["subject_id"].nunique()),
        "n_trials_total": int(len(tidy)),
        "n_valid": int(len(v)),
        "delta_violations": int((~delta_ok).sum()),
        "delta_min": float(d_wb.min()) if len(v) else float("nan"),
        "delta_max": float(d_wb.max()) if len(v) else float("nan"),
        "reward_mismatches": int((~reward_ok).sum()),
        "argmin_violations": argmin_violations,
        "phi_available": bool(has_phi.any()),
    }
    if strict:
        keys = ("delta_violations", "reward_mismatches", "argmin_violations")
        problems = {k: report[k] for k in keys if report[k]}
        if problems:
            raise ValueError(f"データ検証に失敗: {problems}")
    return report


# --------------------------------------------------------------------------
# tidy -> モデル入力配列
# --------------------------------------------------------------------------
def arrays_from_tidy(tidy: pd.DataFrame) -> tuple[list[str], dict]:
    """tidy テーブルを (n_subj, T) の配列辞書に変換する。

    角度はラジアン、報酬は r/10 に正規化する（FR-3.3）。
    """
    subjects = sorted(tidy["subject_id"].unique())
    n_subj = len(subjects)
    t_max = int(tidy.groupby("subject_id").size().max())

    def col(name, dtype=float):
        out = np.zeros((n_subj, t_max), dtype=dtype)
        for i, s in enumerate(subjects):
            out[i] = tidy.loc[tidy["subject_id"] == s, name].to_numpy(dtype=dtype)
        return out

    deg2rad = np.pi / 180.0
    theta_w_deg = col("theta_white_deg")
    theta_b_deg = col("theta_black_deg")
    a_deg = col("response_deg")
    target_deg = col("target_deg")
    reward = col("reward")

    trials = {
        "theta_w": jnp.asarray(theta_w_deg * deg2rad),
        "theta_b": jnp.asarray(theta_b_deg * deg2rad),
        "a": jnp.asarray(a_deg * deg2rad),
        "iw": jnp.asarray(np.rint(theta_w_deg).astype(int) % GRID_N),
        "ib": jnp.asarray(np.rint(theta_b_deg).astype(int) % GRID_N),
        "ia": jnp.asarray(np.rint(a_deg).astype(int) % GRID_N),
        # M1 系の状態（どちらの雲が高報酬か）を導くためのターゲット方向。
        # M2 / M3 系は使わない（これを状態に使うのがまさに M1 の仮定）。
        "it": jnp.asarray(np.rint(target_deg).astype(int) % GRID_N),
        "r_norm": jnp.asarray(reward / float(REWARD_MAX)),
        "reward_positive": jnp.asarray(reward > 0),
        "zone": jnp.asarray(col("zone", int)),
        "valid": jnp.asarray(col("valid", bool)),
    }
    # パラメータ非依存の量を先に計算しておく（NUTS の内側ループから三角関数が消える）
    from dualrdk.pomdp.likelihood import precompute_trial_features

    return subjects, precompute_trial_features(trials)


def stimuli_from_tidy(tidy: pd.DataFrame) -> dict:
    """シミュレーション用に刺激・注意状態のみを取り出す（recovery）。"""
    subjects = sorted(tidy["subject_id"].unique())
    n_subj, t_max = len(subjects), int(tidy.groupby("subject_id").size().max())

    def col(name, dtype=float):
        out = np.zeros((n_subj, t_max), dtype=dtype)
        for i, s in enumerate(subjects):
            out[i] = tidy.loc[tidy["subject_id"] == s, name].to_numpy(dtype=dtype)
        return out

    return {
        "theta_white_deg": col("theta_white_deg"),
        "theta_black_deg": col("theta_black_deg"),
        "target_deg": col("target_deg"),
        "zone": col("zone", int),
        "valid": col("valid", bool),
        "subjects": subjects,
    }


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def save_tidy(tidy: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        tidy.to_parquet(path, index=False)
    else:
        tidy.to_csv(path, index=False)


def load_tidy(path: Path) -> pd.DataFrame:
    path = Path(path)
    tidy = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in TIDY_COLUMNS if c not in tidy.columns]
    if missing:
        raise ValueError(f"{path}: 必要な列が無い: {missing}")
    tidy["valid"] = tidy["valid"].astype(bool)
    tidy["zone"] = tidy["zone"].astype(int)
    return tidy


def load_from_raw(data_dir: Path) -> pd.DataFrame:
    """生データディレクトリから learning ステージの tidy テーブルを作る。

    既存の io.load / features.behavior のパイプライン（参加者除外と OOZ
    ラベル付与を含む）をそのまま使う。
    """
    from dualrdk.io.load import load_all_concatenated

    _, learning, _ = load_all_concatenated(Path(data_dir))
    return tidy_from_concat_list(learning)


def main(argv=None):
    """生データから tidy テーブルを書き出す CLI（§6.3 手順0）。

        python -m dualrdk.pomdp.data --data-dir data/raw/online \\
            --out outputs/pomdp/trials.parquet

    参加者除外（4基準）と OOZ ラベルは load_all_concatenated が適用済み。
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(description="learning ステージの tidy テーブルを作る")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw/online"))
    ap.add_argument("--out", type=Path, default=Path("outputs/pomdp/trials.parquet"))
    ap.add_argument("--strict", action="store_true", help="検証違反があればエラーにする")
    args = ap.parse_args(argv)

    tidy = load_from_raw(args.data_dir)
    report = validate_tidy(tidy, strict=args.strict)
    save_tidy(tidy, args.out)
    print("[validate]", json.dumps(report, ensure_ascii=False))
    print(f"[saved] {args.out}  ({len(tidy)} 行, {tidy['subject_id'].nunique()} 名)")
    return tidy


def build_trials(
    theta_white_deg: Sequence[float],
    theta_black_deg: Sequence[float],
    response_deg: Sequence[float],
    reward: Sequence[float],
    zone: Sequence[int],
    valid: Sequence[bool] | None = None,
    target_deg: Sequence[float] | None = None,
) -> dict:
    """テスト・シミュレーション用に 1 参加者分の trials 辞書を組み立てる。"""
    from dualrdk.pomdp.likelihood import derive_choice_fields

    deg2rad = np.pi / 180.0
    tw = np.asarray(theta_white_deg, dtype=float)
    tb = np.asarray(theta_black_deg, dtype=float)
    a = np.asarray(response_deg, dtype=float)
    r = np.asarray(reward, dtype=float)
    if valid is None:
        valid = np.ones_like(tw, dtype=bool)
    trials = {
        "theta_w": jnp.asarray(tw * deg2rad),
        "theta_b": jnp.asarray(tb * deg2rad),
        "a": jnp.asarray(a * deg2rad),
        "iw": jnp.asarray(np.rint(tw).astype(int) % GRID_N),
        "ib": jnp.asarray(np.rint(tb).astype(int) % GRID_N),
        "ia": jnp.asarray(np.rint(a).astype(int) % GRID_N),
        "r_norm": jnp.asarray(r / float(REWARD_MAX)),
        "reward_positive": jnp.asarray(r > 0),
        "zone": jnp.asarray(np.asarray(zone, dtype=int)),
        "valid": jnp.asarray(np.asarray(valid, dtype=bool)),
    }
    if target_deg is not None:
        trials["it"] = jnp.asarray(
            np.rint(np.asarray(target_deg, dtype=float)).astype(int) % GRID_N
        )
    trials.update(derive_choice_fields(trials))
    return trials


if __name__ == "__main__":
    main()
