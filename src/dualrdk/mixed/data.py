"""混合モデル用の試行水準データフレームを組み立てる。

LMM（目的変数 = log RT）でも GLMM（目的変数 = 選択の二値）でも同じ
long-format のテーブルを使うので、構築はここに一本化する。

1 行 = 1 試行。参加者 61 名 × 最大 48 試行。
"""
import contextlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from dualrdk import io as _io  # noqa: F401  (io.load を確実に import させる)
from dualrdk.config import ONLINE_DATA_DIR
from dualrdk.io import load as _load
from dualrdk.io.load import load_all_concatenated
from dualrdk.mixed.config import RT_CEILING_MS, RT_FLOOR_MS


@contextlib.contextmanager
def _upstream_rt_filter_disabled():
    """io.load.exclude_trials の RT 上限を一時的に無効化する。

    上流の閾値が有効だと、除外がここに届く前に済んでしまい、
    「除外が試行位置に偏っていないか」「rt_cv が歪んでいないか」といった
    診断が空振りする。除外の判断と記録はこのモジュールが持つ方針なので、
    読み込み時は bad_response の処理だけ残して RT 上限を外す。
    """
    original = _load.exclude_trials

    def bad_response_only(df, rt_threshold=None):
        df = df.copy()
        if "rt" not in df.columns or "bad_response" not in df.columns:
            raise ValueError("DataFrame lacks 'rt' / 'bad_response' column.")
        df.loc[df["bad_response"] == True, "rt"] = np.nan  # noqa: E712
        return df

    _load.exclude_trials = bad_response_only
    try:
        yield
    finally:
        _load.exclude_trials = original

# 試行水準テーブルに載せる列（存在するものだけ拾う）
_TRIAL_COLS = [
    "num_trial",
    "rt",
    "bad_response",
    "chosen_item",
    "chosen_color",
    "angular_error_target",
    "angular_error_distractor",
    "reward",
    "is_ooz",
]


@dataclass
class TrialData:
    """試行水準データと、そこに至るまでの除外の記録。"""

    df: pd.DataFrame
    n_raw: int
    n_missing_rt: int
    n_rt_ceiling: int
    n_rt_floor: int
    rt_ceiling_ms: Optional[float]
    rt_floor_ms: Optional[float]
    removed: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_trials(self) -> int:
        return len(self.df)

    @property
    def n_subjects(self) -> int:
        return self.df["subject"].nunique()

    @property
    def trials_per_subject(self) -> pd.Series:
        return self.df.groupby("subject").size()

    def exclusion_summary(self) -> pd.DataFrame:
        """除外の内訳を 1 表にまとめる。"""
        n = self.trials_per_subject
        rows = [
            ("読み込んだ試行", self.n_raw, ""),
            ("RT 欠損 (bad_response 含む)", -self.n_missing_rt, ""),
            (
                f"RT > {self.rt_ceiling_ms:.0f} ms" if self.rt_ceiling_ms else "RT 上限なし",
                -self.n_rt_ceiling,
                "離席とみなす",
            ),
            (
                f"RT < {self.rt_floor_ms:.0f} ms" if self.rt_floor_ms else "RT 下限なし（速い側は除外しない）",
                -self.n_rt_floor,
                "" if self.rt_floor_ms else "RT 最小値 547 ms、予期反応なし",
            ),
            ("解析に使用", self.n_trials, f"{self.n_subjects} 名"),
            ("参加者あたり試行数", f"{n.min()}–{n.max()}", f"中央値 {n.median():.0f}"),
        ]
        return pd.DataFrame(rows, columns=["段階", "試行数", "備考"])


def build_trial_frame(
    data_dir=ONLINE_DATA_DIR,
    rt_ceiling_ms: Optional[float] = RT_CEILING_MS,
    rt_floor_ms: Optional[float] = RT_FLOOR_MS,
    verbose: bool = False,
) -> TrialData:
    """学習フェーズの試行水準データフレームを作る。

    Parameters
    ----------
    rt_ceiling_ms : float or None
        これを超える RT の試行を除外する。None で無効。

        読み込み時は io.load.exclude_trials の RT 上限を無効化してから
        ここで適用する。上流の設定に関わらず、除外の判断・記録・診断が
        このモジュールに一元化される。
    rt_floor_ms : float or None
        これ未満の RT を除外する。既定は None。
    """
    with _upstream_rt_filter_disabled():
        _, learning, _ = load_all_concatenated(data_dir)

    frames = []
    for subj_id, d in learning:
        d = d.copy()
        d["subject"] = subj_id
        cols = ["subject"] + [c for c in _TRIAL_COLS if c in d.columns]
        frames.append(d[cols])
    raw = pd.concat(frames, ignore_index=True)

    n_raw = len(raw)
    valid = raw["rt"].notna() & (raw["rt"] > 0)
    n_missing = int((~valid).sum())
    # 派生列は分割前に作る。除外された試行にも trial_c が付いていないと、
    # 「除外が試行位置に偏っていないか」を診断できない。
    df = _add_derived(raw[valid].copy())

    removed_parts = []
    n_ceiling = n_floor = 0
    if rt_ceiling_ms is not None:
        over = df["rt"] > rt_ceiling_ms
        n_ceiling = int(over.sum())
        removed_parts.append(df[over].assign(reason="rt_ceiling"))
        df = df[~over]
    if rt_floor_ms is not None:
        under = df["rt"] < rt_floor_ms
        n_floor = int(under.sum())
        removed_parts.append(df[under].assign(reason="rt_floor"))
        df = df[~under]

    removed = (
        pd.concat(removed_parts, ignore_index=True)
        if removed_parts
        else pd.DataFrame(columns=list(df.columns) + ["reason"])
    )

    out = TrialData(
        df=df.reset_index(drop=True),
        n_raw=n_raw,
        n_missing_rt=n_missing,
        n_rt_ceiling=n_ceiling,
        n_rt_floor=n_floor,
        rt_ceiling_ms=rt_ceiling_ms,
        rt_floor_ms=rt_floor_ms,
        removed=removed,
    )
    if verbose:
        print(out.exclusion_summary().to_string(index=False))
    return out


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """モデルが使う派生列を足す。"""
    df = df.copy()

    # --- 目的変数（LMM） -----------------------------------------------
    df["logrt"] = np.log(df["rt"])

    # --- 試行番号 --------------------------------------------------------
    # trial_c: 1 試行目を 0 にした中心化。切片 = 実験開始時点の水準になり、
    #          ランダム切片が「開始時点の個人差」として読める。
    # trial_s: -1..+1 に正規化。GLMM で係数の桁を揃えたいときに使う。
    t = df["num_trial"].astype(float)
    t0 = t.min()
    df["trial"] = t
    df["trial_c"] = t - t0
    span = df["trial_c"].max()
    df["trial_s"] = (df["trial_c"] - span / 2) / (span / 2) if span > 0 else 0.0

    # --- 目的変数（将来の GLMM 用） -------------------------------------
    # chosen_item: 1 = ターゲット, 0 = ディストラクタ, -1 = どちらでもない
    if "chosen_item" in df.columns:
        df["chose_target"] = df["chosen_item"].where(df["chosen_item"] >= 0)
        df["task_irrelevant"] = (df["chosen_item"] == -1).astype(int)

    return df


def subject_level_dispersion(
    td: TrialData, value_col: str = "logrt"
) -> pd.DataFrame:
    """参加者ごとの位置・散布度指標を返す。

    rt_cv（＝MW の指標）はここで作る。除外規則が rt_cv をどれだけ
    歪めるかを検証できるよう、複数の散布度指標を並べて出す。
    """
    g = td.df.groupby("subject")[value_col]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean": g.mean(),
            "median": g.median(),
            "sd": g.std(),
            "mad": g.apply(lambda x: (x - x.median()).abs().median() * 1.4826),
            "iqr": g.apply(lambda x: x.quantile(0.75) - x.quantile(0.25)),
        }
    )
    out["cv"] = out["sd"] / out["mean"]
    # 連続試行差の SD: 低周波のドリフトを除いた高周波のばらつき
    out["succ_diff_sd"] = td.df.sort_values(["subject", "trial"]).groupby("subject")[
        value_col
    ].apply(lambda x: np.diff(x.values).std(ddof=1) / np.sqrt(2))
    return out.reset_index()
