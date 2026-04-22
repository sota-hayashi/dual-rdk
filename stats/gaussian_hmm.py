"""
Gaussian HMM による隠れ状態推定（engaged / disengaged）

仕様: gaussian_hmm_spec.md
アルゴリズム: Ashwood et al. 2022, Algorithm 1 に基づく2段階fitting

観測変数（input_type で切り替え）:
    "angular_error"  : abs_angular_error（絶対角度誤差、0°〜180°）
    "reaction_time"  : log(RT)（反応時間の対数、歪み抑制のため）
状態:
    0 = engaged    (観測値が小さい)
    1 = disengaged (観測値が大きい)
"""

from typing import Dict, List, Optional, Tuple
import json
import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm


# ─────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────

def _compute_abs_angular_error(df: pd.DataFrame) -> np.ndarray:
    """
    abs_angular_error を計算して返す。

    優先順位:
      1. "angular_error" 列が存在する場合 → abs(angular_error)
      2. "angular_error_target" と "angular_error_distractor" が存在する場合
         → |target_AE| < |distractor_AE| なら target_AE、さもなくば distractor_AE
         の絶対値

    Returns
    -------
    np.ndarray : shape (n_valid_rows,), float
    """
    if "rt" in df.columns:
        df.dropna(subset=["rt"], inplace=True)
    if "angular_error" in df.columns:
        return df["angular_error"].abs().to_numpy(dtype=float)

    if "angular_error_target" in df.columns and "angular_error_distractor" in df.columns:
        ae_t = df["angular_error_target"].to_numpy(dtype=float)
        ae_d = df["angular_error_distractor"].to_numpy(dtype=float)
        angular_error = np.where(np.abs(ae_t) < np.abs(ae_d), ae_t, ae_d)
        return np.abs(angular_error)

    raise ValueError(
        "DataFrame は 'angular_error' 列、または "
        "'angular_error_target' と 'angular_error_distractor' 列を含む必要があります。"
    )


def _compute_log_reaction_time(df: pd.DataFrame) -> np.ndarray:
    """
    log(RT) を計算して返す。RTの歪みを抑えるために自然対数を適用する。

    優先順位:
      1. "rt" 列が存在する場合
      2. "reaction_time" 列が存在する場合

    Returns
    -------
    np.ndarray : shape (n_valid_rows,), float
    """
    if "rt" in df.columns:
        return np.log(df["rt"].to_numpy(dtype=float))
    if "reaction_time" in df.columns:
        return np.log(df["reaction_time"].to_numpy(dtype=float))
    raise ValueError(
        "DataFrame は 'rt' 列または 'reaction_time' 列を含む必要があります。"
    )


def _drop_na_rows(df: pd.DataFrame, input_type: str = "angular_error") -> pd.DataFrame:
    """入力種別に応じた関連列の NaN 行を除去して返す。"""
    if input_type == "reaction_time":
        if "rt" in df.columns:
            return df.dropna(subset=["rt"]).copy()
        if "reaction_time" in df.columns:
            return df.dropna(subset=["reaction_time"]).copy()
        raise ValueError("rt 関連列が見当たりません。")
    # angular_error
    if "angular_error" in df.columns:
        return df.dropna(subset=["angular_error"]).copy()
    if "angular_error_target" in df.columns and "angular_error_distractor" in df.columns:
        return df.dropna(subset=["angular_error_target", "angular_error_distractor"]).copy()
    raise ValueError("angular_error 関連列が見当たりません。")


# ─────────────────────────────────────────────
# 状態ラベルのアライメント
# ─────────────────────────────────────────────

def align_states(means: np.ndarray) -> Dict[int, int]:
    """
    グローバルモデルのμに基づいて状態ラベルを割り当てる。

    観測値が小さいほど engaged とみなすため（angular_error・log(RT) ともに共通）:
      μが小さい状態 → engaged    (統一番号 0)
      μが大きい状態 → disengaged (統一番号 1)

    Parameters
    ----------
    means : np.ndarray, shape (2, 1)

    Returns
    -------
    dict : {元の状態番号: 統一後の状態番号}
           例: {1: 0, 0: 1}  （元の状態1がengaged、元の状態0がdisengaged）
    """
    means_flat = means.flatten()
    engaged_idx = int(np.argmin(means_flat))
    disengaged_idx = int(np.argmax(means_flat))
    return {engaged_idx: 0, disengaged_idx: 1}


# ─────────────────────────────────────────────
# グローバルfitting
# ─────────────────────────────────────────────

def _fit_global_model(
    X_all: np.ndarray,
    lengths: List[int],
    n_init: int = 20,
    n_iter: int = 200,
    tol: float = 1e-4,
) -> hmm.GaussianHMM:
    """
    全参加者データを結合してグローバル GaussianHMM をfitする。

    n_init 回の異なるランダムシードで試行し、
    最大対数尤度のモデルをグローバルモデルとして返す。

    Parameters
    ----------
    X_all : np.ndarray, shape (total_trials, 1)
    lengths : List[int]  参加者ごとの試行数（hmmlearn の lengths 引数）
    n_init : int         初期値の試行回数
    n_iter : int         EMの最大反復回数
    tol : float          収束判定の閾値

    Returns
    -------
    hmm.GaussianHMM : 最大対数尤度のグローバルモデル
    """
    best_score = -np.inf
    best_model: Optional[hmm.GaussianHMM] = None

    for seed in range(n_init):
        model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_all, lengths=lengths)

        score = model.score(X_all, lengths=lengths)
        if score > best_score:
            best_score = score
            best_model = model

    print(f"Transition matrix of best global model:\n{best_model.transmat_}")
    print(f"Means of best global model:\n{best_model.means_}")
    print(f"Covariances of best global model:\n{best_model.covars_}")
    print(f"Log-likelihood of best global model: {best_model.score(X_all, lengths=lengths)}")
    return best_model  # type: ignore[return-value]


# ─────────────────────────────────────────────
# 個人fitting
# ─────────────────────────────────────────────

def _fit_individual_model(
    X_i: np.ndarray,
    global_model: hmm.GaussianHMM,
    n_iter: int = 200,
    tol: float = 1e-4,
) -> hmm.GaussianHMM:
    """
    グローバルモデルを初期値として、個人データで GaussianHMM をfitする。

    Parameters
    ----------
    X_i : np.ndarray, shape (n_trials_i, 1)
    global_model : 初期値に使うグローバルモデル
    n_iter : int   EMの最大反復回数
    tol : float    収束判定の閾値

    Returns
    -------
    hmm.GaussianHMM : 個人モデル
    """
    individual_model = hmm.GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=n_iter,
        tol=tol,
        init_params="",    # 自動初期化を無効化（グローバル初期値を使うため）
        params="stmc",     # 学習対象: startprob, transmat, means, covars
    )
    individual_model.startprob_ = global_model.startprob_.copy()
    individual_model.transmat_ = global_model.transmat_.copy()
    individual_model.means_ = global_model.means_.copy()
    individual_model.covars_ = global_model.covars_.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        individual_model.fit(X_i)

    return individual_model


# ─────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────

def save_gaussian_hmm_results(results: List[Dict], save_path: str) -> None:
    """
    結果リストを CSV に保存する。行列はJSON文字列として格納。

    Parameters
    ----------
    results : List[Dict]  run_gaussian_hmm() の戻り値
    save_path : str       保存先パス
    """
    rows = []
    for r in results:
        rows.append({
            "participant_id": r["participant_id"],
            "n_trials": r["n_trials"],
            "viterbi_states": json.dumps(r["viterbi_states"].tolist()),
            "means": json.dumps(r["means"].tolist()),
            "covars": json.dumps(r["covars"].tolist()),
            "transmat": json.dumps(r["transmat"].tolist()),
            "state_labels": json.dumps(r["state_labels"]),
            "log_likelihood": r["log_likelihood"],
        })
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"Gaussian HMM results saved to {save_path}")


# ─────────────────────────────────────────────
# メインエントリーポイント
# ─────────────────────────────────────────────

def run_gaussian_hmm(
    concat_list: List[Tuple[str, pd.DataFrame]],
    n_init: int = 20,
    n_iter: int = 200,
    tol: float = 1e-4,
    save_path: Optional[str] = None,
    input_type: str = "angular_error",
) -> List[Dict]:
    """
    Gaussian HMM を全参加者に対して2段階fittingで実行する。

    アルゴリズム（仕様書 §4 全体フロー）:
      Step 1: データ準備（観測変数の計算、shapes 整形）
      Step 2: グローバルfitting（n_init 回の初期値で最大対数尤度のモデルを選択）
      Step 3: 状態ラベルのアライメント（μ小→engaged=0、μ大→disengaged=1）
      Step 4: 個人fitting（グローバルパラメータを初期値として各参加者でEM）
      Step 5: ビタビ状態系列の推定とアライメント適用

    Parameters
    ----------
    concat_list : List[Tuple[str, pd.DataFrame]]
        (participant_id, df) のリスト。
        input_type="angular_error" の場合、各 df は以下のいずれかを含む必要がある:
          - "angular_error" 列（符号付き、-180°〜180°）
          - "angular_error_target" と "angular_error_distractor" 列
        input_type="reaction_time" の場合、各 df は以下のいずれかを含む必要がある:
          - "rt" 列
          - "reaction_time" 列
    n_init : int
        グローバルfittingの初期値試行回数（デフォルト: 20）
    n_iter : int
        EMの最大反復回数（デフォルト: 200）
    tol : float
        収束判定の閾値（デフォルト: 1e-4）
    save_path : str or None
        結果をCSVに保存するパス。None の場合は保存しない。
        文字列中に ``{input_type}`` を含む場合、実際の input_type 値で置換される。
        例: "results/gaussian_hmm_{input_type}.csv"
    input_type : str
        観測変数の種類。"angular_error"（デフォルト）または "reaction_time"。

    Returns
    -------
    list[dict]
        参加者ごとの結果辞書のリスト。各辞書のキー:
            participant_id  : str or int
            n_trials        : int               有効試行数
            viterbi_states  : np.ndarray (n_trials,)  0=engaged, 1=disengaged
            means           : np.ndarray (2, 1)       aligned
            covars          : np.ndarray (2, 1, 1)    aligned
            transmat        : np.ndarray (2, 2)       aligned
            state_labels    : dict   {0: "engaged", 1: "disengaged"}
            log_likelihood  : float
    """
    _VALID_INPUT_TYPES = {"angular_error", "reaction_time"}
    if input_type not in _VALID_INPUT_TYPES:
        raise ValueError(f"input_type は {_VALID_INPUT_TYPES} のいずれかである必要があります。got: {input_type!r}")

    # save_path 内の {input_type} プレースホルダーを置換
    resolved_save_path = save_path.format(input_type=input_type) if save_path is not None else None

    # ─── Step 1: データ準備 ───
    subject_data: List[Tuple[str, np.ndarray, int]] = []
    X_all_parts: List[np.ndarray] = []
    lengths: List[int] = []

    for participant_id, df in concat_list:
        try:
            work = _drop_na_rows(df, input_type=input_type)
        except ValueError as e:
            print(f"Skipping {participant_id}: {e}")
            continue

        if input_type == "reaction_time":
            obs = _compute_log_reaction_time(work)
        else:
            obs = _compute_abs_angular_error(work)
        n_trials = len(obs)

        if n_trials < 5:
            print(f"Skipping {participant_id}: only {n_trials} valid trials (minimum 5 required).")
            continue

        X_i = obs.reshape(-1, 1)
        subject_data.append((participant_id, X_i, n_trials))
        X_all_parts.append(X_i)
        lengths.append(n_trials)

    if not subject_data:
        raise ValueError("有効な参加者データが存在しません。")

    X_all = np.concatenate(X_all_parts, axis=0)

    # ─── Step 2: グローバルfitting ───
    n_subjects = len(subject_data)
    total_trials = len(X_all)
    _unit = "log(RT)" if input_type == "reaction_time" else "°"
    print(
        f"[Gaussian HMM({input_type})] Global fitting: {n_subjects} subjects, "
        f"{total_trials} total trials, {n_init} initializations..."
    )
    global_model = _fit_global_model(X_all, lengths, n_init=n_init, n_iter=n_iter, tol=tol)
    global_score = global_model.score(X_all, lengths=lengths)
    print(
        f"[Gaussian HMM({input_type})] Global model: "
        f"μ0={global_model.means_[0, 0]:.3f}{_unit}, "
        f"μ1={global_model.means_[1, 0]:.3f}{_unit}, "
        f"loglik={global_score:.2f}"
    )

    # ─── Step 3: 状態ラベルのアライメント ───
    alignment_map = align_states(global_model.means_)
    # alignment_map の値の昇順（0→engaged, 1→disengaged）に対応する元のインデックス
    idx_order = [k for k, _v in sorted(alignment_map.items(), key=lambda x: x[1])]

    # ─── Steps 4 & 5: 個人fitting + 状態系列推定 ───
    results: List[Dict] = []

    for participant_id, X_i, n_trials in subject_data:
        try:
            individual_model = _fit_individual_model(X_i, global_model, n_iter=n_iter, tol=tol)

            # ビタビアルゴリズムで最尤状態系列を取得
            viterbi_states_raw = individual_model.predict(X_i)

            # アライメントで状態番号を統一（0=engaged, 1=disengaged）
            viterbi_states = np.vectorize(alignment_map.get)(viterbi_states_raw)

            # パラメータをアライメント後の順序に並び替え
            means_aligned = individual_model.means_[idx_order]          # (2, 1)
            covars_aligned = individual_model.covars_[idx_order]         # (2, 1, 1)
            transmat_aligned = individual_model.transmat_[
                np.ix_(idx_order, idx_order)
            ]                                                            # (2, 2)

            log_likelihood = float(individual_model.score(X_i))

            results.append({
                "participant_id": participant_id,
                "n_trials": n_trials,
                "viterbi_states": viterbi_states,
                "means": means_aligned,
                "covars": covars_aligned,
                "transmat": transmat_aligned,
                "state_labels": {0: "engaged", 1: "disengaged"},
                "log_likelihood": log_likelihood,
            })

            print(
                f"  {participant_id}: "
                f"μ_engaged={means_aligned[0, 0]:.3f}{_unit}, "
                f"μ_disengaged={means_aligned[1, 0]:.3f}{_unit}, "
                f"loglik={log_likelihood:.2f}"
            )

        except Exception as e:
            print(f"  Skipping individual fitting for {participant_id}: {e}")
            continue

    print(f"[Gaussian HMM({input_type})] Done: {len(results)}/{n_subjects} subjects fitted.")

    if resolved_save_path is not None:
        save_gaussian_hmm_results(results, resolved_save_path)

    return results
