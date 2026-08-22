"""結果の整形・保存。

コンソールには読みやすい要約を出し、同じ内容を outputs/mixed/<name>/ に
CSV と JSON で落とす。JSON には実行時の設定も入れるので、後から
「どの除外規則・どの推定法で出した数字か」を追える。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dualrdk.mixed.results import MixedResult

_W = 78


def rule(char: str = "=") -> str:
    return char * _W


def header(title: str, char: str = "=") -> str:
    return f"\n{rule(char)}\n{title}\n{rule(char)}"


def fmt_fixed(res: MixedResult) -> str:
    """固定効果の表。"""
    lines = [
        f"  {'term':<14}{'Estimate':>12}{'SE':>11}{'z':>9}{'p':>11}{'95% CI':>28}"
    ]
    for t in res.fixed.index:
        r = res.fixed.loc[t]
        p = f"{r['p']:.3e}" if r["p"] < 1e-3 else f"{r['p']:.4f}"
        ci = f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]"
        lines.append(
            f"  {t:<14}{r['estimate']:>12.6f}{r['se']:>11.6f}{r['z']:>9.3f}{p:>11}{ci:>28}"
        )
    return "\n".join(lines)


def fmt_varcomp(res: MixedResult) -> str:
    """ランダム効果と残差の表。"""
    lines = [f"  {'component':<28}{'Variance':>14}{'SD':>12}"]
    for c in res.varcomp.index:
        r = res.varcomp.loc[c]
        label = c if c == "Residual" else f"{c} (subject)"
        lines.append(f"  {label:<28}{r['variance']:>14.8f}{r['sd']:>12.6f}")
    for k, v in res.corr.items():
        a, b = k.split("~")
        lines.append(f"  {f'Corr({a}, {b})':<28}{v:>14.3f}{'':>12}")
    return "\n".join(lines)


def fmt_convergence(res: MixedResult) -> str:
    if res.convergence_check is None:
        return "  （未検証）"
    cc = res.convergence_check
    lines = [f"  {'method':<9}{'conv':>6}{'logLik':>14}{'dLogLik':>11}{'agrees':>8}"]
    for _, r in cc.iterrows():
        ll = f"{r['logLik']:.4f}" if pd.notna(r.get("logLik")) else "FAILED"
        dl = f"{r['dLogLik']:+.4f}" if pd.notna(r.get("dLogLik")) else "-"
        lines.append(
            f"  {r['method']:<9}{str(r['converged']):>6}{ll:>14}{dl:>11}"
            f"{str(r.get('agrees', '')):>8}"
        )
    bad = cc[~cc.get("agrees", pd.Series(True, index=cc.index)).fillna(False)]
    if len(bad):
        lines.append(
            f"  ! {', '.join(bad['method'])} は別の解に収束。"
            f"optimizer='{res.optimizer}' の結果を採用"
        )
    return "\n".join(lines)


def fmt_bootstrap(ci: pd.DataFrame) -> str:
    lines = [f"  {'parameter':<22}{'point':>12}{'boot median':>14}{'95% CI':>28}{'≠0':>5}"]
    for _, r in ci.iterrows():
        pt = f"{r['point']:+.6f}" if pd.notna(r["point"]) else "-"
        cis = f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]"
        lines.append(
            f"  {r['parameter']:<22}{pt:>12}{r['boot_median']:>+14.6f}{cis:>28}"
            f"{'✓' if r['excludes_zero'] else '':>5}"
        )
    return "\n".join(lines)


def fmt_dict(d: dict, indent: str = "  ") -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, float):
            s = f"{v:.4g}"
        elif isinstance(v, list):
            s = str(v)
        else:
            s = str(v)
        lines.append(f"{indent}{k:<28}{s}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


def save_tables(out_dir: Path, tables: Dict[str, pd.DataFrame]) -> List[Path]:
    paths = []
    for name, df in tables.items():
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        p = out_dir / "tables" / f"{name}.csv"
        df.to_csv(p, index=df.index.name is not None or not isinstance(df.index, pd.RangeIndex))
        paths.append(p)
    return paths


def save_json(out_dir: Path, name: str, payload: dict) -> Path:
    p = out_dir / f"{name}.json"
    payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), **payload}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable))
    return p


def save_text(out_dir: Path, name: str, text: str) -> Path:
    p = out_dir / f"{name}.txt"
    p.write_text(text)
    return p
