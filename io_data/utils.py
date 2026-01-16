from typing import List, Tuple
import pandas as pd

def combine_subjects(concat_list: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate per-subject concatenated data with a subject column."""
    rows = []
    for subj_id, df in concat_list:
        tmp = df.copy()
        tmp["subject"] = subj_id
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
