from __future__ import annotations

import numpy as np
import pandas as pd


def gse_to_pseudo_rtn(vec_gse: np.ndarray) -> np.ndarray:
    """
    Convert a 3-vector in GSE to a simple, common 'pseudo-RTN':
      R = -X_GSE (anti-sunward),
      T = -Y_GSE,
      N = +Z_GSE.
    """
    out = np.empty_like(vec_gse, dtype=float)
    out[:, 0] = -vec_gse[:, 0]
    out[:, 1] = -vec_gse[:, 1]
    out[:, 2] =  vec_gse[:, 2]
    return out


def ensure_b_rtn(df: pd.DataFrame, b_var: str, prefix: str) -> pd.DataFrame:
    """
    Standardize magnetic field into columns:
      {prefix}_Br, {prefix}_Bt, {prefix}_Bn, {prefix}_Bmag
    `b_var` should refer to a column that is either:
      - a 3-component vector stored as object/array-like, or
      - three separate scalar columns handled elsewhere.
    """
    out = df.copy()
    if b_var not in out.columns:
        return out

    vals = out[b_var].to_numpy()
    # cdasws often stores vectors as arrays in each row; coerce to (N,3)
    vec = np.vstack(vals).astype(float)
    rtn = gse_to_pseudo_rtn(vec)

    out[f"{prefix}_Br"] = rtn[:, 0]
    out[f"{prefix}_Bt"] = rtn[:, 1]
    out[f"{prefix}_Bn"] = rtn[:, 2]
    out[f"{prefix}_Bmag"] = np.sqrt(np.sum(rtn * rtn, axis=1))
    return out


def ensure_v_rtn(df: pd.DataFrame, v_var: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    if v_var not in out.columns:
        return out

    vec = np.vstack(out[v_var].to_numpy()).astype(float)
    rtn = gse_to_pseudo_rtn(vec)

    out[f"{prefix}_Vr"] = rtn[:, 0]
    out[f"{prefix}_Vt"] = rtn[:, 1]
    out[f"{prefix}_Vn"] = rtn[:, 2]
    out[f"{prefix}_Vmag"] = np.sqrt(np.sum(rtn * rtn, axis=1))
    return out
