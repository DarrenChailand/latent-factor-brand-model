"""Truncated singular value decomposition for a brand-attribute matrix."""
from pathlib import Path
import numpy as np
import pandas as pd

def decompose(matrix: pd.DataFrame, k: int):
    if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Input matrix must be non-empty and finite.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    values = matrix.to_numpy(dtype=float)
    u_full, singular_full, vt_full = np.linalg.svd(values, full_matrices=False)
    rank = min(k, len(singular_full))
    names = [f"factor_{i + 1}" for i in range(rank)]
    u = pd.DataFrame(u_full[:, :rank], index=matrix.index, columns=names)
    v = pd.DataFrame(vt_full[:rank].T, index=matrix.columns, columns=names)
    denominator = np.square(singular_full).sum()
    ratios = np.square(singular_full[:rank]) / denominator if denominator else np.zeros(rank)
    s = pd.DataFrame({"factor": names, "singular_value": singular_full[:rank],
                      "explained_variance_ratio": ratios})
    return u, s, v

def run_svd(input_pmi_csv: str, k: int = 10,
            outdir: str = "data/processed/brand_attribute_matrix"):
    u, s, v = decompose(pd.read_csv(input_pmi_csv, index_col=0), k)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    u.to_csv(destination / "svd_U_brands.csv")
    s.to_csv(destination / "svd_S_singular_values.csv", index=False)
    v.to_csv(destination / "svd_V_attributes.csv")
    return u, s, v

def run_svd_on_pmi(pmi_path="data/processed/brand_attribute_matrix/pmi.csv", k=10,
                   outdir="data/processed/brand_attribute_matrix"):
    return run_svd(pmi_path, k, outdir)
