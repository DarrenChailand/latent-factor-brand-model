"""Reconstruct interpretable brand-attribute scores from saved SVD factors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def reconstruct(u: pd.DataFrame, singular_values: pd.Series, v: pd.DataFrame, k=None):
    if list(u.columns) != list(v.columns):
        raise ValueError("U and V factor columns do not match.")
    rank = len(singular_values) if k is None else min(k, len(singular_values))
    if rank < 1:
        raise ValueError("At least one factor is required.")
    values = (u.iloc[:, :rank].to_numpy() @ np.diag(singular_values.iloc[:rank])
              @ v.iloc[:, :rank].to_numpy().T)
    return pd.DataFrame(values, index=u.index, columns=v.index)

def compute_importance_scores(U_path, S_path, V_path,
                              outdir="data/processed/brand_attribute_matrix", k=None):
    u = pd.read_csv(U_path, index_col=0)
    s = pd.read_csv(S_path)["singular_value"]
    v = pd.read_csv(V_path, index_col=0)
    scores = reconstruct(u, s, v, k=k)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    scores.to_csv(destination / "brand_attribute_importance.csv")
    rankings = {
        brand: [{"attribute": attribute, "score": float(scores.loc[brand, attribute])}
                for attribute in scores.loc[brand].sort_values(ascending=False).index]
        for brand in scores.index
    }
    with (destination / "brand_top_attributes.json").open("w", encoding="utf-8") as handle:
        json.dump(rankings, handle, indent=2)
    return scores, rankings

def run_importance_from_outdir(input_outdir: str, k=None):
    root = Path(input_outdir)
    return compute_importance_scores(root / "svd_U_brands.csv",
                                     root / "svd_S_singular_values.csv",
                                     root / "svd_V_attributes.csv", root, k)
