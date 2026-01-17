#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Brand–Attribute Importance Scores from truncated SVD.

Inputs (produced by run_svd.py):
  - svd_U_brands.csv            (brands × k)
  - svd_S_singular_values.csv   (vector of k singular values)
  - svd_V_attributes.csv        (attributes × k)

Outputs (written to outdir):
  - brand_attribute_importance.csv   (brands × attributes)
  - brand_top_attributes.json        (ranked |importance| list per brand)

Notes:
- This script does not modify SVD logic; it only reconstructs a rank-k approximation
  and ranks attributes per brand by absolute importance.
"""

import json
import os

import numpy as np
import pandas as pd


def compute_importance_scores(
    U_path="data/processed/brand_attribute_matrix/svd_U_brands.csv",
    S_path="data/processed/brand_attribute_matrix/svd_S_singular_values.csv",
    V_path="data/processed/brand_attribute_matrix/svd_V_attributes.csv",
    outdir="data/processed/brand_attribute_matrix",
    k=None,
):
    # -------------------------------------------------------------
    # 1) Load SVD components
    # -------------------------------------------------------------
    U_df = pd.read_csv(U_path, index_col=0)
    V_df = pd.read_csv(V_path, index_col=0)
    S_df = pd.read_csv(S_path)

    brands = U_df.index.to_list()
    attributes = V_df.index.to_list()

    U = U_df.values  # (n × k)
    V = V_df.values  # (m × k)

    singular_values = S_df["singular_value"].values  # length=r
    S = np.diag(singular_values)                     # (r × r)

    # If user specifies k smaller than SVD output → truncate
    r = len(singular_values)
    if k is not None:
        k_eff = min(k, r)
        print(f"Using top {k_eff} latent dimensions")
        U = U[:, :k_eff]
        V = V[:, :k_eff]
        S = np.diag(singular_values[:k_eff])
    else:
        k_eff = r

    # -------------------------------------------------------------
    # 2) Rank-k approximation:
    #        N_approx = U × S × V^T
    # -------------------------------------------------------------
    N_approx = U @ S @ V.T  # (brands × attributes)

    importance_df = pd.DataFrame(N_approx, index=brands, columns=attributes)

    # Save full importance matrix
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "brand_attribute_importance.csv")
    importance_df.to_csv(out_csv)
    print(f"Saved brand–attribute importance matrix → {out_csv}")

    # -------------------------------------------------------------
    # 3) Per-brand ranking by |Importance(i,j)|
    # -------------------------------------------------------------
    ranking_dict = {}

    for brand in brands:
        row = importance_df.loc[brand]
        sorted_attributes = row.abs().sort_values(ascending=False).index.tolist()

        ranking_dict[brand] = {
            "top_attributes": sorted_attributes,
            "raw_scores": row.to_dict(),
        }

    out_json = os.path.join(outdir, "brand_top_attributes.json")
    with open(out_json, "w") as f:
        json.dump(ranking_dict, f, indent=2)

    print(f"Saved per-brand attribute rankings → {out_json}")

    return importance_df, ranking_dict


def run_importance_from_outdir(input_outdir: str, k: int = None):
    """
    Notebook-friendly wrapper.

    Assumes the SVD outputs live inside input_outdir:
      - svd_U_brands.csv
      - svd_S_singular_values.csv
      - svd_V_attributes.csv

    Returns (importance_df, ranking_dict).
    """
    U_path = os.path.join(input_outdir, "svd_U_brands.csv")
    S_path = os.path.join(input_outdir, "svd_S_singular_values.csv")
    V_path = os.path.join(input_outdir, "svd_V_attributes.csv")

    return compute_importance_scores(
        U_path=U_path,
        S_path=S_path,
        V_path=V_path,
        outdir=input_outdir,
        k=k,
    )
