#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SVD on the PMI-based Brand × Attribute matrix.

Inputs:
  - pmi.csv (brands × attributes)

Outputs (written to outdir):
  - svd_U_brands.csv
  - svd_S_singular_values.csv
  - svd_V_attributes.csv

U: brands × k factor loadings
S: k singular values (+ explained variance ratio)
V: attributes × k factor loadings
"""

import os

import numpy as np
import pandas as pd


def run_svd_on_pmi(
    pmi_path: str = "data/processed/brand_attribute_matrix/pmi.csv",
    k: int = 10,
    outdir: str = "data/processed/brand_attribute_matrix",
):
    # 1) Load PMI matrix N (brands × attributes)
    N_df = pd.read_csv(pmi_path, index_col=0)
    brands = N_df.index.to_list()
    attributes = N_df.columns.to_list()

    N = N_df.values
    n, m = N.shape
    print(f"Loaded PMI matrix from {pmi_path} with shape {N.shape} (brands × attributes).")

    # 2) Full SVD: N = U Σ V^T  (full_matrices=False -> r = min(n, m))
    U_full, s_full, Vt_full = np.linalg.svd(N, full_matrices=False)

    # 3) Truncate to top k
    r = s_full.shape[0]
    k_eff = min(k, r)
    if k_eff < k:
        print(f"Requested k={k} but rank is only {r}. Using k={k_eff}.")

    U_k = U_full[:, :k_eff]         # (n × k)
    s_k = s_full[:k_eff]            # (k,)
    S_k = np.diag(s_k)              # (k × k)
    V_k = Vt_full[:k_eff, :].T      # (m × k)

    # 4) Explained variance ratio (∝ singular_value^2)
    total_var = np.sum(s_full ** 2)
    explained_var = s_k ** 2
    explained_ratio = explained_var / total_var

    print("\nTop singular values and explained variance ratio:")
    for i in range(k_eff):
        print(
            f"  Dim {i+1}: singular value = {s_k[i]:.4f}, "
            f"explained variance ratio = {explained_ratio[i]:.4%}"
        )

    # 5) Wrap into DataFrames
    factor_names = [f"factor_{i+1}" for i in range(k_eff)]
    U_brands_df = pd.DataFrame(U_k, index=brands, columns=factor_names)
    V_attributes_df = pd.DataFrame(V_k, index=attributes, columns=factor_names)

    S_vector_df = pd.DataFrame(
        {
            "factor": factor_names,
            "singular_value": s_k,
            "explained_variance_ratio": explained_ratio,
        }
    )

    # 6) Save outputs
    os.makedirs(outdir, exist_ok=True)
    U_path = os.path.join(outdir, "svd_U_brands.csv")
    S_path = os.path.join(outdir, "svd_S_singular_values.csv")
    V_path = os.path.join(outdir, "svd_V_attributes.csv")

    U_brands_df.to_csv(U_path)
    S_vector_df.to_csv(S_path, index=False)
    V_attributes_df.to_csv(V_path)

    print(f"\nSaved U (brand-factor) to {U_path}")
    print(f"Saved S (singular values) to {S_path}")
    print(f"Saved V (attribute-factor) to {V_path}")

    return U_brands_df, S_vector_df, V_attributes_df


def run_svd(
    input_pmi_csv: str,
    k: int = 10,
    outdir: str = "data/processed/brand_attribute_matrix",
):
    """
    Notebook-friendly wrapper around run_svd_on_pmi().
    Returns (U_df, S_df, V_df).
    """
    return run_svd_on_pmi(pmi_path=input_pmi_csv, k=k, outdir=outdir)
