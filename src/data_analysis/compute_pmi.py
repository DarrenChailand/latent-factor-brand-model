import math

import pandas as pd


def run_compute_pmi(
    input_csv: str,
    output_csv: str = "data/processed/brand_attribute_matrix/pmi.csv",
):
    """
    Compute a PMI (Pointwise Mutual Information) matrix from a brand×attribute
    frequency matrix.

    Parameters
    ----------
    input_csv : str
        Path to a CSV where rows = brands, columns = attributes, values = counts.
        The first column should be brand names (index column).
    output_csv : str
        Path to write the PMI matrix CSV.

    Returns
    -------
    (PMI_df, output_csv) : (pd.DataFrame, str)
        PMI_df has the same shape/index/columns as the input matrix.
        Cells with zero count are assigned PMI = 0.0 (by definition here).
    """
    # Load
    df = pd.read_csv(input_csv, index_col=0)
    F = df.astype(float)

    # Total
    total = F.values.sum()

    # Probabilities
    Pij = F / total
    Pi = F.sum(axis=1) / total
    Pj = F.sum(axis=0) / total

    # PMI
    PMI = pd.DataFrame(index=F.index, columns=F.columns, dtype=float)
    for brand in F.index:
        for attr in F.columns:
            if F.loc[brand, attr] == 0:
                PMI.loc[brand, attr] = 0.0
            else:
                numerator = Pij.loc[brand, attr]
                denominator = Pi.loc[brand] * Pj.loc[attr]
                PMI.loc[brand, attr] = math.log(numerator / denominator)

    # Save
    PMI.to_csv(output_csv)
    print(f"PMI matrix saved → {output_csv}")
    return PMI, output_csv
