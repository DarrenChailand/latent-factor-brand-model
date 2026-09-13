"""Pointwise mutual information for brand-by-attribute count matrices."""
from pathlib import Path
import numpy as np
import pandas as pd

def validate_count_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The count matrix is empty.")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Brand names and attribute names must be unique.")
    numeric = frame.apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("The count matrix contains missing or infinite values.")
    if (numeric < 0).any().any():
        raise ValueError("Counts cannot be negative.")
    if numeric.to_numpy().sum() <= 0:
        raise ValueError("The count matrix must contain at least one positive count.")
    return numeric

def compute_pmi(counts: pd.DataFrame, positive: bool = True) -> pd.DataFrame:
    """Compute PMI; unseen pairs are zero and negative values can be clipped."""
    counts = validate_count_matrix(counts)
    total = counts.to_numpy().sum()
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    observed = counts.to_numpy()
    values = np.zeros_like(observed, dtype=float)
    mask = observed > 0
    values[mask] = np.log(observed[mask] / expected[mask])
    if positive:
        values = np.maximum(values, 0.0)
    return pd.DataFrame(values, index=counts.index, columns=counts.columns)

def run_compute_pmi(input_csv: str, output_csv: str, positive: bool = True):
    result = compute_pmi(pd.read_csv(input_csv, index_col=0), positive=positive)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination)
    return result, str(destination)
