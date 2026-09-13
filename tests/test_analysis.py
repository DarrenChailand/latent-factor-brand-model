import numpy as np
import pandas as pd
import pytest

from src.data_analysis.compute_importance_scores import reconstruct
from src.data_analysis.compute_pmi import compute_pmi, validate_count_matrix
from src.data_analysis.run_svd import decompose


@pytest.fixture
def counts():
    return pd.DataFrame([[4, 1, 0], [0, 2, 5]], index=["A", "B"], columns=["fast", "safe", "cheap"])


def test_ppmi_is_finite_nonnegative_and_shape_preserving(counts):
    result = compute_pmi(counts)
    assert result.shape == counts.shape
    assert np.isfinite(result.to_numpy()).all()
    assert (result >= 0).all().all()


def test_full_rank_svd_reconstructs_matrix(counts):
    matrix = compute_pmi(counts)
    u, s, v = decompose(matrix, k=10)
    restored = reconstruct(u, s["singular_value"], v)
    np.testing.assert_allclose(restored, matrix, atol=1e-10)


def test_negative_counts_are_rejected(counts):
    counts.iloc[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_count_matrix(counts)
