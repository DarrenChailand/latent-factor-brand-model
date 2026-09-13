"""Run the deterministic latent-factor analysis from a count matrix."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_analysis.compute_importance_scores import run_importance_from_outdir
from src.data_analysis.compute_pmi import run_compute_pmi, validate_count_matrix
from src.data_analysis.run_svd import run_svd


def run(input_csv: str, output_dir: str, factors: int = 3) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts = validate_count_matrix(pd.read_csv(input_csv, index_col=0))
    counts.to_csv(output / "brand_attribute_counts.csv")
    _, pmi_path = run_compute_pmi(input_csv, output / "ppmi.csv", positive=True)
    _, singular, _ = run_svd(pmi_path, factors, output)
    scores, _ = run_importance_from_outdir(output, factors)

    top = scores.apply(lambda row: row.nlargest(8).index.tolist(), axis=1)
    top.rename("top_attributes").to_csv(output / "top_attributes_summary.csv")
    singular.to_csv(output / "factor_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    singular.plot.bar(x="factor", y="explained_variance_ratio", legend=False, ax=ax)
    ax.set(title="Variance captured by each latent factor", xlabel="", ylabel="Share of squared singular values")
    fig.tight_layout()
    fig.savefig(output / "factor_variance.png", dpi=180)
    plt.close(fig)
    print(f"Analyzed {counts.shape[0]} brands and {counts.shape[1]} attributes.")
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sample/brand_attribute_counts.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--factors", type=int, default=3)
    args = parser.parse_args()
    run(args.input, args.output, args.factors)
