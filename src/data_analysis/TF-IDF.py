import numpy as np
import pandas as pd

def compute_tf(F):
    """
    Compute TF (row-normalized frequencies)
    """
    F = F.astype(float)
    row_sum = F.sum(axis=1).replace(0, 1)
    return F.div(row_sum, axis=0)


def compute_tfidf(F):
    """
    Compute TF-IDF using:
        TF[i,j] = F[i,j] / sum_k F[i,k]
        DF[j]   = number of brands with F[i,j] > 0
        IDF[j]  = log(n / DF[j])
        M[i,j]  = TF[i,j] * IDF[j]
    """
    F = F.astype(float)
    n = F.shape[0]

    # TF
    TF = compute_tf(F)

    # DF and IDF
    DF = (F > 0).sum(axis=0)
    IDF = np.log(n / DF)

    # TF-IDF
    TFIDF = TF.mul(IDF, axis=1)

    return TF, TFIDF, IDF


if __name__ == "__main__":
    # Load raw csv (your input)
    F = pd.read_csv("data/processed/brand_attribute_matrix/raw.csv", index_col=0)

    TF, TFIDF, IDF = compute_tfidf(F)

    # Print results
    print("\n=== TF Matrix ===")
    print(TF.round(6))

    print("\n=== IDF Values ===")
    print(IDF.round(6))

    print("\n=== TF-IDF Matrix ===")
    print(TFIDF.round(6))

    # Save
    TF.to_csv("data/processed/brand_attribute_matrix/tf.csv")
    TFIDF.to_csv("data/processed/brand_attribute_matrix/tfidf.csv")
    IDF.to_csv("data/processed/brand_attribute_matrix/idf.csv")
