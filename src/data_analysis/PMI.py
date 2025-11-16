import pandas as pd
import numpy as np
import math

# ============================
# Load brand × attribute matrix
# ============================
path = "data/processed/brand_attribute_matrix/raw.csv"

df = pd.read_csv(path, index_col=0)   # first column is brand names
F = df.astype(float)                  # frequency matrix

# ============================
# Total co-occurrences
# ============================
total = F.values.sum()

# ============================
# Compute probabilities
# ============================
# Joint probability P(i,j)
Pij = F / total

# Brand marginals  P(i)
Pi = F.sum(axis=1) / total      # sum across attributes

# Attribute marginals P(j)
Pj = F.sum(axis=0) / total      # sum across brands

# ============================
# PMI(i,j) = log( P(i,j) / (P(i)*P(j)) )
# ============================
PMI = pd.DataFrame(index=F.index, columns=F.columns, dtype=float)

for brand in F.index:
    for attr in F.columns:
        if F.loc[brand, attr] == 0:
            PMI.loc[brand, attr] = 0.0   # define PMI=0 for zero counts
        else:
            numerator = Pij.loc[brand, attr]
            denominator = Pi.loc[brand] * Pj.loc[attr]
            PMI.loc[brand, attr] = math.log(numerator / denominator)

# ============================
# Save output
# ============================
PMI.to_csv("data/processed/brand_attribute_matrix/pmi.csv")

print("PMI matrix saved → data/processed/brand_attribute_matrix/pmi.csv")
PMI.head()
