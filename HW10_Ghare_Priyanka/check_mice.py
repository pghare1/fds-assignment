import numpy as np
import pandas as pd
from scipy.stats import chi2

# Read in the data
mice_df = pd.read_csv("mice.csv")

# Figure out the possible gene types
gene_types = list(mice_df.gene_type.unique())
print(f"Possible Gene Types:{gene_types}")

## Your code here
contingency_matrix = pd.crosstab(
    mice_df.gene_type,
    mice_df.has_cancer,
    margins=True,
    margins_name="Total",
    rownames=["Gene"],
    colnames=["has_cancer"],
)
contingency_matrix.columns = ["No Cancer", "Has Cancer", ""]
print(f"\nContingency table:")
print(f"{contingency_matrix}")

condi_proportion = pd.crosstab(
    mice_df.gene_type,
    mice_df.has_cancer,
    margins=True,
    margins_name="Total",
    rownames=["Gene"],
    colnames=[""],
    normalize="index",
)
row_marginal_percent = contingency_matrix.iloc[0:3, 2] / contingency_matrix.iloc[3, 2]
condi_proportion["row_marginal_percent"] = row_marginal_percent
condi_proportion.columns = ["No Cancer", "Has Cancer", "row_marginal_per"]
condi_proportion = condi_proportion.fillna(1)
print(f"\nConditional proportions table:")
print((condi_proportion.apply(lambda x: x * 100)).round(1).astype(str) + "%")


expected_counts = (
    contingency_matrix.sum(axis=1)
    .to_frame()
    .dot(contingency_matrix.sum(axis=0).to_frame().T)
    / contingency_matrix.sum().sum()
)

# remove the margins from the expected_counts table
expected_counts = expected_counts.iloc[0:3, 0:2]
expected_counts.columns = ["No Cancer", "Has Cancer"]
expected_counts["Total"] = contingency_matrix.iloc[0:3, 2]

# add row totals to the expected counts table
expected_counts.loc["Total"] = (
    condi_proportion.iloc[3, 0:2].apply(lambda x: x * 100).round(1).astype(str) + "%"
)
expected_counts = expected_counts.fillna("")

print(f"\nExpected counts table:")
print(round(expected_counts, 1))

chisquare = (
    (
        (contingency_matrix.iloc[0:3, 0:2] - expected_counts.iloc[0:3, 0:2]) ** 2
        / expected_counts.iloc[0:3, 0:2]
    )
    .sum()
    .sum()
)
print(f"\n X^2 = {chisquare:.2f}")

deg_freedom = (contingency_matrix.iloc[0:3, 0:2].shape[0] - 1) * (
    contingency_matrix.iloc[0:3, 0:2].shape[1] - 1
)
print(f"Degree of freedom = {deg_freedom}")

p_val = 1 - chi2.cdf(chisquare, deg_freedom)
print(f"P-value = {p_val:.4f}")
