import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load data
data = pd.read_excel("correlation matrix.xlsx")

# Compute correlation matrix and p-values
def correlation_with_pvals(df):
    corr_matrix = df.corr(method="pearson").round(2)
    pval_matrix = np.zeros_like(corr_matrix)
    
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if i != j:
                r, p = pearsonr(df.iloc[:, i], df.iloc[:, j])
                pval_matrix[i, j] = p
            else:
                pval_matrix[i, j] = np.nan  # Diagonal remains NaN
    
    return corr_matrix, pval_matrix

my_matrix, p_values = correlation_with_pvals(data)

# Significance notation function
def significance(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

# Create annotation text with correlation and significance
annot_matrix = my_matrix.astype(str)
for i in range(len(my_matrix)):
    for j in range(len(my_matrix)):
        if i != j:
            annot_matrix.iloc[i, j] = f"{my_matrix.iloc[i, j]}\n{significance(p_values[i, j])}"
        else:
            annot_matrix.iloc[i, j] = ""

# Mask upper triangle
my_mask = np.triu(np.ones_like(my_matrix, dtype=bool))

# Set figure size (Adjust width & height)
fig, ax = plt.subplots(figsize=(12, 10))

# Plot heatmap
sns.heatmap(my_matrix, cmap="Reds", vmin=0, vmax=1,
            annot=annot_matrix, fmt="", square=True, mask=my_mask, 
            linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})

# Add legend at the bottom
legend_labels = ["ns (p ≥ 0.05)", "* (p < 0.05)", "** (p < 0.01)", "*** (p < 0.001)"]
legend_patches = [plt.Line2D([1.5], [1.5], linestyle="none", marker="s", markersize=10, color="black") for _ in legend_labels]

plt.legend(legend_patches, legend_labels, loc="upper center",
           bbox_to_anchor=(0.5, -0.20), ncol=4, frameon=False, fontsize=12)

plt.title("Correlation Matrix with Significance Levels", fontsize=14, fontweight="bold")
plt.show()
