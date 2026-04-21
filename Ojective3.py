
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="white")

df = pd.read_csv("cleaned_ev_data.csv")

# Correlation
corr = df.corr(numeric_only=True)
print("Correlation:\n", corr)
plt.figure(figsize=(9,6))
sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",   
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 9}
)
plt.title("Correlation Heatmap", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# Distribution
plt.figure(figsize=(8,5))

sns.histplot(
    df['Electric Range'].dropna(),
    bins=40,
    kde=True,
    color="steelblue",
    edgecolor="black"
)
plt.title("Electric Range Distribution", fontsize=13)
plt.xlabel("Electric Range")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
# Outliers (IQR)
data = df['Electric Range'].dropna()

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = data[(data < lower) | (data > upper)]
print("\nIQR Outliers Count:", len(outliers_iqr))
print("Lower Bound:", lower)
print("Upper Bound:", upper)
# Outliers (Z-score)
z = (data - data.mean()) / data.std()
outliers_z = data[np.abs(z) > 3]
print("Z-score Outliers Count:", len(outliers_z))

# Boxplot
plt.figure()
sns.boxplot(
    x=data,
    color="steelblue",
    width=0.4
)
plt.title("Outlier Detection in Electric Range", fontsize=13)
plt.tight_layout()
plt.show()