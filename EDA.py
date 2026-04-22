import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


df = pd.read_csv("cleaned_ev_data.csv")
df = df.rename(columns={
    'Postal Code': 'Zip',
    'Model Year': 'Year',
    'Electric Range': 'Range',
    'Legislative District': 'District',
    'DOL Vehicle ID': 'VehicleID',
    '2020 Census Tract': 'Census'
})
print("First 5 Rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nStatistical Summary:\n", df.describe())
print("\nShape of Dataset:", df.shape)

# DISTRIBUTION ANALYSIS

plt.figure(figsize=(8,5))

sns.histplot(df['Range'].dropna(),
             bins=40,
             kde=True,
             color="steelblue",
             edgecolor="black")

plt.title("Distribution of Electric Vehicle Range", fontsize=14, fontweight='bold')
plt.xlabel("Electric Range")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

print("Insight: Electric range shows a skewed distribution with most vehicles having moderate range.")

# CORRELATION HEATMAP
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(9,6))

sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5)

plt.title("Correlation Matrix of Numerical Features", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("Insight: Correlation heatmap shows relationships between numerical variables.")



# BOXPLOT (OUTLIER VISUALIZATION)

plt.figure(figsize=(7,4))

sns.boxplot(x=df['Range'],
            color="steelblue")

plt.title("Outlier Visualization in Electric Range", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("Insight: Presence of extreme values indicates variability in EV range.")



# PAIRPLOT (CLEAN LABELS)

sample_data = numeric_df.sample(500, random_state=42)

sns.pairplot(sample_data,
             diag_kind='kde',
             height=1.5)

plt.suptitle("Pairwise Relationships Between Features", y=1.02)

plt.show()

print("Insight: Pairplot helps visualize relationships and patterns among variables.")
# FINAL EDA SUMMARY
print("""
EDA SUMMARY:
- EV range distribution is slightly skewed.
- Some extreme values (outliers) are present.
- Relationships between features are generally weak to moderate.
- Dataset is suitable for further statistical analysis and modeling.
""")