import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_ev_data.csv")

data = df['Electric Range'].dropna()

# IQR Method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = data[(data < lower) | (data > upper)]
print("IQR Outliers:", len(outliers_iqr))

# Z-score
z = (data - data.mean()) / data.std()
outliers_z = data[np.abs(z) > 3]
print("Z-score Outliers:", len(outliers_z))

# Boxplot
plt.figure(figsize=(7,4))
sns.boxplot(x=data, color="indianred")

plt.title("Outlier Detection using IQR and Z-score", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()