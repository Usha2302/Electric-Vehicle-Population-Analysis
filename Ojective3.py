import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# Correlation 
corr = df.corr(numeric_only=True)
print(corr)

# Heatmap
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Outliers using IQR
data = df['Electric Range'].dropna()

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = data[(data < lower) | (data > upper)]
print("IQR Outliers:\n", outliers_iqr)

# ---- Outliers using Z-score ----
z = (data - data.mean()) / data.std()
outliers_z = data[np.abs(z) > 3]
print("Z-score Outliers:\n", outliers_z)