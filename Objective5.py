from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_ev_data.csv")

# Clean data
df = df[(df['Electric Range'] > 50) & 
        (df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)')]

year_data = df.groupby('Model Year')['Electric Range'].mean().reset_index()

X = year_data[['Model Year']]
y = year_data['Electric Range']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("R² Score:", round(r2_score(y, y_pred), 3))

plt.figure(figsize=(8,5))
plt.scatter(X, y, color="steelblue", s=60)
plt.plot(X, y_pred, color="indianred", linewidth=2)

plt.title("Model Year vs Electric Range", fontsize=14, fontweight='bold')
plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.tight_layout()
plt.show()