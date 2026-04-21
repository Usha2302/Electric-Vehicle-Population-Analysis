
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("cleaned_ev_data.csv")
# Aggregate EV count per year
year_count = df.groupby('Model Year')['VIN (1-10)'].count().reset_index()
year_count.columns = ['Year', 'EV Count']
# SORT DATA 
year_count = year_count.sort_values(by='Year')
year_count = year_count[year_count['EV Count'] > 0]
year_count = year_count[year_count['Year'] < year_count['Year'].max()]

X = year_count[['Year']]
y = year_count['EV Count']
# Train model
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
# Metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("Slope (m):", round(model.coef_[0], 2))
print("Intercept (c):", round(model.intercept_, 2))
print("R² Score:", round(r2, 4))
print("RMSE:", round(rmse, 2))
# Plot
plt.figure(figsize=(8,5))
plt.scatter(X, y, color="steelblue", s=60, label="Actual Data")
plt.plot(X, y_pred, color="indianred", linewidth=2.5, label="Regression Line")
plt.title("EV Growth Trend using Linear Regression", fontsize=14)
plt.xlabel("Year")
plt.ylabel("EV Count")
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
eq_text = f"y = {model.coef_[0]:.0f}x + {model.intercept_:.0f}\nR² = {r2:.3f}"
plt.text(
    0.05, 0.85,
    eq_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.7)
)
plt.show()