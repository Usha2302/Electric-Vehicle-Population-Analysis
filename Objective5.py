from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# Prepare data
year_count = df.groupby('Model Year')['VIN (1-10)'].count().reset_index()
X = year_count[['Model Year']]
y = year_count['VIN (1-10)']
# Model
model = LinearRegression()
model.fit(X, y)
# Prediction
y_pred = model.predict(X)
# Print equation
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
# Plot
plt.figure()
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.title("EV Growth Prediction using Linear Regression")
plt.xlabel("Model Year")
plt.ylabel("EV Count")
plt.legend()
plt.show()