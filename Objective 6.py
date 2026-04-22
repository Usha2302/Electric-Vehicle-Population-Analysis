import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("cleaned_ev_data.csv")

df['EV Type Encoded'] = df['Electric Vehicle Type'].map({
    'Battery Electric Vehicle (BEV)': 1,
    'Plug-in Hybrid Electric Vehicle (PHEV)': 0
})

state_data = df.groupby('State').agg({
    'VIN (1-10)': 'count',
    'Electric Range': 'mean',
    'EV Type Encoded': 'mean',
    'Model Year': 'mean'
}).reset_index()

state_data.columns = ['State','EV Count','Avg Range','BEV Ratio','Avg Year']

X = state_data[['Avg Range','BEV Ratio','Avg Year']]
y = state_data['EV Count']

model = LinearRegression()
model.fit(X, y)

scenario = state_data.copy()
scenario['Avg Range'] *= 1.15
scenario['BEV Ratio'] = np.clip(scenario['BEV Ratio'] + 0.1, 0, 1)
scenario['Avg Year'] += 1

state_data['Current'] = model.predict(X)
state_data['Improved'] = model.predict(
    scenario[['Avg Range','BEV Ratio','Avg Year']]
)

top = state_data.sort_values('Current', ascending=False).head(8)

x = np.arange(len(top))
width = 0.35

plt.figure(figsize=(10,5))

plt.bar(x - width/2, top['Current'], width,
        color="steelblue", label="Current")

plt.bar(x + width/2, top['Improved'], width,
        color="indianred", label="Improved")

plt.xticks(x, top['State'], rotation=45)

plt.title("Scenario-Based EV Adoption Analysis", fontsize=14, fontweight='bold')
plt.xlabel("State")
plt.ylabel("EV Count")
plt.legend()
plt.tight_layout()
plt.show()