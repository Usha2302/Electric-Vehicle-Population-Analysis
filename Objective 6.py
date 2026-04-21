
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme(style="white")
df = pd.read_csv("cleaned_ev_data.csv")
#FEATURE ENGINEERING
df['EV Type Encoded'] = df['Electric Vehicle Type'].map({
    'Battery Electric Vehicle (BEV)': 1,
    'Plug-in Hybrid Electric Vehicle (PHEV)': 0
})
# MODEL YEAR TREND
year_data = df.groupby('Model Year')['VIN (1-10)'].count().reset_index()
year_data.columns = ['Year', 'EV Count']
year_data = year_data[year_data['Year'] < year_data['Year'].max()]
year_data = year_data.sort_values(by='Year')

# STATE-LEVEL FACTORS
state_data = df.groupby('State').agg({'VIN (1-10)': 'count','Electric Range': 'mean','EV Type Encoded': 'mean'}).reset_index()
state_data.columns = ['State', 'EV Count', 'Average Range', 'BEV Ratio']

# REGRESSION MODEL 
X = state_data[['Average Range', 'BEV Ratio']]
y = state_data['EV Count']
model = LinearRegression()
model.fit(X, y)
print("\nFACTOR IMPACT")
print("Electric Range Impact:", round(model.coef_[0], 2))
print("BEV Ratio Impact:", round(model.coef_[1], 2))

#  SEGMENTATION 
high = state_data['EV Count'].quantile(0.75)
low = state_data['EV Count'].quantile(0.25)
def segment(x):
    if x >= high:
        return "High Growth"
    elif x <= low:
        return "Low Growth"
    else:
        return "Moderate Growth"

state_data['Segment'] = state_data['EV Count'].apply(segment)

print("\nSegment Distribution:\n", state_data['Segment'].value_counts())

# SINGLE SCENARIO 
scenario = state_data.copy()

scenario['Average Range'] *= 1.15
scenario['BEV Ratio'] = np.clip(scenario['BEV Ratio'] + 0.1, 0, 1)

state_data['Current'] = model.predict(X)
state_data['Improved'] = model.predict(scenario[['Average Range', 'BEV Ratio']])
#VISUALIZATION 1: ADVANCED TREND
plt.figure(figsize=(8,4))
plt.plot(year_data['Year'], year_data['EV Count'],color='steelblue', linewidth=2, marker='o')
# Trend curve
z = np.polyfit(year_data['Year'], year_data['EV Count'], 2)
p = np.poly1d(z)
plt.plot(year_data['Year'], p(year_data['Year']),
         color='indianred', linestyle='--', label='Growth Trend')
plt.title("Temporal Growth Analysis of Electric Vehicle Adoption (with Trend Modeling)")
plt.xlabel("Model Year")
plt.ylabel("Number of Electric Vehicles")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# VISUALIZATION 2: FACTOR RELATION 
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=state_data,
    x='Average Range',
    y='EV Count',
    hue='Segment',
    palette={
        "High Growth": "steelblue",
        "Moderate Growth": "yellowgreen",
        "Low Growth": "indianred"
    },
    s=90
)
plt.yscale('log')

plt.title("Impact of Electric Range on EV Adoption Across Growth Segments")
plt.xlabel("Average Electric Range")
plt.ylabel("EV Count (Log Scale)")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

#  VISUALIZATION 3: SCENARIO IMPACT 
top_states = state_data.sort_values('Current', ascending=False).head(8)

x = np.arange(len(top_states))
width = 0.35
plt.figure(figsize=(10,5))
plt.bar(x - width/2, top_states['Current'], width,
        color='steelblue', edgecolor='black', label='Current')
plt.bar(x + width/2, top_states['Improved'], width,
        color='indianred', edgecolor='black', label='Improved Scenario')
plt.xticks(x, top_states['State'], rotation=45)
plt.title("Projected Impact of Policy Intervention on EV Adoption Across States")
plt.xlabel("State")
plt.ylabel("Number of Electric Vehicles")
plt.legend()
plt.tight_layout()
plt.show()

# IMPACT ANALYSIS
growth = np.where(
    state_data['Current'] != 0,
    ((state_data['Improved'] - state_data['Current']) /
     state_data['Current']) * 100,
    0
)

print("\nAverage Growth:", round(growth.mean(), 2), "%")

#  STRATEGIC RECOMMENDATIONS 
print("\nSTRATEGY:")
print("- High-growth states: Expand charging infrastructure")
print("- Moderate-growth states: Increase incentives and awareness")
print("- Low-growth states: Focus on battery improvement and subsidies")
print("- Improving electric range and BEV adoption significantly increases EV growth")