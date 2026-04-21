
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
df = pd.read_csv("cleaned_ev_data.csv")

# Aggregations
year_total = df.groupby('Model Year')['VIN (1-10)'].count().reset_index()
year_total.columns = ['Year', 'EV Count']
state_total = df.groupby('State')['VIN (1-10)'].count().reset_index()
state_total.columns = ['State', 'EV Count']
type_total = df.groupby('Electric Vehicle Type')['VIN (1-10)'].count().reset_index()
type_total.columns = ['Type', 'Count']
type_total['Type'] = type_total['Type'].replace({'Battery Electric Vehicle (BEV)': 'BEV',
    'Plug-in Hybrid Electric Vehicle (PHEV)': 'PHEV'
})
print("Total EV per Year:\n", year_total)

print("\nEV count per State:\n", state_total)

print("\nEV by Vehicle Type:\n", type_total)

# LINE CHART 
plt.figure(figsize=(9,5))
sns.lineplot(data=year_total, x='Year', y='EV Count', color="steelblue", linewidth=2.5)
plt.fill_between(year_total['Year'], year_total['EV Count'], color="steelblue", alpha=0.3)
plt.title("EV Adoption Trend", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Vehicles")
plt.tight_layout()
plt.show()

# BAR CHART 
state_top = state_total.sort_values(by='EV Count', ascending=False).head(10)
plt.figure(figsize=(9,5))
sns.barplot(data=state_top, x='EV Count', y='State', color="steelblue")
plt.xscale('log')
plt.title("Top 10 States EV Population", fontsize=14)
plt.xlabel("EV Count")
plt.ylabel("State")
plt.tight_layout()
plt.show()

#  PIE CHART 
plt.figure(figsize=(6,6))
plt.pie(
    type_total['Count'],
    labels=type_total['Type'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['steelblue', 'indianred']
)
plt.title("Vehicle Type Distribution", fontsize=14)
plt.tight_layout()
plt.show()
