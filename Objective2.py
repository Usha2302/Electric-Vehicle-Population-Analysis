import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("cleaned_ev_data.csv")

# EV Trend
year_data = df.groupby('Model Year').size().reset_index(name='EV Count')
year_data = year_data[year_data['Model Year'] < year_data['Model Year'].max()]

plt.figure(figsize=(9,5))
sns.lineplot(data=year_data, x='Model Year', y='EV Count',
             color="steelblue", marker='o', linewidth=2.5)

plt.fill_between(year_data['Model Year'], year_data['EV Count'],
                 color="steelblue", alpha=0.2)

plt.title("Electric Vehicle Adoption Trend", fontsize=14, fontweight='bold')
plt.xlabel("Model Year")
plt.ylabel("EV Count")
plt.tight_layout()
plt.show()

# Region Distribution
region_top = df.groupby('County').size().reset_index(name='EV Count') \
               .sort_values(by='EV Count', ascending=False).head(10)

plt.figure(figsize=(9,5))
sns.barplot(data=region_top, x='EV Count', y='County', color="steelblue")

plt.title("Top 10 Regions by EV Population", fontsize=14, fontweight='bold')
plt.xlabel("EV Count")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# Vehicle Type
type_data = df['Electric Vehicle Type'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(type_data,
        labels=["BEV", "PHEV"],
        autopct='%1.1f%%',
        colors=["steelblue", "indianred"])

plt.title("EV Type Distribution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
