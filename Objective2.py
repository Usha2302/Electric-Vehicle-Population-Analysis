import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")
# Total EV per year 
year_total = df.groupby('Model Year')['VIN (1-10)'].count()
print("Total EV per Year:\n", year_total)
# EV per state 
state_total = df.groupby('State')['VIN (1-10)'].count()
print("\nEV count per State:\n", state_total)
# Vehicle type distribution 
type_total = df.groupby('Electric Vehicle Type')['VIN (1-10)'].count()
print("\nEV by Vehicle Type:\n", type_total)

#Visualization 

# Line chart (Year-wise growth)
plt.figure()
plt.plot(year_total.index, year_total.values)
plt.title("Yearly EV Growth")
plt.xlabel("Year")
plt.ylabel("EV Count")
plt.show()

# Horizontal bar chart (Top 10 States)
state_top = state_total.sort_values(ascending=False).head(10)
plt.figure()
state_top.plot(kind='barh')   
plt.title("Top 10 States EV Population")
plt.xlabel("EV Count")
plt.ylabel("State")
plt.show()

# Bar chart (Vehicle Type)
plt.figure()
type_total.plot(kind='bar')
plt.title("Vehicle Type Distribution")
plt.xlabel("Vehicle Type")
plt.ylabel("EV Count")
plt.xticks(rotation=5)
plt.show()