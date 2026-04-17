import pandas as pd

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")
print(df)

# First 10 rows
print(df.iloc[0:10])

# Structure
print(df.columns)
print(df.dtypes)
print(df.shape)

# Summary
print(df.describe())

# Missing values
print(df.isnull().sum())

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))