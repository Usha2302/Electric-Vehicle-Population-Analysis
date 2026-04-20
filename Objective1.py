import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Electric_Vehicle_Population_Data.csv")
print(df)
print(df.info())
print(df.describe())
print(df.head(10))
print("Column Names:\n", df.columns)
print("Data Types:\n", df.dtypes)
print("Shape of Dataset:\n", df.shape)

# Count missing values in each column
print("Missing values in each column:\n")

print(df.isnull().sum())
# Fill missing values for numerical columns (mean)
df_fillna = df.fillna(df.mean(numeric_only=True),inplace=True)
# Fill missing values for categorical columns (mode)
for col in df.select_dtypes(include=['object', 'string']):
    df[col] = df[col].fillna(df[col].mode()[0])
print("After replacing missing values:\n")
print(df.isnull().sum())
#Create new column (Range Category) 
def categorize_range(x):
    if x < 50:
        return "Low"
    elif x < 150:
        return "Medium"
    else:
        return "High"

df['Range Category'] = df['Electric Range'].apply(categorize_range)

# Count each category
category_count = df['Range Category'].value_counts()
print("\nRange Categories:\n", category_count)

# Bar chart
category_count.plot(kind='bar')
plt.title("Electric Range Categories")
plt.xlabel("Category")
plt.ylabel("Number of Vehicles")
plt.show()