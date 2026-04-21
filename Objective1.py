
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
df = pd.read_csv("Electric_Vehicle_Population_Data.csv")
print(df.info())
print(df.describe())
print("Shape:", df.shape)

# MISSING VALUES
print("\nMissing values:\n", df.isnull().sum())

# Fill numerical columns with mean
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nAfter cleaning:\n", df.isnull().sum())

# FEATURE ENGINEERING
def categorize_range(x):
    if pd.isna(x):
        return "Low"
    elif x < 50:
        return "Low"
    elif x < 150:
        return "Medium"
    else:
        return "High"

df['Range Category'] = df['Electric Range'].apply(categorize_range)

# CATEGORY COUNT 
category_count = df['Range Category'].value_counts()
order = ["Low", "Medium", "High"]
category_count = category_count.reindex(order, fill_value=0)

print("\nRange Categories:\n", category_count)

# Bar chart
plt.figure(figsize=(7,4))
sns.barplot(
    x=category_count.index,
    y=category_count.values,
    hue=category_count.index,
    palette=["steelblue", "yellowgreen", "indianred"],
    legend=False
)

plt.title("Electric Range Categories", fontsize=14)
plt.xlabel("Category", fontsize=12)
plt.ylabel("Number of Vehicles", fontsize=12)

plt.tight_layout()
plt.show()
df.to_csv("cleaned_ev_data.csv", index=False)