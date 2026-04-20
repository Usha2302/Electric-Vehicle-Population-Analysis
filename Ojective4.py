import pandas as pd
from scipy import stats

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")


# Significance level
alpha = 0.05

# Two-sample independent t-test (EV dataset)
state1 = df[df['State'] == 'CA']['Electric Range'].dropna()
state2 = df[df['State'] == 'WA']['Electric Range'].dropna()

t_stat, p_value = stats.ttest_ind(state1, state2, equal_var=True)

print("Two-sample t-test (Independent samples, equal variance):")
print(f"T-statistic = {t_stat:.4f}")
print(f"P-value = {p_value:.4f}")

if p_value < alpha:
    print("Conclusion: Reject the null hypothesis. There is a significant difference between the group means.")
else:
    print("Conclusion: Fail to reject the null hypothesis. No significant difference between the group means.")