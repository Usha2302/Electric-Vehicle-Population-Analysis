import pandas as pd
from scipy import stats

df = pd.read_csv("cleaned_ev_data.csv")

alpha = 0.05

# Select two groups (CA vs WA)
state1 = df[df['State'] == 'CA']['Electric Range'].dropna()
state2 = df[df['State'] == 'WA']['Electric Range'].dropna()
# Perform Independent T-test (Welch)
t_stat, p_value = stats.ttest_ind(state1, state2, equal_var=False)
print("T-test Results")
print(f"T-statistic = {t_stat:.4f}")
if p_value < 0.001:
    print("P-value < 0.001")
else:
    print(f"P-value = {p_value:.4f}")
# Interpretation
if p_value < alpha:
    print("Conclusion: Reject H0 → Significant difference between the two states.")
else:
    print("Conclusion: Fail to reject H0 → No significant difference between the two states.")