from scipy import stats
import pandas as pd

df = pd.read_csv("cleaned_ev_data.csv")

df = df[df['Electric Range'] > 50]

old = df[df['Model Year'] <= 2020]['Electric Range']
new = df[df['Model Year'] > 2020]['Electric Range']

t_stat, p_value = stats.ttest_ind(old, new, equal_var=False)

print("T-statistic:", round(t_stat, 4))
print("P-value:", round(p_value, 4))

if p_value < 0.05:
    print("Significant difference exists")
else:
    print("No significant difference")