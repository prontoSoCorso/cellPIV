import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data generation
np.random.seed(42)
num_patients = 50
ages = np.random.randint(25, 45, size=num_patients)
total_embryos = np.random.randint(5, 15, size=num_patients)
blasto_counts = [np.random.randint(0, n+1) for n in total_embryos]
non_blasto_counts = [n - b for n, b in zip(total_embryos, blasto_counts)]

df = pd.DataFrame({
    'age': ages,
    'blasto': blasto_counts,
    'non_blasto': non_blasto_counts
})

# Calculate proportions
df['blasto_perc'] = df['blasto'] / (df['blasto'] + df['non_blasto']) * 100
df['non_blasto_perc'] = df['non_blasto'] / (df['blasto'] + df['non_blasto']) * 100

# Group by age and calculate average proportions
age_grouped = df.groupby('age')[['blasto_perc', 'non_blasto_perc']].mean().reset_index()

# Plotting
plt.figure(figsize=(12, 6))
bar_width = 0.6
indices = np.arange(len(age_grouped))

# Plot blastocyst percentages
plt.bar(indices, age_grouped['blasto_perc'], bar_width, label='Blastocysts (%)', color='green')

# Plot non-blastocyst percentages on top
plt.bar(indices, age_grouped['non_blasto_perc'], bar_width, bottom=age_grouped['blasto_perc'], label='Non-Blastocysts (%)', color='red')

# Annotate percentages
for i in indices:
    plt.text(i, age_grouped['blasto_perc'][i]/2, f"{age_grouped['blasto_perc'][i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    plt.text(i, age_grouped['blasto_perc'][i] + age_grouped['non_blasto_perc'][i]/2, f"{age_grouped['non_blasto_perc'][i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')

# Customize x-axis labels
plt.xticks(indices, age_grouped['age'], rotation=45)
plt.xlabel("Age")
plt.ylabel("Average Percentage of Embryos")
plt.title("Average Proportions of Blastocysts vs. Non-Blastocysts by Age")
plt.legend()
plt.tight_layout()
plt.show()
