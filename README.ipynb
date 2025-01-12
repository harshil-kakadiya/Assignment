# prompt: Python implementation is required to Perform Exploratory Data Analysis (EDA) on the Mushrooms dataset,
# which consists of data observations and features. Your EDA should include summary statistics, distribution
# analysis, and visualizations of relationships between variables

# Check for missing values
print(dataset.isnull().sum())

# Summary statistics for numerical features (if any)
# In this dataset, all features are categorical.
# If there were numerical features, you would use:
# print(dataset.describe())

# Distribution analysis for categorical features
for column in dataset.columns:
    print(f"Distribution of '{column}':")
    print(dataset[column].value_counts())
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=dataset)
    plt.title(f"Distribution of {column}")
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.show()

# Visualizations of relationships between variables
# Since all variables are categorical, we can use pair plots
# and correlation matrices
# Create a smaller subset for pair plot to avoid overwhelming plot
subset = dataset.sample(n=500, random_state=42) # Randomly selects 500 rows
sns.pairplot(subset, hue="class", vars=dataset.columns[1:6], diag_kind='hist') # first five columns (excluding class)
plt.show()
# Calculate a correlation matrix (for categorical features)
# Using Cramer's V
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

#Calculate correlations
correlations = []
for col1 in dataset.columns:
    for col2 in dataset.columns:
      if col1 != col2: # Skip self comparison
          corr = cramers_v(dataset[col1], dataset[col2])
          correlations.append((col1, col2, corr))

#convert into pandas dataframe
corr_df = pd.DataFrame(correlations, columns=["Column1", "Column2", "Correlation"])

#print correlation values
print(corr_df)


# Heatmap of Correlation
corr_matrix = corr_df.pivot(index='Column1', columns='Column2', values='Correlation')
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
