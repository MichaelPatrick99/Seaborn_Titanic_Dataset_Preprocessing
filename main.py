import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

titanic_dataframe = sns.load_dataset("titanic")
# titanic_dataframe = pd.DataFrame(titanic_data)

missing_data = titanic_dataframe.isnull()
missing_data_count = missing_data.sum()
missing_data_percentage = (missing_data_count / len(titanic_dataframe) ) * 100

titanic_dataframe_clean = titanic_dataframe.dropna()

age_data = titanic_dataframe_clean['age']

# Ploting the histogram to show age distribution of passengers

plt.figure(figsize=(8, 6))
plt.hist(age_data, bins=10, color='blue')

# Customizing the plot
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')

# Show the plot
plt.show()

# Using a box plot to compare the fares paid by passengers
plt.figure(figsize=(8, 6))
sns.boxplot(x='survived', y='fare', data=titanic_dataframe_clean)

# Plot customization
plt.title('Comparison of Fares: Survived vs. Not Survived')
plt.xlabel('Survival Status')
plt.ylabel('Fare')

# Show plot
plt.show()

# display option to show all columns
pd.set_option('display.max_columns', None)

# display option to show all rows
pd.set_option('display.max_rows', None)

# Viewing the first few rows of the dataset
print(titanic_data.head())

# Getting information about the dataset
print(titanic_data.info())

# Statistical summary of the dataset
print(titanic_data.describe())
