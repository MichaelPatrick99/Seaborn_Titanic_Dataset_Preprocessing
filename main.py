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

# Set the display option to show all columns
pd.set_option('display.max_columns', None)

# Set the display option to show all rows
pd.set_option('display.max_rows', None)

# View the first few rows of the dataset
print(titanic_data.head())

# Get information about the dataset
print(titanic_data.info())

# Statistical summary of the dataset
print(titanic_data.describe())
