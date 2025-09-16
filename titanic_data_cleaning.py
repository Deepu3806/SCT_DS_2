import pandas as pd
import numpy as np

# 1. Load Dataset
df = pd.read_csv('train.csv')

# 2. Remove Duplicate Rows
df = df.drop_duplicates()

# 3. Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# 4. Convert Data Types
df['Pclass'] = df['Pclass'].astype('category')

# 5. Extract and Clean Titles from Names
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# Use Title to Impute Age if needed
title_median_age = df.groupby('Title')['Age'].median()
df['Age'] = df.apply(lambda row: title_median_age[row['Title']] if pd.isnull(row['Age']) else row['Age'], axis=1)

# 6. Remove Irrelevant Columns (optional for modeling)
df = df.drop(['Cabin', 'Ticket', 'Name'], axis=1)

# 7. Reset Index
df = df.reset_index(drop=True)

# 8. Validate Cleaning
print("Missing values:\n", df.isnull().sum())
print("First 5 rows:\n", df.head())