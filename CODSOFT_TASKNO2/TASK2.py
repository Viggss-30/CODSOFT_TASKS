import pandas as pd
df=pd.read_csv("cleaned_train.csv")
print(df.head())
print(df.shape)
print("\nDescriptive Statistics:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)

print("\nAverage Fare by Class:")
print(df.groupby('Pclass')['Fare'].mean())

print("\nSurvival Rate by Class:")
print(df.groupby('Pclass')['Survived'].mean())

print("\nSurvival Rate by Gender:")
print(df.groupby('Sex')['Survived'].mean())

print("\nCorrelation :")
print(df.select_dtypes(include="number").corr())

print("\nOutliers:")
Q1=df['Fare'].quantile(0.25)
Q3=df['Fare'].quantile(0.75)
IQR=Q3-Q1
outliers=df[(df['Fare']<Q1-1.5*IQR)| (df['Fare']>Q3+1.5*IQR)]
print("Number of Fare outliers:",len(outliers))

print("\nSurvival Count:")
print(df["Survived"].value_counts())

print("\nAverage Age by Survival:")
print(df.groupby('Survived')['Age'].mean())

print("\nAverage Fare by Survival:")
print(df.groupby('Survived')['Fare'].mean())

import matplotlib.pyplot as plt
df["Survived"].value_counts().plot(kind='bar', title='Survival Count')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Number of Passengers')
plt.show()