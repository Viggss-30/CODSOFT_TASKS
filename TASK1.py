import pandas as pd
df=pd.read_csv("train.csv")
print(df.head())
print(df.shape)
df.info()

print(df.isnull().sum())
print("Duplicate rows:",
df.duplicated().sum())

df["Age"] =df["Age"].fillna(df["Age"].median())

df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df=df.drop("Cabin",axis=1)

print(df.isnull().sum())

df.to_csv("cleaned_train.csv",index=False)