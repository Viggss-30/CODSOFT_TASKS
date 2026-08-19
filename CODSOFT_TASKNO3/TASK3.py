import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_train.csv")

# Set visualization style
sns.set_theme(style="whitegrid")

print("Dataset loaded successfully!")
print(df.head())

# --------------------------------------------------
# 1. BAR CHART - Survival by Passenger Class
# --------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Pclass", y="Survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. LINE CHART - Average Fare by Passenger Class
# --------------------------------------------------
fare_by_class = df.groupby("Pclass")["Fare"].mean()

plt.figure(figsize=(8, 5))
plt.plot(fare_by_class.index, fare_by_class.values, marker="o")
plt.title("Average Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare")
plt.xticks(fare_by_class.index)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. PIE CHART - Survival Distribution
# --------------------------------------------------
survival_counts = df["Survived"].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(
    survival_counts,
    labels=["Did Not Survive", "Survived"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Passenger Survival Distribution")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. HISTOGRAM - Age Distribution
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["Age"].dropna(), bins=20, edgecolor="black")
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. SCATTER PLOT - Age vs Fare
# --------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived")
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------
print("\n--- TASK 3 INSIGHTS ---")

print("\nSurvival Rate by Class:")
print(df.groupby("Pclass")["Survived"].mean())

print("\nAverage Fare by Class:")
print(df.groupby("Pclass")["Fare"].mean())

print("\nSurvival Rate by Gender:")
print(df.groupby("Sex")["Survived"].mean())

print("\nAge Statistics:")
print(df["Age"].describe())