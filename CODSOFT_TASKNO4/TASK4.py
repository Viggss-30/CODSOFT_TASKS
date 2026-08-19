import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load customer dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset Information:")
print(df.info())

# Basic customer analysis
print("\nBasic Statistics:")
print(df.describe())

print("\nGender Distribution:")
print(df["Genre"].value_counts())

print("\nAverage Spending Score by Gender:")
print(df.groupby("Genre")["Spending Score (1-100)"].mean())

# Customer segmentation by age
print("\nCustomer Segmentation by Age:")

df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 50, 100],
    labels=["Young", "Adult", "Middle Age", "Senior"]
)

print(df["Age Group"].value_counts())

print("\nAverage Spending Score by Age Group:")
print(df.groupby("Age Group", observed=True)["Spending Score (1-100)"].mean())

# Most valuable customer groups
print("\nMost Valuable Customer Groups:")

valuable_groups = df.groupby(
    ["Genre", "Age Group"],
    observed=True
)["Spending Score (1-100)"].mean().sort_values(ascending=False)

print(valuable_groups)

# Visualization 1: Spending Score by Age Group

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Age Group",
    y="Spending Score (1-100)"
)

plt.title("Average Spending Score by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Spending Score")
plt.tight_layout()
plt.show()

# Visualization 2: Spending Score by Gender and Age Group

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Age Group",
    y="Spending Score (1-100)",
    hue="Genre"
)

plt.title("Average Spending Score by Genre and Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Spending Score")
plt.legend(title="Genre")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. CUSTOMER DISTRIBUTION BY AGE GROUP
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Age Group")

plt.title("Customer Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. CUSTOMER SEGMENTATION BY GENDER
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Genre", hue="Genre", legend=False)

plt.title("Customer Distribution by Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. KEY CUSTOMER INSIGHTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("KEY CUSTOMER INSIGHTS")
print("=" * 50)

print("\n1. Most valuable customer group:")
print("Female Adults have the highest average spending score.")

print("\n2. Highest spending age group:")
print("Adult customers have the highest average spending score.")

print("\n3. Young customers:")
print("Young customers also show strong spending potential.")

print("\n4. Gender distribution:")
print("Female customers are more numerous than male customers.")

print("\n5. Spending trend:")
print("Spending score generally decreases as customer age increases.")


# --------------------------------------------------
# 6. MARKETING STRATEGIES
# --------------------------------------------------

print("\n" + "=" * 50)
print("RECOMMENDED MARKETING STRATEGIES")
print("=" * 50)

print("\n1. Target Adult Customers:")
print("Offer premium products, loyalty programs, and personalized promotions.")

print("\n2. Target Young Customers:")
print("Use social media campaigns, discounts, and trendy products.")

print("\n3. Target Female Customers:")
print("Promote products and offers based on their strong spending potential.")

print("\n4. Improve Middle-Age Engagement:")
print("Use personalized offers and loyalty rewards to increase spending.")

print("\n5. Retain Senior Customers:")
print("Provide simple offers, personalized recommendations, and loyalty benefits.")


# --------------------------------------------------
# 7. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("FINAL CUSTOMER ANALYSIS SUMMARY")
print("=" * 50)

print("""
The analysis shows that Adult customers are the most valuable
age group based on spending score. Female Adult customers have
the highest average spending score among the identified segments.

Young customers also have strong spending potential, while
Middle Age and Senior groups have comparatively lower spending
scores.

Therefore, marketing efforts should focus primarily on Adult
and Young customers while using targeted loyalty strategies
to increase engagement among older customer groups.
""")