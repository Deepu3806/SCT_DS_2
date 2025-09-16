import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# 1. Survival Counts
fig = plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Survival Counts")
fig.savefig("survival_counts.png")
plt.show(block=False)
plt.pause(3)   # keeps the plot open for 3 seconds
plt.close(fig)
print("Observation: More passengers did not survive than those who survived.")

# 2. Survival by Sex
fig = plt.figure()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Sex")
fig.savefig("survival_by_sex.png")
plt.show(block=False)
plt.pause(3)   # keeps the plot open for 3 seconds
plt.close(fig)
print("Observation: Females had a much higher survival rate compared to males.")

# 3. Survival by Passenger Class
fig = plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
fig.savefig("survival_by_class.png")
plt.show(block=False)
plt.pause(3)   # keeps the plot open for 3 seconds
plt.close(fig)
print("Observation: First-class passengers had better survival chances than third-class.")

# 4. Age Distribution with Survival
fig = plt.figure()
sns.histplot(data=df, x='Age', kde=True, bins=30, hue='Survived')
plt.title("Age Distribution with Survival")
fig.savefig("age_distribution.png")
plt.show(block=False)
plt.pause(3)   # keeps the plot open for 3 seconds
plt.close(fig)
print("Observation: Younger passengers had relatively better survival rates.")

# 5. Fare Distribution with Survival
fig = plt.figure()
sns.histplot(data=df, x='Fare', kde=True, bins=30, hue='Survived')
plt.title("Fare Distribution with Survival")
fig.savefig("fare_distribution.png")
plt.show(block=False)
plt.pause(3)   # keeps the plot open for 3 seconds
plt.close(fig)
print("Observation: Passengers who paid higher fares had better chances of survival.")