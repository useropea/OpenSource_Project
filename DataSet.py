import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv("netflix_titles.csv")

print("shape:", df.shape)
print(df.head(3))

df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")

year_count = df.groupby("release_year")["show_id"].count()

type_year_mean = df.groupby("type")["release_year"].mean()

rating_year_max = df.groupby("rating")["release_year"].max()

print("\n[GroupBy 통계]")
print("연도별 콘텐츠 개수(count) 상위 5개:\n", year_count.sort_values(ascending=False).head())
print("\n타입별 출시연도 평균(mean):\n", type_year_mean)
print("\n등급별 출시연도 최댓값(max) 상위 10개:\n", rating_year_max.sort_values(ascending=False).head(10))

plt.figure()
df["type"].value_counts().plot(kind="bar")
plt.title("Type Distribution (Movie vs TV Show)")
plt.xlabel("Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure()
year_count.sort_index().plot(kind="line")
plt.title("Number of Titles by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

df_ml = df[["type", "release_year", "rating", "country", "listed_in"]].copy()
df_ml = df_ml.dropna(subset=["type", "release_year"])

df_ml["label"] = (df_ml["type"] == "TV Show").astype(int)

X = pd.get_dummies(df_ml[["release_year", "rating", "country", "listed_in"]],
                   columns=["rating", "country", "listed_in"],
                   drop_first=True)
y = df_ml["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n[ML 평가]")
print("Accuracy:", acc)
print("measure:", f1)
print("Confusion Matrix:\n", cm)
