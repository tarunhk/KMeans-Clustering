import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

print(df.head())


X = df.select_dtypes(include=['int64','float64'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = labels

score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)

plt.figure(figsize=(6,4))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=labels, palette='Set1')
plt.title("K-Means Clustering")
plt.show()