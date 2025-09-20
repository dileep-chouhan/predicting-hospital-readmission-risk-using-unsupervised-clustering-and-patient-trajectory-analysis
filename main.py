import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_patients = 200
data = {
    'Age': np.random.randint(30, 80, num_patients),
    'LengthOfStay': np.random.randint(1, 14, num_patients),
    'NumComorbidities': np.random.randint(0, 5, num_patients),
    'Readmitted': np.random.randint(0, 2, num_patients) # 0: No, 1: Yes
}
df = pd.DataFrame(data)
#Adding some correlation for better clustering
df['LengthOfStay'] = df['LengthOfStay'] + df['NumComorbidities'] * 2 + np.random.normal(0,1, num_patients)
df['Age'] = df['Age'] + df['NumComorbidities'] * 0.5 + np.random.normal(0,2, num_patients)
# --- 2. Data Preprocessing ---
# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Age', 'LengthOfStay', 'NumComorbidities']])
df_scaled = pd.DataFrame(scaled_features, columns=['Age_scaled', 'LengthOfStay_scaled', 'NumComorbidities_scaled'])
# --- 3. Unsupervised Clustering (K-Means) ---
kmeans = KMeans(n_clusters=3, random_state=42) # Choosing 3 clusters as an example.  Optimal k should be determined.
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_
# --- 4. Analysis ---
# Analyze cluster characteristics
cluster_stats = df.groupby('Cluster').agg({'Readmitted': 'mean', 'Age': 'mean', 'LengthOfStay': 'mean', 'NumComorbidities': 'mean'})
print("Cluster Statistics:")
print(cluster_stats)
# --- 5. Visualization ---
# Visualize clusters (example: scatter plot of Age vs. LengthOfStay)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='LengthOfStay', hue='Cluster', data=df, palette='viridis')
plt.title('Patient Clusters based on Age and Length of Stay')
plt.xlabel('Age')
plt.ylabel('Length of Stay')
plt.savefig('patient_clusters.png')
print("Plot saved to patient_clusters.png")
#Visualizing Readmission Rates per Cluster
plt.figure(figsize=(8,6))
sns.barplot(x=cluster_stats.index, y=cluster_stats['Readmitted'])
plt.title('Readmission Rate per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Readmission Rate')
plt.savefig('readmission_rates.png')
print("Plot saved to readmission_rates.png")
#Note:  Further analysis would involve more sophisticated techniques to determine optimal number of clusters, assess cluster validity, and potentially incorporate time-series analysis for patient trajectory analysis.  This is a simplified example.