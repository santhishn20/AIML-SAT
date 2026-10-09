from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(Path(__file__).resolve().parent / "IN.csv")

features_to_use = [
    "new_confirmed", "cumulative_confirmed", "new_deceased",
    "cumulative_deceased", "new_tested", "cumulative_tested",
    "new_persons_vaccinated", "cumulative_persons_vaccinated",
    "new_persons_fully_vaccinated",
    "cumulative_persons_fully_vaccinated",
    "new_vaccine_doses_administered",
    "cumulative_vaccine_doses_administered",
    "population_density", "human_development_index", "life_expectancy",
    "smoking_prevalence", "diabetes_prevalence", "nurses_per_1000",
    "physicians_per_1000", "average_temperature_celsius",
    "minimum_temperature_celsius", "maximum_temperature_celsius",
    "rainfall_mm", "dew_point", "relative_humidity",
]
features_to_use += [column for column in data.columns
                    if column.startswith("mobility_")]
features_to_use = [column for column in features_to_use
                   if column in data.columns]

features = data[features_to_use].apply(pd.to_numeric, errors="coerce")
features = features.dropna(axis=1, how="all")
features = features.loc[:, features.nunique(dropna=True) > 1]
features = features.fillna(features.mean())

scaled_data = StandardScaler().fit_transform(features)
pca = PCA()
principal_components = pca.fit_transform(scaled_data)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
components_for_95 = np.argmax(cumulative_variance >= 0.95) + 1

print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nExplained variance ratio:")
print(explained_variance)
print("\nCumulative explained variance:")
print(cumulative_variance)
print("\nComponents needed for 95% variance:", components_for_95)
print(f"Variance retained by 2D: {explained_variance[:2].sum():.4f}")
print(f"Variance retained by 3D: {explained_variance[:3].sum():.4f}")

plt.figure()
plt.scatter(principal_components[:, 0], principal_components[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("COVID-19 PCA: 2D")

figure = plt.figure()
axis = figure.add_subplot(111, projection="3d")
axis.scatter(principal_components[:, 0],
             principal_components[:, 1],
             principal_components[:, 2])
axis.set_xlabel("PC1")
axis.set_ylabel("PC2")
axis.set_zlabel("PC3")
axis.set_title("COVID-19 PCA: 3D")

plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100,
         marker="o")
plt.axhline(y=95, color="red", linestyle="--")
plt.xlabel("Principal Component Number")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("COVID-19 PCA: Explained Variance")
plt.grid(True)
plt.savefig(Path(__file__).resolve().parent / "pca_explained_variance.png")
plt.show()
