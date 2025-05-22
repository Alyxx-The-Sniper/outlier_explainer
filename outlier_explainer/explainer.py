import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer

class OutlierExplainerLime:
    def __init__(self, X, model_type="iso", contamination=0.1):
        self.X_raw = X.reset_index(drop=True)  # ensure clean indexing
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X)
        self.model_type = model_type
        self.contamination = contamination
        self.model = self._init_model()
        self.outlier_indices = None

        # LIME should use unscaled data
        self.explainer = LimeTabularExplainer(
            training_data=self.X_raw.values,
            feature_names=self.X_raw.columns.tolist(),
            class_names=["inlier", "outlier"],
            mode="classification",
            discretize_continuous=True
        )

    def _init_model(self):
        if self.model_type == "iso":
            return IsolationForest(contamination=self.contamination, random_state=42)
        elif self.model_type == "ocsvm":
            return OneClassSVM(nu=self.contamination, gamma='scale')
        elif self.model_type == "lof":
            return LocalOutlierFactor(n_neighbors=20, contamination=self.contamination, novelty=True)
        else:
            raise ValueError("Unsupported model type")

    def fit_predict(self):
        self.model.fit(self.X)
        preds = self.model.predict(self.X)
        self.outlier_indices = np.where(preds == -1)[0]
        return self.outlier_indices

    def list_outliers(self):
        return self.X_raw.iloc[self.outlier_indices].reset_index(drop=True)

    def explain_outlier(self, index, num_features=5):
        instance_raw = self.X_raw.iloc[index].values
        explanation = self.explainer.explain_instance(instance_raw, self._predict_lime, num_features=num_features)
        explanation.show_in_notebook()
        return explanation

    def _predict_lime(self, X_raw):
        X_scaled = self.scaler.transform(X_raw)
        preds = self.model.predict(X_scaled)
        # Convert model's -1/1 output to LIME's expected class probabilities
        return np.array([[1 if p == -1 else 0, 0 if p == -1 else 1] for p in preds])


    def visualize_outliers(self, method='pca', dims=2):
      if method == 'pca':
          reducer = PCA(n_components=dims)
      elif method == 'umap':
          reducer = umap.UMAP(n_components=dims, random_state=42)
      else:
          raise ValueError("Only 'pca' and 'umap' are supported")

      X_reduced = reducer.fit_transform(self.X)
      labels = ["outlier" if i in self.outlier_indices else "inlier" for i in range(len(self.X))]

      # Define consistent colors for labels
      color_map = {"inlier": "C0", "outlier": "C1"}  # Can use any colors you prefer

      if dims == 2:
          plt.figure(figsize=(10, 6))
          sns.scatterplot(
              x=X_reduced[:, 0],
              y=X_reduced[:, 1],
              hue=labels,
              palette=color_map
          )
          plt.title(f"{method.upper()} 2D Visualization of Outliers")
          plt.show()

      elif dims == 3:
          from mpl_toolkits.mplot3d import Axes3D
          fig = plt.figure(figsize=(10, 6))
          ax = fig.add_subplot(111, projection='3d')
          for label in set(labels):
              idx = [i for i, l in enumerate(labels) if l == label]
              ax.scatter(
                  X_reduced[idx, 0],
                  X_reduced[idx, 1],
                  X_reduced[idx, 2],
                  label=label,
                  color=color_map[label]
              )
          ax.set_title(f"{method.upper()} 3D Visualization of Outliers")
          ax.legend()
          plt.show()

      else:
          raise ValueError("Only 2D and 3D visualizations are supported")