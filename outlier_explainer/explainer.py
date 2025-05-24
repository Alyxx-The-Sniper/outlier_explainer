import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt

import umap
import shap
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lime.lime_tabular import LimeTabularExplainer


from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from IPython.display import display, HTML

#### Lime Class
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
      

##### Dlime Class
class OutlierExplainerDlime:
    def __init__(self, X, model_type="iso", contamination=0.1, n_clusters=10):
        self.X_raw = X.reset_index(drop=True)
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X)
        self.model_type = model_type
        self.contamination = contamination
        self.n_clusters = n_clusters
        self.model = self._init_model()
        self.outlier_indices = None
        self.cluster_labels = AgglomerativeClustering(n_clusters=self.n_clusters).fit_predict(self.X)

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

    def explain_outlier(self, index, top_k=5, visualize=True):
        instance = self.X[index].reshape(1, -1)
        instance_raw_values = self.X_raw.iloc[index].values
        k = 10
        nn = NearestNeighbors(n_neighbors=k).fit(self.X)
        _, indices = nn.kneighbors(instance)

        neighbor_clusters = self.cluster_labels[indices[0]]
        majority_cluster = np.bincount(neighbor_clusters).argmax()
        cluster_indices = np.where(self.cluster_labels == majority_cluster)[0]
        X_cluster = self.X[cluster_indices]

        y_cluster = self.model.predict(X_cluster)
        y_cluster = np.array([1 if y == -1 else 0 for y in y_cluster])

        lr = LinearRegression()
        lr.fit(X_cluster, y_cluster)
        feature_importances = lr.coef_

        explanation_df = pd.DataFrame({
            'Feature': self.X_raw.columns,
            'Importance': feature_importances,
            'Value': instance_raw_values
        }).sort_values(by='Importance', ascending=False)

        display(HTML(explanation_df.head(top_k).to_html(index=False)))

        if visualize:
            self._visualize_explanation(explanation_df.head(top_k))

        return explanation_df.reset_index(drop=True)

    def _visualize_explanation(self, explanation_df):
        plt.figure(figsize=(8, 5))
        bars = plt.barh(explanation_df['Feature'], explanation_df['Importance'], color='orange')
        plt.xlabel('Importance')
        plt.title('DLIME Explanation - Top Contributing Features')
        plt.gca().invert_yaxis()

        for bar, value in zip(bars, explanation_df['Value']):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'Value: {value}', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

    def visualize_outliers(self, method='pca', dims=2):
        reducer = PCA(n_components=dims) if method == 'pca' else umap.UMAP(n_components=dims, random_state=42)
        X_reduced = reducer.fit_transform(self.X)
        labels = ["outlier" if i in self.outlier_indices else "inlier" for i in range(len(self.X))]
        color_map = {"inlier": "C0", "outlier": "C1"}

        if dims == 2:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=labels, palette=color_map)
            plt.title(f"{method.upper()} 2D Visualization of Outliers")
            plt.show()
        elif dims == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            for label in set(labels):
                idx = [i for i, l in enumerate(labels) if l == label]
                ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], X_reduced[idx, 2],
                           label=label, color=color_map[label])
            ax.set_title(f"{method.upper()} 3D Visualization of Outliers")
            ax.legend()
            plt.show()
        else:
            raise ValueError("Only 2D and 3D visualizations are supported")


#### Shap Class
class OutlierExplainerShap:
    def __init__(self, method='isolation_forest', contamination=0.01, random_state=42, **kwargs):
        supported_methods = ['isolation_forest', 'pca', 'ocsvm', 'lof']
        assert method in supported_methods, f"Method must be one of {supported_methods}"

        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.kwargs = kwargs

        self.scaler = StandardScaler()
        self.X_train = None 
        self.X_scaled = None

        self.model = None
        self.explainer = None
        self.pca = None
        self.outlier_indices = None
        self.reconstruction_errors = None

    def fit(self, X):
        self.X_train = X.copy()
        self.X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        if self.method == 'isolation_forest':
            self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state, **self.kwargs)
            self.model.fit(self.X_scaled)
            self.explainer = shap.Explainer(self.model, self.X_scaled)
            preds = self.model.predict(self.X_scaled)
            self.outlier_indices = np.where(preds == -1)[0]

        elif self.method == 'pca':
            self.pca = PCA(n_components=0.9, random_state=self.random_state)
            X_proj = self.pca.fit_transform(self.X_scaled)
            X_reconstructed = self.pca.inverse_transform(X_proj)
            errors = np.mean((self.X_scaled - X_reconstructed) ** 2, axis=1)
            threshold = np.quantile(errors, 1 - self.contamination)
            self.reconstruction_errors = errors
            self.outlier_indices = np.where(errors >= threshold)[0]

        elif self.method == 'ocsvm':
            self.model = OneClassSVM(**self.kwargs)
            self.model.fit(self.X_scaled)
            preds = self.model.predict(self.X_scaled)
            self.outlier_indices = np.where(preds == -1)[0]

        elif self.method == 'lof':
            self.model = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination, **self.kwargs)
            preds = self.model.fit_predict(self.X_scaled)
            self.outlier_indices = np.where(preds == -1)[0]

    def detect_outliers(self):
        return self.outlier_indices

    def explain_outlier(self, index, visualize=False, top_n=None):
        index = int(index)
        instance_scaled = self.X_scaled.iloc[[index]]
        original_instance = self.scaler.inverse_transform(instance_scaled)[0]

        if self.method == 'isolation_forest':
            shap_values = self.explainer(instance_scaled)
            if visualize:
                shap.plots.waterfall(shap_values[0])
            shap_df = pd.DataFrame({
                "feature": self.X_scaled.columns,
                "shap_value": shap_values.values[0],
                "feature_scaled_value": instance_scaled.values[0],
                "original_value": original_instance
            }).sort_values(by="shap_value", key=abs, ascending=False)

        elif self.method == 'pca':
            reconstruction_error = np.abs(self.X_scaled.iloc[index] - self.pca.inverse_transform(self.pca.transform(instance_scaled))[0])
            shap_df = pd.DataFrame({
                "feature": self.X_scaled.columns,
                "reconstruction_error": reconstruction_error,
                "feature_value": instance_scaled.values[0],
                "original_value": original_instance
            }).sort_values(by="reconstruction_error", ascending=False)

        else:
            shap_df = pd.DataFrame({
                "feature": self.X_scaled.columns,
                "feature_value": instance_scaled.values[0],
                "original_value": original_instance
            })

        if top_n is not None:
            shap_df = shap_df.head(top_n)

        return shap_df

    def explain_all_outliers_(self, top_n=5):
        rows = []
        for idx in self.outlier_indices:
            df = self.explain_outlier(int(idx), visualize=False, top_n=top_n).copy()
            df.insert(0, "outlier_index", idx)
            rows.append(df)
        return pd.concat(rows, ignore_index=True)

    def score(self):
        if self.method == 'isolation_forest':
            return self.model.decision_function(self.X_scaled)
        elif self.method == 'pca':
            return -self.reconstruction_errors
        elif self.method == 'ocsvm':
            return -self.model.decision_function(self.X_scaled)

    def visualize(self, dimension=2, method='pca'):
        if method == 'pca':
            reducer = PCA(n_components=dimension)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=dimension, random_state=self.random_state)
        else:
            raise ValueError("Method must be 'pca' or 'umap'")

        embedding = reducer.fit_transform(self.X_scaled)
        is_outlier = np.zeros(self.X_scaled.shape[0])
        if self.outlier_indices is not None:
            is_outlier[self.outlier_indices] = 1

        fig = plt.figure(figsize=(8, 6))
        if dimension == 2:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=is_outlier, cmap='coolwarm', edgecolor='k')
            plt.title(f"Outlier Visualization (2D - {method.upper()})")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        elif dimension == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=is_outlier, cmap='coolwarm')
            ax.set_title(f"Outlier Visualization (3D - {method.upper()})")
        else:
            raise ValueError("Only 2D or 3D visualizations are supported")

        plt.show()