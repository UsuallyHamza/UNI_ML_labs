"""
Lab 6 Starter Code
K-Means From Scratch

This code provides a skeleton for implementing K-Means clustering.
Students are expected to complete the TODO sections.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs

# ============================================================
# DATASET LOADERS
# ============================================================

def load_iris_dataset():
    """
    Load the Iris dataset.

    Tasks:
    - Select only two features: Sepal Length and Petal Length
      as specified in the lab (for 2D visualization).
    - Return the data as a NumPy array X of shape (150, 2).
    """
    data = load_iris()

    # TODO: select two features
    X = data.data[:, [0, 2]]

    return X


def generate_blobs(separable=True):
    """
    Generate synthetic clusters using sklearn.make_blobs.

    Parameters:
    - separable: if True, generate well-separated clusters
                 if False, generate overlapping clusters

    Tasks:
    - Create a dataset X with multiple clusters.
    - This allows observing K-Means behavior on different
      cluster distributions.
    """
    if separable:
        # TODO: generate well-separated blobs
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    else:
        # TODO: generate overlapping blobs
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=42)

    return X


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def centroid_shift(old_centroids, new_centroids):
    """
    Compute how much centroids moved after an update.

    Parameters:
    - old_centroids: previous centroids
    - new_centroids: updated centroids

    Tasks:
    - Compute L2 norm of the difference between old and new centroids.
    - Used as a convergence criterion in K-Means.
    """
    # TODO implement actual computation
    # Example: np.linalg.norm(old_centroids - new_centroids)
    # L2 norm of the difference vector gives us the absolute shift distance.
    return np.linalg.norm(old_centroids - new_centroids)


# ============================================================
# KMEANS CLASS
# ============================================================

class MyKMeans:
    """
    K-Means clustering class.

    Tasks:
    - Implement K-Means algorithm from scratch.
    - Experiment with different distance metrics: Euclidean, Manhattan, Cosine.
    - Implement cluster assignment, centroid update, and convergence check.
    """

    # I have inserted 'tol' and 'random_state' to prevent the fit() method from crashing
    def __init__(self, k=3, metric="euclidean", max_iter=100, tol=1e-4, random_state=42):
        """
        Initialize K-Means parameters.

        Parameters:
        - k: number of clusters
        - metric: distance metric to use
        - max_iter: maximum iterations before stopping
        """
        self.k = k
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    # --------------------------------------------------------
    # DISTANCE FUNCTION
    # --------------------------------------------------------

    def _distance(self, x1, x2):
        """
        Compute distance between two points.

        Tasks:
        - Implement three distance metrics:
          * Euclidean: circular clusters
          * Manhattan: diamond-shaped clusters
          * Cosine: measures angle between vectors
        - Hint for cosine: distance = 1 - (dot(x1,x2)/(||x1||*||x2||))
        """
        if self.metric == "euclidean":
            # Produces circular cluster boundaries
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.metric == "manhattan":
            # Produces diamond-shaped cluster boundaries
            return np.sum(np.abs(x1 - x2))
        elif self.metric == "cosine":
            # Measures the angle between two vectors
            norm1 = np.linalg.norm(x1)
            norm2 = np.linalg.norm(x2)
            if norm1 == 0 or norm2 == 0:
                return 1.0 
            similarity = np.dot(x1, x2) / (norm1 * norm2)
            return 1.0 - similarity 
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # --------------------------------------------------------

    def initialize_centroids(self, X):
        """
        Randomly select K points from dataset as initial centroids.

        Tasks:
        - Pick k random indices from X
        - Assign these points as initial centroids
        """
        # We sample without replacement to ensure we don't pick the exact same point twice.
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    # --------------------------------------------------------

    def assign_clusters(self, X, centroids):
        """
        Assign each point to the nearest centroid.

        Tasks:
        - Compute distance from each point to every centroid
        - Assign label based on nearest centroid
        - Return array of cluster labels
        """
        labels = []
        for x in X:
            distances = [self._distance(x, c) for c in centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    # --------------------------------------------------------

    def update_centroids(self, X, labels):
        """
        Update centroid positions based on cluster assignments.

        Tasks:
        - For each cluster, compute mean of points assigned to it
        - If a cluster has no points (empty), randomly reinitialize
          its centroid (safeguard)
        """
        centroids = []

        for i in range(self.k):
            # TODO: get points belonging to cluster i
            cluster_points = X[labels == i]

            if len(cluster_points) == 0:
                # SAFEGUARD: choose random point as centroid
                # SAFEGUARD: The opportunity cost of a dead cluster is wasted computational space.
                # If a cluster is starved, we resurrect it at a random data point.
                centroid = X[np.random.choice(X.shape[0])]
            else:
                # TODO: compute mean of cluster points
                centroid = np.mean(cluster_points, axis=0)

            centroids.append(centroid)

        return np.array(centroids)

    # --------------------------------------------------------

    def fit(self, X):
        """
        Fit the K-Means model to the data X.

        Algorithm:
        1. Initialize centroids
        2. Assign points to nearest centroid
        3. Update centroids
        4. Repeat until convergence (centroid movement < tol)
        or max iterations reached

        Tasks:
        - Implement K-Means training loop
        - Use centroid_shift to detect convergence
        """
        np.random.seed(self.random_state)
        centroids = self.initialize_centroids(X)
        # Dataset mein se randomly $K$ points ko initial "Centroids" (clusters ke centers) maan liya

        for iteration in range(self.max_iter):
            labels = self.assign_clusters(X, centroids)
            # Har data point ka fasla in centroids se nikala (assign_clusters) aur use sabse qareeb wale centroid ke hawale kar diya
            new_centroids = self.update_centroids(X, labels)
            # Un assigned points ka mean (average) nikal kar centroid ki nayi position set ki (update_centroi)

            # TODO compute centroid movement
            shift = centroid_shift(centroids, new_centroids)
            # Yeh process tab tak repeat kiya jab tak centroids ne move hona band nahi kar diya (convergence)

            if shift < self.tol:
                break

            centroids = new_centroids

        self.centroids = centroids

    # --------------------------------------------------------

    def predict(self, X):
        """
        Predict cluster labels for new data points.

        Tasks:
        - Compute distance from each point to centroids
        - Assign each point to nearest centroid
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before calling predict.")
        return self.assign_clusters(X, self.centroids)

    # --------------------------------------------------------
    # INERTIA
    # --------------------------------------------------------

    def compute_inertia(self, X, labels, centroids):
        """
        Compute Within-Cluster Sum of Squares (WCSS / Inertia).

        Tasks:
        - For each point, compute squared distance to its centroid
        - Sum all squared distances
        - Lower values indicate tighter clusters
        """
        # Sum of Squared Errors (SSE) or Inertia
        # Lower values indicate tighter clusters
        inertia = 0
        for i, x in enumerate(X):
            centroid = centroids[labels[i]]
            # Inertia is strictly squared Euclidean distance
            inertia += np.sum((x - centroid)**2) 
        return inertia

# ============================================================
# VISUALIZATION
# ============================================================

def plot_clusters(X, labels, centroids, title):
    """
    Scatter plot of clusters.

    Tasks:
    - Color points based on cluster labels
    - Plot centroids as 'X' markers
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', edgecolor='k', alpha=0.7)

    if centroids is not None:
        plt.scatter(
            centroids[:,0],
            centroids[:,1],
            marker='X',
            s=200,
            c='red',
            edgecolor='white'
        )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.5)


def plot_decision_boundaries(model, X, title_suffix=""):
    """
    Visualize clustering regions (decision boundaries).

    Tasks:
    - Create a mesh grid over the feature space
    - Predict cluster label for each grid point
    - Plot colored decision regions
    - Useful for comparing distance metrics
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a dense grid to probe the model's spatial assignments
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # FIX: Dynamically predict labels instead of relying on non-existent model.labels_
    labels = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker='X', s=200, c='red', edgecolor='white')
    
    plt.title(f"Decision Boundaries: {model.metric.capitalize()} {title_suffix}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

# ============================================================
# ELBOW METHOD
# ============================================================

def plot_elbow_curve(X, k_values):
    """
    Elbow method to determine optimal K.

    Tasks:
    - Train K-Means for multiple K values
    - Compute inertia for each K
    - Plot inertia vs K
    - Observe the 'elbow' to estimate optimal clusters
    """
    inertias = []

    for k in k_values:
        model = MyKMeans(k=k)
        model.fit(X)
        labels = model.predict(X)

        # TODO compute inertia
        inertia = model.compute_inertia(X, labels, model.centroids)
        inertias.append(inertia)

    # TODO: plot inertia vs k
    plt.figure()
    plt.plot(k_values, inertias, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method For Optimal K')
    plt.grid(True)
    plt.show()

# ============================================================
# MAIN DEMO
# ============================================================

def main():
    """
    Main demonstration.

    Tasks:
    - Load Iris dataset
    - Run K-Means with different metrics: Euclidean, Manhattan, Cosine
    - Visualize resulting clusters
    - Run Elbow method to determine optimal K
    """
    print("Loading datasets...")
    X_iris = load_iris_dataset()
    X_separable = generate_blobs(separable=True)
    X_overlapping = generate_blobs(separable=False)

    # 1. Compare Distance Metrics on Iris (Part E & F)
    print("Generating Decision Boundaries for Iris Dataset...")
    metrics = ["euclidean", "manhattan", "cosine"]
    for metric in metrics:
        model = MyKMeans(k=3, metric=metric)
        model.fit(X_iris)
        plot_decision_boundaries(model, X_iris, title_suffix="(Iris)")

    # 2. Compare Separable vs Overlapping Blobs (Part F)
    print("Evaluating Euclidean on Synthetic Blobs...")
    model_sep = MyKMeans(k=3, metric="euclidean")
    model_sep.fit(X_separable)
    plot_decision_boundaries(model_sep, X_separable, title_suffix="(Separable Blobs)")

    model_over = MyKMeans(k=3, metric="euclidean")
    model_over.fit(X_overlapping)
    plot_decision_boundaries(model_over, X_overlapping, title_suffix="(Overlapping Blobs)")

    # 3. Elbow Method (Part G)
    print("Generating Elbow Curve...")
    plot_elbow_curve(X_iris, range(1, 10))

    # Render all generated plots
    print("Displaying plots. Close the windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()