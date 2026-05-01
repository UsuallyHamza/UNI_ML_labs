"""
Lab 7 Starter Code
Agglomerative Clustering from Scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram

# ============================================================
# DATASET
# ============================================================

def generate_dataset():

    X, _ = make_blobs(
        n_samples=100,
        centers=3,
        cluster_std=1.2,
        random_state=42
    )

    return X


# ============================================================
# PART A
# ============================================================

def compute_distance_matrix(X):

    # TODO
    """
    Computes the NxN pairwise Euclidean distance matrix.
    Using numpy broadcasting for performance instead of slow nested loops.
    """
    # X[:, np.newaxis, :] reshapes to (N, 1, D)
    # X[np.newaxis, :, :] reshapes to (1, N, D)
    # The difference broadcasts to (N, N, D), and we take the norm along the D axis.
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


# ============================================================
# PART B
# ============================================================

def linkage_distance(cluster1, cluster2, X, method="single"):

    # TODO
    """
    Computes the distance between two clusters based on the linkage method.
    """
    pts1 = X[cluster1]
    pts2 = X[cluster2]
    
    # Pairwise distances between all points in cluster1 and cluster2
    diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    if method == "single":
        return np.min(distances)
    elif method == "complete":
        return np.max(distances)
    elif method == "average":
        return np.mean(distances)
    else:
        raise ValueError(f"Unknown linkage method: {method}")


# ============================================================
# PART C
# ============================================================

def find_closest_clusters(clusters, X, method):

    # TODO
    """
    Iterates through all cluster pairs to find the minimum linkage distance.
    """
    min_dist = float('inf')
    closest_pair = (None, None)

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = linkage_distance(clusters[i], clusters[j], X, method)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (i, j)

    return closest_pair[0], closest_pair[1], min_dist


# ============================================================
# MODEL
# ============================================================

class MyAgglomerative:

    def __init__(self, n_clusters=3, linkage="single"):

        self.n_clusters = n_clusters
        self.linkage = linkage

        self.labels_ = None
        self.history_ = []


    def fit(self, X):
        n = len(X)

        clusters = [[i] for i in range(n)]
        cluster_ids = list(range(n))
        next_new_id = n
        
        # Initialize default labels in case n_clusters=1
        self.labels_ = np.zeros(n, dtype=int)

        # We MUST merge all the way down to 1 cluster to build a valid SciPy dendrogram
        while len(clusters) > 1:
            
            # SNAPSHOT: Capture the labels the moment we reach the target number of clusters
            if len(clusters) == self.n_clusters:
                for label, cluster_indices in enumerate(clusters):
                    for idx in cluster_indices:
                        self.labels_[idx] = label

            # Find closest clusters
            i, j, dist = find_closest_clusters(clusters, X, self.linkage)

            # Merge clusters
            new_cluster = clusters[i] + clusters[j]

            # Store merge for the dendrogram in SciPy format
            id1, id2 = cluster_ids[i], cluster_ids[j]
            self.history_.append([id1, id2, dist, len(new_cluster)])

            # Update clusters (delete larger index first)
            if i > j:
                i, j = j, i
                
            del clusters[j]
            del clusters[i]
            del cluster_ids[j]
            del cluster_ids[i]

            clusters.append(new_cluster)
            cluster_ids.append(next_new_id)
            next_new_id += 1


    def predict(self, X):
        return self.labels_


# ============================================================
# VISUALIZATION
# ============================================================

def plot_clusters(X, labels, title):

    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.title(title)


# ============================================================
# DENDROGRAM (BONUS)
# ============================================================

def plot_dendrogram(history):

    # TODO
    """
    Plots the dendrogram using the history tracked during fitting.
    Requires history to match SciPy's linkage matrix format.
    """
    linkage_matrix = np.array(history)
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")


# ============================================================
# MAIN
# ============================================================

def main():

    X = generate_dataset()

    model = MyAgglomerative(n_clusters=3, linkage="single")

    model.fit(X)

    labels = model.predict(X)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_clusters(X, labels, f"Agglomerative ({model.linkage} linkage)")
    
    plt.subplot(1, 2, 2)
    # The dendrogram needs the full history to root. If n_clusters > 1, 
    # the tree is technically "cut off" before a single root. 
    # It will plot the merges that *did* happen.
    plot_dendrogram(model.history_)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()