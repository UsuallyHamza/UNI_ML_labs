"""
Lab 6 Starter Code
Decision Tree Split using Gini and Information Gain
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ============================================================
# DATASET
# ============================================================

def load_binary_iris():
    """
    Load iris dataset but keep only two classes.

    This simplifies the task to binary classification
    so that Gini impurity is easier to compute.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    data = load_iris()

    X = data.data
    y = data.target

    mask = y < 2
    X = X[mask]
    y = y[mask]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


# ============================================================
# STUDENT TASKS
# ============================================================

def gini(y):
    """
    TODO

    Compute Gini impurity.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    float
    """

    if len(y) == 0:
        return 0.0
    
    #compute the probability of class 1
    p1 = np.sum(y) / len(y)
    # compute the probability of class 0
    p0 = 1.0 - p1
    
    # apply the Gini formula
    return 1.0 - (p0**2 + p1**2)


def information_gain(parent, left, right):
    """
    TODO

    Compute Information Gain.

    Parameters
    ----------
    parent : labels before split
    left : labels in left child
    right : labels in right child

    Returns
    -------
    float
    """

    n = len(parent)
    if n == 0:
        return 0.0
    
    # Calculate the weighted average of child impurities
    weighted_child_impurity = (len(left) / n) * gini(left) + (len(right) / n) * gini(right)
    
    # Subtract from parent impurity
    return gini(parent) - weighted_child_impurity


def find_best_split(X, y):
    """
    TODO

    Implement decision tree split search.

    Steps
    -----
    1. Loop over features
    2. Generate candidate thresholds
    3. Split dataset
    4. Compute information gain
    5. Track best split

    Returns
    -------
    best_feature
    best_threshold
    best_gain
    """

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    n_samples, n_features = X.shape

    # loop over features
    for feature in range(n_features):
        # generate candidate thresholds
        thresholds = candidate_thresholds(X[:, feature])
        
        for threshold in thresholds:
            #split dataset using the helper function
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature, threshold)
            
            # skip invalid splits
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            #compute information gain
            gain = information_gain(y, y_left, y_right)
            
            #track best split
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def split_dataset(X, y, feature, threshold):
    """
    Split dataset into left and right partitions.

    All samples with feature value <= threshold
    go to the left node.

    Returns
    -------
    X_left, X_right, y_left, y_right
    """

    mask = X[:, feature] <= threshold

    return X[mask], X[~mask], y[mask], y[~mask]


def candidate_thresholds(values):
    """
    Compute candidate thresholds for a feature.

    Thresholds are chosen as midpoints between
    consecutive sorted unique values.

    This mirrors the approach used in most
    decision tree implementations.
    """

    values = np.sort(np.unique(values))

    return (values[:-1] + values[1:]) / 2


def add_high_cardinality_feature(X):
    """
    Add deterministic high-cardinality feature.

    Each sample receives a unique value derived
    from its index.

    This allows us to observe the bias of
    decision trees toward features with many
    unique values.
    """

    N = len(X)

    index_feature = (np.arange(N) / N).reshape(-1,1)

    return np.hstack([X, index_feature])


# ============================================================
# EXPERIMENT
# ============================================================

def run_split_experiment(X, y):

    feature, threshold, gain = find_best_split(X, y)

    print("Best Feature:", feature)
    print("Threshold:", threshold)
    print("Information Gain:", gain)


# ============================================================
# MAIN
# ============================================================

def main():

    X_train, X_test, y_train, y_test = load_binary_iris()

    print("Original Dataset")

    run_split_experiment(X_train, y_train)

    print("\nWith High Cardinality Feature")

    X_aug = add_high_cardinality_feature(X_train)

    run_split_experiment(X_aug, y_train)


if __name__ == "__main__":
    main()