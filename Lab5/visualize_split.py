"""
Visualization of Decision Tree Split on Iris Dataset

This script visualizes the best split selected by the
decision tree split algorithm implemented in Lab 5.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import lab5_starter as student


# ============================================================
# LOAD DATA
# ============================================================

def load_binary_iris():

    data = load_iris()

    X = data.data
    y = data.target

    mask = y < 2

    X = X[mask]
    y = y[mask]

    return X, y


# ============================================================
# VISUALIZATION
# ============================================================

def plot_split(X, y, feature, threshold):

    """
    Plot the dataset and the decision boundary.
    """

    plt.figure(figsize=(7,5))

    # Plot class 0
    plt.scatter(
        X[y==0,0],
        X[y==0,2],
        label="Class 0",
        marker="o"
    )

    # Plot class 1
    plt.scatter(
        X[y==1,0],
        X[y==1,2],
        label="Class 1",
        marker="x"
    )

    # Draw decision boundary
    if feature == 0:

        # vertical line
        plt.axvline(
            x=threshold,
            linestyle="--",
            label="Split boundary"
        )

    elif feature == 2:

        # horizontal line
        plt.axhline(
            y=threshold,
            linestyle="--",
            label="Split boundary"
        )

    else:

        print(
            f"Split is on feature {feature}, which is not shown in the plot."
        )

    plt.xlabel("Sepal Length (feature 0)")
    plt.ylabel("Petal Length (feature 2)")

    plt.title("Decision Tree Split Visualization")

    plt.legend()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    X, y = load_binary_iris()

    feature, threshold, gain = student.find_best_split(X, y)

    print("Best Feature:", feature)
    print("Threshold:", threshold)
    print("Information Gain:", gain)

    plot_split(X, y, feature, threshold)


if __name__ == "__main__":

    main()