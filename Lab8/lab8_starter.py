import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# ==========================================================
# DATASET (PROVIDED)
# ==========================================================

def load_fraud_data():
    """
    Generates a synthetic dataset for fraud detection.

    The dataset contains two features:
    - Transaction Amount
    - Time Gap Since Last Transaction

    The data consists of:
    - Normally distributed points representing legitimate transactions
    - Uniformly distributed outliers representing fraudulent transactions

    Returns:
    --------
    X : np.ndarray of shape (N, 2)
        Feature matrix

    y : np.ndarray of shape (N,)
        Labels where:
        1  -> Normal transaction
        -1 -> Fraudulent transaction
    """

    np.random.seed(42)

    normal = np.random.multivariate_normal(
        mean=[50, 30],
        cov=[[100, 20],[20, 50]],
        size=300
    )

    fraud = np.random.uniform(
        low=[0, 0],
        high=[120, 100],
        size=(40,2)
    )

    X = np.vstack([normal, fraud])

    y = np.ones(len(X))
    y[-40:] = -1

    return X, y


# ==========================================================
# ISOLATION FOREST
# ==========================================================

def fit_isolation_forest(X):
    """
    Trains an Isolation Forest model on the given dataset.

    The model should learn the structure of the data and identify
    patterns corresponding to normal behavior. It uses random
    partitioning to isolate anomalies.

    Parameters:
    -----------
    X : np.ndarray of shape (N, 2)
        Input feature matrix

    Returns:
    --------
    model :
        Trained Isolation Forest model
    """
    model = IsolationForest(contamination=0.12,random_state=42)
    model.fit(X)
    return model


def predict_iforest(model, X):
    """
    Predicts whether each data point is normal or anomalous
    using a trained Isolation Forest model.

    Parameters:
    -----------
    model :
        Trained Isolation Forest model

    X : np.ndarray of shape (N, 2)
        Input feature matrix

    Returns:
    --------
    predictions : np.ndarray of shape (N,)
        Predicted labels:
        1  -> Normal
        -1 -> Anomaly
    """
    return model.predict(X)


def compute_iforest_scores(model, X):
    """
    Computes anomaly scores for each data point.

    These scores reflect how easily a point is isolated
    by the model. Points that are easier to isolate are
    more likely to be anomalies.

    Parameters:
    -----------
    model :
        Trained Isolation Forest model

    X : np.ndarray of shape (N, 2)

    Returns:
    --------
    scores : np.ndarray of shape (N,)
        Anomaly scores for each data point
    """
    # In sklearn, decision_function returns the anomaly score. 
    # Lower scores indicate higher likelihood of being an anomaly.
    return model.decision_function(X)


# ==========================================================
# GAUSSIAN (FROM SCRATCH)
# ==========================================================

def compute_gaussian_params(X):
    """
    Estimates the parameters of a multivariate Gaussian distribution.

    The parameters include:
    - Mean vector representing the center of the data
    - Covariance matrix representing the spread and relationship
      between features

    Parameters:
    -----------
    X : np.ndarray of shape (N, 2)

    Returns:
    --------
    mu : np.ndarray of shape (2,)
        Mean vector

    sigma : np.ndarray of shape (2, 2)
        Covariance matrix
    """
    mu= np.mean(X, axis=0)
    # rowvar=False ensures columns are treated as variables (features)
    sigma= np.cov(X,rowvar=False)
    return mu,sigma



def gaussian_density(X, mu, sigma):
    """
    Computes the probability density of each data point under
    a multivariate Gaussian distribution.

    Points closer to the mean will have higher density values,
    while points far from the mean will have lower values.

    Parameters:
    -----------
    X : np.ndarray of shape (N, 2)

    mu : np.ndarray of shape (2,)
        Mean vector

    sigma : np.ndarray of shape (2, 2)
        Covariance matrix

    Returns:
    --------
    densities : np.ndarray of shape (N,)
        Probability density values for each data point
    """
    d= X.shape[1]
    det_sigma= np.linalg.det(sigma)
    inv_sigma= np.linalg.inv(sigma)
    coeff= 1.0/np.sqrt(((2*np.pi)**d)*det_sigma)
    diff= X-mu
    exponent= -0.5*np.sum((diff @ inv_sigma)*diff, axis=1)

    return coeff* np.exp(exponent)


def predict_gaussian(X, mu, sigma, threshold):
    """
    Classifies data points as normal or anomalous based on
    their probability density under the Gaussian model.

    Points with density below the specified threshold are
    considered anomalies.

    Parameters:
    -----------
    X : np.ndarray of shape (N, 2)

    mu : np.ndarray
        Mean vector

    sigma : np.ndarray
        Covariance matrix

    threshold : float
        Decision threshold for anomaly detection

    Returns:
    --------
    predictions : np.ndarray of shape (N,)
        Predicted labels:
        1  -> Normal
        -1 -> Anomaly
    """
    densities= gaussian_density(X, mu, sigma)
    predictions= np.ones(len(X))
    predictions[densities<threshold]=-1
    return predictions


# ==========================================================
# VISUALIZATION
# ==========================================================

def plot_points(X, labels, title):
    """
    Plots the dataset in a 2D feature space.

    Points are visualized based on their labels, allowing
    comparison between normal and anomalous transactions.

    Parameters:
    -----------
    X : np.ndarray of shape (N, 2)

    labels : np.ndarray of shape (N,)
        Labels for each point

    title : str
        Title of the plot
    """
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='coolwarm', edgecolors='k')
    plt.title(title)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Time gap since last transaction")


def plot_decision_boundary(model_or_fn, X, method):
    """
    Visualizes the decision regions of the anomaly detection method.

    The plot should show how the model separates normal data
    from anomalous data across the feature space.

    Parameters:
    -----------
    model_or_fn :
        Trained model or function used for prediction

    X : np.ndarray of shape (N, 2)

    method : str
        Identifier for the method ("iforest" or "gaussian")

    Returns:
    --------
    None
    """
    x_min, x_max= X[:,0].min()-10, X[:,0].max()+10
    y_min, y_max= X[:,1].min()-10, X[:,1].max()+10
    
    xx, yy= np.meshgrid(np.linspace(x_min,x_max,100),
                        np.linspace(y_min,y_max,100))
    
    grid= np.c_[xx.ravel(), yy.ravel()]

    if method=="iforest":
        Z=model_or_fn.predict(grid)
    elif method=="gaussian":
        Z=model_or_fn(grid)
    else:
        raise ValueError("invalid method, use iforest or gaussian")
    
    Z=Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c='k', marker='.', s=20)
    plt.title(f"Decision Boundary ({method.upper()})")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Time Gap Since Last Transaction")


if __name__ == "__main__":
    print("Loading data...")
    X, y_true = load_fraud_data()

    # --- Isolation Forest Execution ---
    print("Training Isolation Forest...")
    if_model = fit_isolation_forest(X)
    
    # --- Gaussian Density Execution ---
    print("Computing Gaussian Parameters...")
    mu, sigma = compute_gaussian_params(X)
    
    # --- Visualization ---
    print("Generating visualizations...")
    plt.figure(figsize=(14, 6))

    # Plot 1: Isolation Forest Boundary
    plt.subplot(1, 2, 1)
    plot_decision_boundary(if_model, X, method="iforest")

    # Plot 2: Gaussian Density Boundary
    plt.subplot(1, 2, 2)
    # The visualization function expects a callable for the Gaussian method that only takes X
    gaussian_callable = lambda grid: predict_gaussian(grid, mu, sigma, threshold=0.0005)
    plot_decision_boundary(gaussian_callable, X, method="gaussian")

    plt.tight_layout()
    plt.show()



length=len(arr)
for (i in range arr):
    sum=sum+arr[i]

mu=sum/length

