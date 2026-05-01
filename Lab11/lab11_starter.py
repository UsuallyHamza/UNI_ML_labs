# ============================================================
# Imports
# ============================================================

# Standard data / ML libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import mlflow
from mlflow.sklearn import log_model
import os

# FastAPI framework — HTTPException is available for use in the API section
from fastapi import FastAPI, HTTPException


# ============================================================
# ML Pipeline
# ============================================================

# Accuracy threshold used by tests to validate pipeline quality
expected_accuracy = 0.8


def load_data():
    """
    Load the dataset from the CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe containing the data.
    """
    # TODO
    df = pd.read_csv("lab10_data.csv")  # reading the data file
    return df


def preprocess_data(df):
    """
    Preprocess the input dataframe by cleaning and preparing features and target.

    Args:
        df (pd.DataFrame): The raw dataframe loaded from the CSV.

    Returns:
        tuple: A tuple containing (X, y) where X is the feature matrix and y is the target vector.
    """
    # TODO:
    # 1. Drop the "city" column — it is categorical and not used by the model.
    # 2. Separate the remaining columns into X (features) and y (target).
    #    - X: all columns except "target"
    #    - y: the "target" column
    # 3. Return (X, y).
    #
    # NOTE: if either "city" or "target" is missing from df, pandas will raise
    # a KeyError — consider whether the caller should handle that.
    
    df = df.drop(columns=["city"])  # features (city dropped here)

    X = df.drop(columns=["target"])
    y = df["target"]  # labels
    return X,y


def train_model(X, y):
    """
    Train a machine learning model using the provided features and target.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix.
        y (pd.Series or np.ndarray): The target vector.

    Returns:
        object: The trained model object.
    """
    # TODO
    
    model = DecisionTreeClassifier(random_state=42)  # lots of params removed
    model.fit(X, y)
    
    return model


def evaluate_model(model, X, y):
    """
    Evaluate the trained model on the given data and return the accuracy.

    Args:
        model (object): The trained model.
        X (pd.DataFrame or np.ndarray): The feature matrix for evaluation.
        y (pd.Series or np.ndarray): The true target values.

    Returns:
        float: The accuracy score of the model.
    """
    # TODO

    pred = model.predict(X)

    acc = accuracy_score(y, pred)
    print("Accuracy:", acc)

    print("Model trained successfully")
    print("Predictions made")

    return acc


def run_pipeline():
    """
    Run the complete ML pipeline: load data, preprocess, train, evaluate, and save the model.

    Returns:
        float: The accuracy of the trained model on the training data.
    """
    # TODO:
    # Execute your modular functions
    df= load_data()
    X , y  = preprocess_data(df)

    #Wrap training and evaluation in an MLflow run
    with mlflow.start_run():
        model = train_model(X, y)
        acc = evaluate_model(model, X, y)

        # Log Parameters (Inputs)
        mlflow.log_param("model_type", "DescisionTreeClassifier")
        mlflow.log_param("random_state", 42)

        # Log Metrics (Outputs)
        mlflow.log_metric("accuracy", float(acc))

        # Log the Model (Artifacts)
        log_model(model, "model")

        # Part C Task 4: Serializing the model for my FastAPI backend
        joblib.dump(model, "model.joblib")
        
        return acc




# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI()

# TODO: load model
try:
    model = joblib.load("model.joblib")

except FileNotFoundError:
    model=None
    print("Warning: model.joblib not found. Run the training pipeline first")



@app.get("/")
def home():
    # Returns a simple status message to confirm the API is reachable.
    return {"message": "ML Model API is running"}


@app.post("/predict")
def predict(data: dict):
    """
    Run inference for a single sample.

    Expected JSON body:
        { "features": [age, income, hour, leak_feature] }

    Returns:
        dict: { "prediction": 0 or 1 }
    """
    # TODO:
    # 1. Check that "features" is present in data.
    #    If not, raise an HTTPException with status_code=422 and a descriptive detail message.
    # 2. Convert data["features"] to a numpy array and reshape to (1, -1).
    # 3. Call model.predict(...) and return {"prediction": int(result[0])}.
    #
    # NOTE: if the model is still None (not loaded), calling model.predict will
    # raise an AttributeError — consider handling that with an appropriate
    # HTTPException(status_code=503, detail="Model not loaded").
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if "features" not in data:
        raise HTTPException(status_code=422, detail="Missing features key in payload")
    
    # The Matrix Reshape 
    # The user sends a 1D list: [30, 50000, 8, 0.5]
    # Scikit-Learn demands a 2D matrix: [[30, 50000, 8, 0.5]]
    input_features = np.array(data["features"]).reshape(1, -1)
    # The 1 tells NumPy: "I am giving you exactly 1 row (1 sample)."
    # The -1 tells NumPy: "Calculate the number of columns automatically based on whatever is left over."

    result = model.predict(input_features)

    return {"prediction:": int(result[0])}