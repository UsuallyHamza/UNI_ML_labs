from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load

app = Flask(__name__)

# Load the serialized model
try:
    model = load("model.joblib")
except FileNotFoundError:
    model = None
    print("Warning: model.joblib not found. Train the model first.")

@app.route("/", methods=["GET"])
def home():
    # Renders the UI [cite: 136]
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Handles API requests and form submissions
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Handle JSON (API) or Form Data (Web Interface)
        if request.is_json:
            data = request.get_json()
            features = data.get("features")
        else:
            # Extracting from HTML form
            age = float(request.form["age"])
            income = float(request.form["income"])
            hour = float(request.form["hour"])
            leak_feature = float(request.form["leak_feature"])
            features = [age, income, hour, leak_feature]

        if not features:
            return jsonify({"error": "Missing features"}), 422

        # The Matrix Reshape
        input_features = np.array(features).reshape(1, -1)
        result = model.predict(input_features)

        if request.is_json:
            return jsonify({"prediction": int(result[0])})
        else:
            return render_template("index.html", prediction=int(result[0]), features=features)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)