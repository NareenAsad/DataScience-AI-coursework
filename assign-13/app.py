from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return "House Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Extract features
        grlivarea = float(data["GrLivArea"])
        overallqual = int(data["OverallQual"])
        garagecars = int(data["GarageCars"])

        features = np.array([[grlivarea, overallqual, garagecars]])

        prediction = model.predict(features)[0]

        return jsonify({
            "Predicted_SalePrice": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
