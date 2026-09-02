from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os

# =========================================================
# FOOD DELIVERY TIME PREDICTION - FINAL FLASK APP
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "food_delivery_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")

app = Flask(
    __name__,
    static_folder="ui",
    static_url_path="/ui"
)

# ---------------------------------------------------------
# LOAD SAVED MODEL + SAVED PREPROCESSOR
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("=" * 60)
print("FOOD DELIVERY PREDICTION SERVER")
print("=" * 60)
print("MODEL FILE:", MODEL_PATH)
print("PREPROCESSOR FILE:", PREPROCESSOR_PATH)
print("Model loaded successfully")
print("Preprocessor loaded successfully")
print("=" * 60)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return send_from_directory(
        os.path.join(BASE_DIR, "ui"),
        "index.html"
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "No JSON input received."
            }), 400

        print("\n" + "-" * 60)
        print("NEW PREDICTION REQUEST")
        print("-" * 60)
        print("Traffic received:", data.get("Road_traffic_density"))

        # -------------------------------------------------
        # CLEAN VALUES COMING FROM THE UI
        # -------------------------------------------------

        traffic = str(
            data.get("Road_traffic_density", "Low")
        ).strip()

        # The saved preprocessor uses these traffic labels.
        traffic_map = {
            "Low": "Low",
            "Medium": "Medium",
            "Moderate": "Medium",
            "High": "High",
            "Jam": "Jam"
        }

        traffic = traffic_map.get(traffic, traffic)

        data["Road_traffic_density"] = traffic

        # -------------------------------------------------
        # IMPORTANT:
        # Your SAVED PREPROCESSOR was trained with these
        # columns. The UI does not need to display all of
        # them, so we supply safe defaults for the fields
        # that are not visible in the UI.
        # -------------------------------------------------

        defaults = {
            "ID": np.nan,
            "Delivery_person_ID": np.nan,
            "Order_Date": np.nan,
            "Time_Orderd": np.nan,
            "Time_Order_picked": np.nan,

            # Representative valid coordinates from the
            # training data. The saved model expects these
            # columns even though the current UI uses
            # Distance_km instead.
            "Restaurant_latitude": 18.520016,
            "Restaurant_longitude": 73.830547,
            "Delivery_location_latitude": 18.530016,
            "Delivery_location_longitude": 73.840547
        }

        for column, default_value in defaults.items():
            if column not in data or data[column] in ("", None):
                data[column] = default_value

        # -------------------------------------------------
        # BUILD EXACT DATAFRAME EXPECTED BY PREPROCESSOR
        # -------------------------------------------------

        required_columns = [
            "ID",
            "Delivery_person_ID",
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
            "Order_Date",
            "Time_Orderd",
            "Time_Order_picked",
            "Weatherconditions",
            "Road_traffic_density",
            "Vehicle_condition",
            "Type_of_order",
            "Type_of_vehicle",
            "multiple_deliveries",
            "Festival",
            "City"
        ]

        # Create missing fields before selecting columns.
        for column in required_columns:
            if column not in data:
                data[column] = np.nan

        sample = pd.DataFrame(
            [{column: data[column] for column in required_columns}]
        )

        # Convert numeric columns safely.
        numeric_columns = [
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
            "Vehicle_condition"
        ]

        for column in numeric_columns:
            sample[column] = pd.to_numeric(
                sample[column],
                errors="coerce"
            )

        # Clean categorical values.
        categorical_columns = [
            "Weatherconditions",
            "Road_traffic_density",
            "Type_of_order",
            "Type_of_vehicle",
            "multiple_deliveries",
            "Festival",
            "City"
        ]

        for column in categorical_columns:
            if column in sample.columns:
                sample[column] = sample[column].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )

        # Normalize UI labels to the exact labels used during training.
        weather_map = {
            "Rainy": "Stormy"
        }
        vehicle_map = {
            "Motorcycle": "motorcycle",
            "Scooter": "scooter",
            "Electric Scooter": "electric_scooter",
            "Bicycle": "bicycle"
        }
        city_map = {
            "Metropolitan": "Metropolitian"
        }

        sample["Weatherconditions"] = sample["Weatherconditions"].replace(weather_map)
        sample["Type_of_vehicle"] = sample["Type_of_vehicle"].replace(vehicle_map)
        sample["City"] = sample["City"].replace(city_map)

        print("Traffic sent to saved preprocessor:",
              sample["Road_traffic_density"].iloc[0])

        # -------------------------------------------------
        # TRANSFORM WITH THE SAVED PREPROCESSOR
        # -------------------------------------------------

        encoded = preprocessor.transform(sample)

        print("Encoded shape:", encoded.shape)

        # -------------------------------------------------
        # PREDICT WITH THE SAVED RANDOM FOREST
        # -------------------------------------------------

        prediction = float(
            model.predict(encoded)[0]
        )

        prediction = round(prediction, 2)

        print(
            "FINAL PREDICTION:",
            traffic,
            "->",
            prediction,
            "minutes"
        )

        # -------------------------------------------------
        # RETURN TO FRONTEND
        # -------------------------------------------------

        return jsonify({
            "predicted_delivery_time": prediction,
            "traffic": traffic
        })

    except Exception as e:

        print("\nPREDICTION ERROR:")
        print(type(e).__name__, ":", str(e))

        return jsonify({
            "error": str(e)
        }), 400


# =========================================================
# MODEL PERFORMANCE
# =========================================================

@app.route("/model-performance")
def model_performance():

    return jsonify({
        "selected_model": "Random Forest Regressor",

        "metrics": {
            "mae": 4.335,
            "mse": 30.16,
            "rmse": 5.4918,
            "r2": 0.6499,
            "r2_percent": 64.99
        },

        "comparison": [
            {
                "model": "Linear Regression",
                "r2": -0.6208,
                "r2_percent": -62.08
            },
            {
                "model": "Decision Tree",
                "r2": 0.6748,
                "r2_percent": 67.48
            },
            {
                "model": "Random Forest",
                "r2": 0.6499,
                "r2_percent": 64.99
            }
        ]
    })


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model": "Random Forest Regressor",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
