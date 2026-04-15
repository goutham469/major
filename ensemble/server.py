from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

QNN_URL = "http://localhost:5000/predict"
QVC_URL = "http://localhost:5001/predict"

@app.route("/")
def home():
    return "Ensemble API running (QNN + QVC)"

@app.route("/predict", methods=["POST"])
def ensemble_predict():
    try:
        data = request.json

        payload = {
            "f1": data["f1"],
            "f2": data["f2"],
            "f3": data["f3"],
            "f4": data["f4"]
        }

        # -----------------------
        # Call QNN server
        # -----------------------
        qnn_res = requests.post(QNN_URL, json=payload)
        qnn_data = qnn_res.json()

        if "error" in qnn_data:
            return jsonify({"error": "QNN error", "details": qnn_data}), 400

        qnn_prob = float(qnn_data["probability"])
        qnn_pred = qnn_data["prediction"]

        # Convert label to numeric
        qnn_pred_num = 1 if qnn_pred == "Malicious" else 0

        # -----------------------
        # Call QVC server
        # -----------------------
        qvc_res = requests.post(QVC_URL, json=payload)
        qvc_data = qvc_res.json()

        if "error" in qvc_data:
            return jsonify({"error": "QVC error", "details": qvc_data}), 400

        qvc_prob = float(qvc_data["probability"])
        qvc_pred = int(qvc_data["prediction"])

        # -----------------------
        # Ensemble Logic
        # -----------------------
        avg_prob = (qnn_prob + qvc_prob) / 2

        final_pred = 1 if avg_prob > 0.5 else 0
        final_label = "Malicious" if final_pred == 1 else "Benign"

        # -----------------------
        # Response
        # -----------------------
        return jsonify({
            "final_prediction": final_label,
            "final_probability": avg_prob,

            "models": {
                "QNN": {
                    "prediction": qnn_pred,
                    "probability": qnn_prob
                },
                "QVC": {
                    "prediction": qvc_pred,
                    "probability": qvc_prob
                }
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)