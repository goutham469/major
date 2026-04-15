from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
with open("qvc_model.pkl", "rb") as f:
    theta = pickle.load(f)

# =========================
# QVC FUNCTIONS (same as yours)
# =========================
def amplitude_encode(vector):
    magnitude = sum([num**2 for num in vector])
    vector = np.array(vector, dtype=np.complex128)
    return vector / np.sqrt(magnitude)

def RY(theta):
    return np.array([
        [ np.cos(theta / 2), -np.sin(theta / 2) ],
        [ np.sin(theta / 2),  np.cos(theta / 2) ]
    ], dtype=np.complex128)

def RZ(theta):
    return np.array([
        [ np.exp(-0.5j * theta), 0 ],
        [ 0, np.exp(0.5j * theta) ]
    ], dtype=np.complex128)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)

I2 = np.eye(2, dtype=complex)

def circuit(theta):
    a, b, c = theta
    U = np.kron(RY(a), RY(b))
    U = CNOT @ U
    U = np.kron(RZ(c), RY(a)) @ U
    return U

Z = np.array([[1, 0], [0, -1]], dtype=complex)
ZI = np.kron(Z, I2)

def expectation(state, operator):
    return np.real(np.conjugate(state).T @ operator @ state)

def predict(vector, theta):
    vector = amplitude_encode(vector)
    state = circuit(theta) @ vector

    exp_val = expectation(state, ZI)
    prob = (exp_val + 1) / 2

    pred = 1 if prob > 0.5 else 0
    return pred, prob

# =========================
# API ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json

    try:
        vector = [
            float(data["f1"]),
            float(data["f2"]),
            float(data["f3"]),
            float(data["f4"])
        ]

        pred, prob = predict(vector, theta)

        return jsonify({
            "prediction": int(pred),
            "probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)