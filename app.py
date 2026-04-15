from flask import Flask, request, jsonify
import numpy as np
import pickle

from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister
import qiskit

# -----------------------------
# Load trained model
# -----------------------------
with open("qnn_model.pkl", "rb") as f:
    data = pickle.load(f)

theta = data["theta"]
scaler = data["scaler"]

N = 4

# -----------------------------
# Quantum Functions
# -----------------------------
def feature_map(X):
    q = QuantumRegister(N)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q, c)

    for i, x in enumerate(X):
        qc.rx(x, i)

    return qc, c

def variational_circuit(qc, theta):
    for i in range(N - 1):
        qc.cx(i, i + 1)
    qc.cx(N - 1, 0)

    for i in range(N):
        qc.ry(theta[i], i)

    return qc

def quantum_nn(X, theta):
    qc, c = feature_map(X)
    qc = variational_circuit(qc, theta)
    qc.measure(0, c)

    shots = 1000
    backend = Aer.get_backend("qasm_simulator")

    job = qiskit.execute(qc, backend, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    prob = counts.get("1", 0) / shots
    return prob

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "QNN API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Expecting 4 values
        values = [
            data["f1"],
            data["f2"],
            data["f3"],
            data["f4"]
        ]

        # Convert to numpy
        X = np.array(values).reshape(1, -1)

        # Scale (VERY IMPORTANT)
        X_scaled = scaler.transform(X)[0]

        # Predict
        prob = quantum_nn(X_scaled, theta)

        label = "Malicious" if prob >= 0.5 else "Benign"

        return jsonify({
            "prediction": label,
            "probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)