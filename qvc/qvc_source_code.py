import random as re
import numpy as np
# 2. normalization (Amplitude Encoding)
def amplitude_encode(vector):
    """
    Converts a classical vector into a normalized quantum state.

    In quantum computing, the amplitudes of a state vector
    must satisfy: sum(|amplitude|^2) = 1.
    This function enforces that condition.
    """

    # Compute the squared magnitude (L2 norm squared)
    # Equivalent to: ||vector||^2
    magnitude = sum([num**2 for num in vector])

    # Convert the vector to a NumPy array with complex dtype
    # Quantum amplitudes are complex numbers by definition
    vector = np.array(vector, dtype=np.complex128)

    # Normalize the vector so that sum(|amplitude|^2) = 1
    # This allows the vector to be interpreted as a valid quantum state
    return vector / np.sqrt(magnitude)

def RY(theta):
    """
    Single-qubit rotation around the Y-axis of the Bloch sphere.

    This gate is commonly used in variational circuits because:
    - It introduces trainable parameters (theta)
    - It can move a qubit from |0⟩ into a superposition
    """

    return np.array([
        [ np.cos(theta / 2), -np.sin(theta / 2) ],
        [ np.sin(theta / 2),  np.cos(theta / 2) ]
    ], dtype=np.complex128)


def RZ(theta):
    return np.array([
        [ np.exp(-0.5j * theta), 0 ],
        [ 0, np.exp(0.5j * theta) ]
    ], dtype=np.complex128)


# Controlled-NOT (CNOT) gate
# A 2-qubit entangling gate
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)


# Identity gate for a single qubit
# Used when expanding single-qubit gates to multi-qubit systems
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

def find_probabilities(vector):
    """
    Computes measurement probabilities from a quantum state vector.

    Parameters
    ----------
    vector : numpy array of complex numbers
        Quantum state amplitudes

    Returns
    -------
    numpy array
        Probabilities for each computational basis state
    """

    probs = []

    for amplitude in vector:
        # Quantum probability = |amplitude|^2
        prob = (amplitude.real ** 2) + (amplitude.imag ** 2)
        probs.append(prob)

    probs = np.array(probs)

    # Normalize to guard against numerical errors
    return probs / np.sum(probs)

def predict(vector, theta):
    vector = amplitude_encode(vector)
    state = circuit(theta) @ vector

    exp_val = expectation(state, ZI)  # ∈ [-1, 1]
    prob = (exp_val + 1) / 2           # map to [0, 1]

    pred = 1 if prob > 0.5 else 0
    return pred, prob

import pandas as pd

train = pd.read_csv("/content/drive/MyDrive/ML/MAJOR/DATASET/train.csv")
test = pd.read_csv("/content/drive/MyDrive/ML/MAJOR/DATASET/test.csv")

train.describe()

train.head()

train[:5]

# Bengin = 0
# Attack = 1

train["Label"] = train["Label"].map(
    {"Benign": 0, "Attack": 1}
)

test["Label"] = test["Label"].map(
    {"Benign": 0, "Attack": 1}
)

correct_outputs = [0, 0]
incorrect_outputs = [0, 0]

cosines = [0.3, 0.5, 0.7]

for row in train.itertuples(index=False):
    # row = (PCA1, PCA2, PCA3, PCA4, Label)
    input = [row.PCA_Component_1,
             row.PCA_Component_2,
             row.PCA_Component_3,
             row.PCA_Component_4]

    actual_output = row.Label

    result, confidence = predict(input, cosines)

    if result == actual_output:
        correct_outputs[actual_output] += 1
    else:
        incorrect_outputs[actual_output] += 1

print("Correct:", correct_outputs)
print("Incorrect:", incorrect_outputs)
print(f"Accuracy : {sum(correct_outputs) / len(train)}")


# Extract features and labels
X = train[[
    "PCA_Component_1",
    "PCA_Component_2",
    "PCA_Component_3",
    "PCA_Component_4"
]].values

y = train["Label"].values


import numpy as np

def loss_fn(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-8, 1 - 1e-8)
    return -(y_true*np.log(y_prob) + (1-y_true)*np.log(1-y_prob))


def compute_gradient(X, y, theta):
    grad = np.zeros_like(theta)
    shift = np.pi / 2

    for i in range(len(theta)):
        theta_p = theta.copy()
        theta_m = theta.copy()

        theta_p[i] += shift
        theta_m[i] -= shift

        loss_p, loss_m = 0.0, 0.0

        for x, lbl in zip(X, y):
            _, prob_p = predict(x, theta_p)
            _, prob_m = predict(x, theta_m)

            loss_p += loss_fn(lbl, prob_p)
            loss_m += loss_fn(lbl, prob_m)

        grad[i] = (loss_p - loss_m) / 2

    return grad / len(X)


from sklearn.metrics import accuracy_score

theta = np.random.rand(3)   # variational parameters
lr = 0.5
epochs = 20

for epoch in range(epochs) :
    grad = compute_gradient(X, y, theta)
    theta -= lr * grad

    y_pred = [predict(x, theta)[0] for x in X]
    acc = accuracy_score(y, y_pred)

    print(f"Epoch {epoch+1}/{epochs} | Accuracy={acc:.4f} | theta={theta}")


from numpy.linalg import norm

norms = [norm(x) for x in X[:1000]]
print(min(norms), max(norms))

def evaluate_model(X, y, theta):
    y_pred = []
    y_prob = []

    for x in X:
        pred, prob = predict(x, theta)
        y_pred.append(pred)
        y_prob.append(prob)

    return np.array(y_pred), np.array(y_prob)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

y_train_pred, y_train_prob = evaluate_model(X, y, theta)

print("Train Accuracy :", accuracy_score(y, y_train_pred))
print("Train Precision:", precision_score(y, y_train_pred))
print("Train Recall   :", recall_score(y, y_train_pred))
print("Train F1-score :", f1_score(y, y_train_pred))

print("\nConfusion Matrix (Train):")
print(confusion_matrix(y, y_train_pred))

print("\nClassification Report (Train):")
print(classification_report(y, y_train_pred))

X_test = test[[
    "PCA_Component_1",
    "PCA_Component_2",
    "PCA_Component_3",
    "PCA_Component_4"
]].values

y_test = test["Label"].values

y_test_pred, y_test_prob = evaluate_model(X_test, y_test, theta)

print("Test Accuracy :", accuracy_score(y_test, y_test_pred))
print("Test Precision:", precision_score(y_test, y_test_pred))
print("Test Recall   :", recall_score(y_test, y_test_pred))
print("Test F1-score :", f1_score(y_test, y_test_pred))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def plot_metric_boxes(y_true, y_pred, title="VQC Performance Metrics"):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    for i, (metric, value) in enumerate(metrics.items()):
        ax.text(
            0.125 + i * 0.25,
            0.5,
            f"{metric}\n\n{value:.3f}",
            ha="center",
            va="center",
            fontsize=14,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor=colors[i],
                edgecolor="black"
            )
        )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.show()



import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_box(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=2,
        linecolor="black",
        cbar=False
    )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.show()


def plot_probability_distribution(y_prob):
    plt.figure(figsize=(6, 4))
    plt.hist(y_prob, bins=30, color="#673AB7", edgecolor="black")
    plt.axvline(0.5, color="red", linestyle="--", label="Decision Boundary")

    plt.title("Quantum Output Probability Distribution", fontweight="bold")
    plt.xlabel("P(Attack)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


