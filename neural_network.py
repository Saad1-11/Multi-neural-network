import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score


# =========================
# Initialisation
# =========================

def initialisation(dimensions):

    parametres = {}
    C = len(dimensions)

    for i in range(1, C):

        parametres['W' + str(i)] = np.random.randn(dimensions[i], dimensions[i-1])
        parametres['b' + str(i)] = np.random.randn(dimensions[i], 1)

    return parametres


# =========================
# Forward propagation
# =========================

def forward_propagation(X, parametres):

    activations = {'A0': X}

    C = len(parametres) // 2

    for i in range(1, C+1):

        Z = parametres['W'+str(i)].dot(activations['A'+str(i-1)]) + parametres['b'+str(i)]

        A = 1 / (1 + np.exp(-Z))

        activations['A'+str(i)] = A

    return activations


# =========================
# Log Loss
# =========================

def logloss(A, y):

    return 1/y.shape[1] * np.sum(-y*np.log(A) - (1-y)*np.log(1-A))


# =========================
# Backpropagation
# =========================

def back_propagation(y, activations, parametres):

    m = y.shape[1]
    C = len(parametres) // 2

    dZ = activations['A'+str(C)] - y
    gradients = {}

    for i in reversed(range(1, C+1)):

        gradients['dW'+str(i)] = 1/m * np.dot(dZ, activations['A'+str(i-1)].T)
        gradients['db'+str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dZ = np.dot(parametres['W'+str(i)].T, dZ) * activations['A'+str(i-1)] * (1-activations['A'+str(i-1)])

    return gradients


# =========================
# Mise à jour des poids
# =========================

def update(parametres, gradients, learning_rate):

    C = len(parametres) // 2

    for i in range(1, C+1):

        parametres['W'+str(i)] = parametres['W'+str(i)] - learning_rate * gradients['dW'+str(i)]
        parametres['b'+str(i)] = parametres['b'+str(i)] - learning_rate * gradients['db'+str(i)]

    return parametres


# =========================
# Prediction
# =========================

def predict(X, parametres):

    activations = forward_propagation(X, parametres)

    C = len(parametres) // 2

    A = activations['A'+str(C)]

    return A >= 0.5


# =========================
# Neural Network
# =========================

def neurone_network(X, y, hidden_layers=(32,32,32), learning_rate=0.1, n_iter=1000):

    np.random.seed(0)

    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])

    parametres = initialisation(dimensions)

    train_loss = []
    train_acc = []

    for i in range(n_iter):

        activations = forward_propagation(X, parametres)

        gradients = back_propagation(y, activations, parametres)

        parametres = update(parametres, gradients, learning_rate)

        if i % 10 == 0:

            C = len(parametres) // 2

            loss = logloss(activations['A'+str(C)], y)

            y_pred = predict(X, parametres)

            acc = accuracy_score(y.flatten(), y_pred.flatten())

            train_loss.append(loss)
            train_acc.append(acc)

    # Graphiques
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    ax[0].plot(train_loss)
    ax[0].set_title("Loss")

    ax[1].plot(train_acc)
    ax[1].set_title("Accuracy")

    plt.show()

    return parametres


# =========================
# Dataset
# =========================

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)

X = X.T
y = y.reshape((1, y.shape[0]))

print("dimension X:", X.shape)
print("dimension y:", y.shape)


# =========================
# Train
# =========================

parametres = neurone_network(X, y)