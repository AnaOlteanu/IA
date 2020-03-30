import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def compute_y(x, W, bias):
 # dreapta de decizie
 # [x, y] * [W[0], W[1]] + b = 0
    return (-x * W[0] - bias) / (W[1] + 1e-10)

def plot_decision_boundary(X, y , W, b, current_x, current_y):
    x1 = -0.5
    y1 = compute_y(x1, W, b)
    x2 = 0.5
    y2 = compute_y(x2, W, b)
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    color = 'r'
    if(current_y == -1):
        color = 'b'

    plt.ylim((-1, 2))
    plt.xlim((-1, 2))

    plt.plot(X[y == -1, 0], X[y == -1, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')

    # ploteaza exemplul curent
    plt.plot(current_x[0], current_x[1], color+'s')

    # afisarea dreptei de decizie
    plt.plot([x1, x2] ,[y1, y2], 'black')
    plt.show(block=False)
    plt.pause(0.3)

def compute_accuracy(x, y, w, b):
    accuracy = (np.sign(np.dot(x,w) + b) == y).mean()
    return accuracy

def train_perc(x, y, epochs, learning_rate):

    num_features = x.shape[1]
    num_samples = x.shape[0]

    weights = np.zeros(num_features)
    bias = 0

    accuracy = 0.0
    for epoch in range(epochs):
        x, y = shuffle(x, y)
        for i in range(num_samples):
            y_hat = np.dot(x[i][:], weights) + bias
            loss = (y_hat - y[i]) ** 2
            weights -= learning_rate * (y_hat - y[i]) * x[i][:]
            bias -= learning_rate * (y_hat - y[i])
            print(compute_accuracy(x,y,weights,bias))
            plot_decision_boundary(x,y,weights,bias,x[i][:],y[i])

    return weights, bias


# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([-1, 1, 1, 1])
# X =np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
# y = np.array([-1, 1, 1, -1])
e = 70
lr = 0.1

#weights, bias = train_perc(X, y, e, lr)
#print(weights)
#print(bias)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def forward(x, w_1, b_1, w_2, b_2):
    z_1 = np.matmul(x, w_1) + b_1
    a_1 = np.tanh(z_1)
    z_2 = np.matmul(a_1, w_2) + b_2
    a_2 = sigmoid(z_2)
    return z_1, a_1, z_2, a_2


def backward(a_1, a_2, z_1, W_2, X, y, num_samples):
    dz_2 = a_2 - y

    dw_2 = np.matmul(a_1.T, dz_2) / num_samples
    db_2 = np.sum(dz_2, axis=0) / num_samples

    da_1 = np.matmul(dz_2, W_2.T)
    dz_1 = np.multiply(da_1, tanh_derivative(z_1))
    dw_1 = np.matmul(X.T, dz_1) / num_samples
    db_1 = np.sum(dz_1, axis=0) / num_samples

    return dw_1, db_1, dw_2, db_2


def compute_y(x, W, bias):
 # dreapta de decizie
 # [x, y] * [W[0], W[1]] + b = 0
    return (-x*W[0] - bias) / (W[1] + 1e-10)

def plot_decision(X_, W_1, W_2, b_1, b_2):
    # sterge continutul ferestrei
    plt.clf()
    # ploteaza multimea de antrenare
    plt.ylim((-0.5, 1.5))
    plt.xlim((-0.5, 1.5))
    xx = np.random.normal(0, 1, (100000))
    yy = np.random.normal(0, 1, (100000))
    X = np.array([xx, yy]).transpose()
    X = np.concatenate((X, X_))

    _, _, _, output = forward(X, W_1, b_1, W_2, b_2)
    y = np.squeeze(np.round(output))
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'b+')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'r+')
    plt.show(block=False)
    plt.pause(0.1)


X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.expand_dims(np.array([0, 1, 1, 0]), 1)

nr_hidden_neurons = 2
nr_output_neurons = 1

W_1 = np.random.normal(0, 1, (2, nr_hidden_neurons))
b_1 = np.zeros(nr_hidden_neurons)
W_2 = np.random.normal(0, 1, (nr_hidden_neurons, nr_output_neurons))
b_2 = np.zeros(nr_output_neurons)

num_samples = X.shape[0]

num_epochs = 250
lr = 0.5

for epoch_idx in range(num_epochs):
    X, y = shuffle(X, y)

    z_1, a_1, z_2, a_2 = forward(X, W_1, b_1, W_2, b_2)

    loss = (-y * np.log(a_2) - (1 - y) * np.log(1 - a_2)).mean()
    accuracy = (np.round(a_2) == y).mean()
    print("==============")
    print("epoch_idx:", epoch_idx)
    print("loss:",loss)
    print("accuracy:",accuracy)

    plot_decision(X, W_1, W_2, b_1, b_2)

    dw_1, db_1, dw_2, db_2 = backward(a_1, a_2, z_1, W_2, X, y, num_samples)
    W_1 = W_1 - lr * dw_1
    b_1 = b_1 - lr * db_1
    W_2 = W_2 - lr * dw_2
    b_2 = b_2 - lr * db_2

