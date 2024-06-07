import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Forward pass
def hidden_layer_perceptron(x1, x2, x3, w1, w2, w3, w0):
    return sigmoid(w1 * x1 + w2 * x2 + w3 * x3 + w0)


# Forward pass
def output_layer_perceptron(x1, x2, w1, w2, w0):
    return sigmoid(w1 * x1 + w2 * x2 + w0)


def mlp(w11, w12, w13, w10, w21, w22, w23, w20, w1, w2, w0, x1, x2, x3):
    y1 = hidden_layer_perceptron(x1, x2, x3, w11, w12, w13, w10)
    y2 = hidden_layer_perceptron(x1, x2, x3, w21, w22, w23, w20)
    y = output_layer_perceptron(y1, y2, w1, w2, w0)
    return y, y1, y2


# Inputs
x1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
x2 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
x3 = np.array([0, 1, 0, 1, 0, 1, 0, 1])

# Outputs
y1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])  # AND
y2 = np.array([0, 1, 1, 1, 1, 1, 1, 1])  # OR
y3 = np.array([0, 0, 0, 1, 0, 1, 0, 0])  # (x1 XOR x2) AND x3
y_arr = [y1, y2, y3]

w11_t = np.random.normal(-1, 1)
w12_t = np.random.normal(-1, 1)
w13_t = np.random.normal(-1, 1)
w10_t = np.random.normal(-1, 1)
w21_t = np.random.normal(-1, 1)
w22_t = np.random.normal(-1, 1)
w23_t = np.random.normal(-1, 1)
w20_t = np.random.normal(-1, 1)

w1_t = np.random.normal(-1, 1)
w2_t = np.random.normal(-1, 1)
w0_t = np.random.normal(-1, 1)

num_of_epochs = 5000
lr = 0.03 # learning rate
N = len(x1)
i = 1
print("Functions: 1 - AND, 2 - OR , 3 - (x1 XOR x2) AND x3")
for y in y_arr:
    w11_t = np.random.normal(-1, 1)
    w12_t = np.random.normal(-1, 1)
    w13_t = np.random.normal(-1, 1)
    w10_t = np.random.normal(-1, 1)
    w21_t = np.random.normal(-1, 1)
    w22_t = np.random.normal(-1, 1)
    w23_t = np.random.normal(-1, 1)
    w20_t = np.random.normal(-1, 1)

    w1_t = np.random.normal(-1, 1)
    w2_t = np.random.normal(-1, 1)
    w0_t = np.random.normal(-1, 1)
    print()
    print("Function number:", i)
    for e in range(num_of_epochs):
        y_h, y_1, y_2 = mlp(w11_t, w12_t, w13_t, w10_t, w21_t, w22_t, w23_t, w20_t, w1_t, w2_t, w0_t, x1, x2, x3)

        # Backward pass

        nabla_L = 2 * (y - y_h) * -1

        nabla_y_h_y1 = nabla_L * y_h * (1 - y_h) * w1_t
        nabla_y_h_y2 = nabla_L * y_h * (1 - y_h) * w2_t

        # Update
        w1_t = w1_t - lr * np.sum(nabla_L * y_h * (1 - y_h) * y_1)
        w2_t = w2_t - lr * np.sum(nabla_L * y_h * (1 - y_h) * y_2)
        w0_t = w0_t - lr * np.sum(nabla_L * y_h * (1 - y_h) * 1)

        w11_t = w11_t - lr * np.sum(nabla_y_h_y1 * y_1 * (1 - y_1) * x1)
        w12_t = w12_t - lr * np.sum(nabla_y_h_y1 * y_1 * (1 - y_1) * x2)
        w13_t = w13_t - lr * np.sum(nabla_y_h_y1 * y_1 * (1 - y_1) * x3)
        w10_t = w10_t - lr * np.sum(nabla_y_h_y1 * y_1 * (1 - y_1) * 1)

        w21_t = w21_t - lr * np.sum(nabla_y_h_y2 * y_2 * (1 - y_2) * x1)
        w22_t = w22_t - lr * np.sum(nabla_y_h_y2 * y_2 * (1 - y_2) * x2)
        w23_t = w23_t - lr * np.sum(nabla_y_h_y2 * y_2 * (1 - y_2) * x3)
        w20_t = w20_t - lr * np.sum(nabla_y_h_y2 * y_2 * (1 - y_2) * 1)

        if np.mod(e, 500) == 0 or e == 1:
            MSE = np.sum((y - y_h) ** 2) / (len(y))
            #print(MSE)

    y_pred, _, _ = mlp(w11_t, w12_t, w13_t, w10_t, w21_t, w22_t, w23_t, w20_t, w1_t, w2_t, w0_t, x1, x2, x3)
    print("Ground truth values:", y)
    print("Predicted values:", y_pred)
    i += 1
