import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# Inputs
x1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
x2 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
x3 = np.array([0, 1, 0, 1, 0, 1, 0, 1])

# Outputs
y1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])  # AND
y2 = np.array([0, 1, 1, 1, 1, 1, 1, 1])  # OR
y3 = np.array([0, 0, 0, 1, 0, 1, 0, 0])  # (x1 XOR x2) AND x3
y_arr = [y1, y2, y3]

w1_t = 0
w2_t = 0
w3_t = 0
w0_t = 0

num_of_epochs = 1000
lr = 0.5 # learning rate
N = len(x1)
i = 1
print("Functions: 1 - AND, 2 - OR , 3 - (x1 XOR x2) AND x3")
for y in y_arr:
    print()
    #print("Function number:", i)
    for e in range(num_of_epochs):
        y_h = expit(w1_t * x1 + w2_t * x2 + w3_t * x3 + w0_t)
        nablaL_w1 = 2 / N * sum((y - y_h) * -y_h * (1 - y_h) * x1)
        nablaL_w2 = 2 / N * sum((y - y_h) * -y_h * (1 - y_h) * x2)
        nablaL_w3 = 2 / N * sum((y - y_h) * -y_h * (1 - y_h) * x3)
        nablaL_w0 = 2 / N * sum((y - y_h) * -y_h * (1 - y_h) * 1)
        w1_t = w1_t - lr * nablaL_w1
        w2_t = w2_t - lr * nablaL_w2
        w3_t = w3_t - lr * nablaL_w3
        w0_t = w0_t - lr * nablaL_w0
        if np.mod(e, 50) == 0 or e == 1:  # Plot after every 20th epoch
            y_pred = expit(w1_t * x1 + w2_t * x2 + w3_t * x3 + w0_t)
            MSE = np.sum((y - y_pred) ** 2) / (len(y))
            #print(MSE)
    print("Ground truth values:", y)
    print("Predicted values:", y_pred)
    i += 1
