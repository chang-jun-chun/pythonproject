import numpy as np

# 激活函數與導數
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 損失函數與導數（二元交叉熵 + sigmoid 輸出）
def compute_loss(y_hat, y):
    m = y.shape[1]
    return -np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8)) / m

# 初始化
np.random.seed(42)
input_size = 3
hidden1_size = 4
hidden2_size = 3
output_size = 1
num_samples = 100

# 產生隨機資料（假資料）
X = np.random.randn(input_size, num_samples)
y = (np.sum(X, axis=0, keepdims=True) > 0).astype(int)  # 根據和正負來分類

# 初始化參數
W1 = np.random.randn(hidden1_size, input_size) * np.sqrt(2. / input_size)
W2 = np.random.randn(hidden2_size, hidden1_size) * np.sqrt(2. / hidden1_size)
W3 = np.random.randn(output_size, hidden2_size) * np.sqrt(2. / hidden2_size)
b1 = np.zeros((hidden1_size, 1))
b2 = np.zeros((hidden2_size, 1))
b3 = np.zeros((output_size, 1))

# 訓練參數
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # === Forward Propagation ===
    Z1 = W1 @ X + b1
    A1 = relu(Z1)

    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)

    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    # === Compute Loss ===
    loss = compute_loss(A3, y)

    # === Backward Propagation ===
    m = X.shape[1]
    dZ3 = A3 - y
    dW3 = (1/m) * dZ3 @ A2.T
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = W3.T @ dZ3
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    # === Update Weights ===
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # === 每100次列印 Loss ===
    if epoch % 100 == 0:
        predictions = (A3 > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        print(f"訓練週期 {epoch}, Loss: {loss:.4f}, 準確率: {accuracy:.2f}")