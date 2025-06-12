import numpy as np

# 🟢 初始化參數（放在函數外，全域）
conv_filter = np.random.randn(3, 3) * 0.1
fc_weights = np.random.randn(2, 9) * 0.1
fc_bias = np.zeros(2)

# 🧮 卷積
def conv2d(input_matrix, kernel):
    h, w = input_matrix.shape
    kh, kw = kernel.shape
    output = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = input_matrix[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

# 🔃 softmax 和 loss
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def cross_entropy(pred, target):
    return -np.sum(target * np.log(pred + 1e-8))

# 🔁 數值微分
def numerical_gradient(param, loss_fn, eps=1e-4):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + eps
        loss_plus = loss_fn()

        param[idx] = orig - eps
        loss_minus = loss_fn()

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        param[idx] = orig  # 恢復原值
        it.iternext()
    return grad

# 🔒 loss function 包裝器
def make_loss_fn(image, label):
    def loss_fn():
        feature = conv2d(image, conv_filter).flatten()
        logits = fc_weights @ feature + fc_bias
        probs = softmax(logits)
        return cross_entropy(probs, label)
    return loss_fn

# 🚀 訓練一步
def train_step(image, label, lr=0.1):
    global conv_filter, fc_weights, fc_bias

    loss_fn = make_loss_fn(image, label)
    loss = loss_fn()

    grad_filter = numerical_gradient(conv_filter, loss_fn)
    grad_weights = numerical_gradient(fc_weights, loss_fn)
    grad_bias = numerical_gradient(fc_bias, loss_fn)

    conv_filter -= lr * grad_filter
    fc_weights -= lr * grad_weights
    fc_bias -= lr * grad_bias

    return loss
image_0 = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
label_0 = np.array([1, 0])

# 類別 1：右亮
image_1 = np.array([
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
label_1 = np.array([0, 1])

# 🔁 訓練
for epoch in range(20):
    loss0 = train_step(image_0, label_0)
    loss1 = train_step(image_1, label_1)
    print(f"Epoch {epoch+1}: Loss0={loss0:.4f}, Loss1={loss1:.4f}")
