import numpy as np
def numerical_gradient(f, x, eps=1e-4):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        original_value = x[idx]

        x[idx] = original_value + eps
        fx_plus_eps = f(x)

        x[idx] = original_value - eps
        fx_minus_eps = f(x)

        grad[idx] = (fx_plus_eps - fx_minus_eps) / (2 * eps)
        x[idx] = original_value  # restore

        it.iternext()

    return grad
def conv2d(input_matrix, kernel):
    h, w = input_matrix.shape
    kh, kw = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            region = input_matrix[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

def loss_fn(kernel_flat, input_matrix, target_output, kernel_shape):
    kernel = kernel_flat.reshape(kernel_shape)
    conv_output = conv2d(input_matrix, kernel)
    return np.mean((conv_output - target_output)**2)  # MSE
input_matrix = np.random.rand(7, 7)
target_output = np.ones((5, 5))
kernel = np.random.randn(3, 3)

# 數值微分求梯度
kernel_flat = kernel.flatten()
grad = numerical_gradient(
    lambda k: loss_fn(k, input_matrix, target_output, kernel.shape),
    kernel_flat
)

# 顯示梯度並更新權重
kernel_updated = kernel_flat - 0.1 * grad
kernel = kernel_updated.reshape(kernel.shape)

print("更新後的 kernel：")
print(kernel)