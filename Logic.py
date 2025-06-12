import numpy as np

# 資料
x = np.array([1, 1, 2, 4, 5, 11, 50, 45, 36, 49]).reshape(-1, 1)  # 轉為 10x1
y = np.array([10, 15, 17, 20, 30, 45, 46, 48, 60, 66])
y = (y >= 40).astype(int).reshape(-1, 1)  # 二元分類，轉為 10x1

# 初始化參數
A = np.zeros((1, 1))  # 權重維度：1x1，因為只有一個特徵
b = 0                # 偏置項

# 超參數
learning_rate = 0.002
epochs = 3000
m = len(x)

# Sigmoid 函數
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(epochs):
    # 預測值
    z = np.dot(x, A) + b  # 線性組合
    h = sigmoid(z)        # 預測機率

    # 損失函數
    J = -(1/m) * np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))

    # 梯度計算
    dA = (1/m) * np.dot(x.T, (h - y))  # 對 A 的梯度
    db = (1/m) * np.sum(h - y)         # 對 b 的梯度

    # 更新參數
    A -= learning_rate * dA
    b -= learning_rate * db

    # 每100步印一次
    if i % 100 == 0:
        print(f"Epoch {i}, Cost: {J:.4f}")

c = int(input("輸入x的大小: "))
c = np.array([[c]])   # 轉成 1x1 陣列

new_z = np.dot(c, A) + b
new_h = sigmoid(new_z)

print(f"輸出結果: {new_h[0][0]:.4f}")

if new_h[0][0] >= 0.5:
    print("高機率有腫瘤")
else:
    print("高機率沒腫瘤")