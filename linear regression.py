import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 讀資料
df = pd.read_csv("HOUSE.csv", encoding='utf-8')
X = df['AREA'].values
y = df['PRICE'].values
print(X,y)
# 初始化參數
a = 0.0
b = 0.0

# 超參數
learning_rate = 0.002
epochs = 3000
m = len(y)

for i in range(epochs):
    # 預測值
    h = a * X + b

    # 計算偏微分
    da = (1/m) * np.sum((h - y) * X)  # ∂J/∂a
    db = (1/m) * np.sum(h - y)        # ∂J/∂b

    # 更新參數
    a -= learning_rate * da
    b -= learning_rate * db

    # 計算 cost（可選）
    J = (1/(2*m)) * np.sum((h - y) ** 2)

    # 每100次印出一次成本
    if i % 100 == 0:
        print(f"Epoch {i}: Cost={J:.2f}, a={a:.4f}, b={b:.4f}")

# 繪圖
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
plt.scatter(X, y, color='blue', label='原始資料')  # 原始資料點
plt.plot(X, a * X + b, color='red', label=f'線性迴歸線: y={a:.2f}x + {b:.2f}')  # 預測線

plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.legend()
plt.grid(True)
plt.show()
#實際應用
x = int(input("輸入房子的面積"))
r = a * x + b
print(f"預測房價={r}元")


