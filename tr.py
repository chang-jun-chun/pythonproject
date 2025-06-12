import torch
import torch.nn as nn
import torch.optim as optim #Optimizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#把圖片轉成 [C, H, W] 格式的 tensor
transform = transforms.ToTensor()
#訓練資料及測試資料
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#批次分裝資料
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),# 28x28 → 784 全連接層：784 → 128
            nn.ReLU(),
            nn.Linear(128, 64),# 全連接層：128 → 64
            nn.ReLU(),
            nn.Linear(64, 10)#輸出層：64 → 10
        )

    def forward(self, x):
        return self.model(x)

model = DigitClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 評估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images)
        predicted = torch.argmax(preds, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total * 100:.2f}%")