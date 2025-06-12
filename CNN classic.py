import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# 設定裝置（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定義資料轉換（轉 tensor + 標準化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 2. 下載完整 MNIST 訓練集
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 3. 選取前 1000 筆資料索引
subset_indices = random.sample(range(len(full_dataset)), 1000)
# 4. 建立子資料集（只包含 1000 筆）
subset_dataset = Subset(full_dataset, subset_indices)

# 5. 建立 DataLoader（只用 1000 筆）
train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)

# 6. 下載測試集（完整測試集）
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 7. 定義模型
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = DigitCNN().to(device)

# 8. 損失函數與優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 9. 訓練迴圈
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# 10. 測試模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total * 100:.2f}%")

torch.save(model.state_dict(), "digit_cnn.pth")
