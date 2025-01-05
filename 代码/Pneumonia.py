import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
from glob import glob

# 设置图像尺寸和数据目录
IMAGESHAPE = (224, 224)
training_data = r'C:\Users\86172\Desktop\实训\Pneumonia-Detection-using-Deep-Learning-main\train'
testing_data = r'C:\Users\86172\Desktop\实训\Pneumonia-Detection-using-Deep-Learning-main\val'

# 获取类别数
classes = glob(training_data + '/*')
num_classes = len(classes)

# 数据增强与预处理
train_transforms = transforms.Compose([
    transforms.Resize(IMAGESHAPE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG16 的预训练均值和标准差
])

test_transforms = transforms.Compose([
    transforms.Resize(IMAGESHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练集和测试集数据
train_dataset = ImageFolder(root=training_data, transform=train_transforms)
test_dataset = ImageFolder(root=testing_data, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义自定义 CNN 架构
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_relu = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_relu = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.conv1_relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(self.conv2_relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(self.conv3_relu(self.conv3_bn(self.conv3(x))))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化自定义 CNN 模型
model = CustomCNN(num_classes=num_classes)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()

# 优化器：Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器：每 5 个 epoch 学习率衰减为原来的 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练过程
num_epochs = 20
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# 开始训练
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # 训练集迭代
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    # 计算训练集损失和准确率
    train_loss = running_loss / len(train_loader)
    train_acc = correct_preds / total_preds

    # 验证集迭代
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    # 计算验证集损失和准确率
    val_loss = val_loss / len(test_loader)
    val_acc = correct_preds / total_preds

    # 记录损失和准确率
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    # 打印训练信息
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # 学习率更新
    scheduler.step()

# 绘制损失和准确率曲线
plt.plot(train_loss_history, label='训练损失')
plt.plot(val_loss_history, label='验证损失')
plt.legend()
plt.savefig('LossVal_loss.png')
plt.show()

plt.plot(train_acc_history, label='训练准确率')
plt.plot(val_acc_history, label='验证准确率')
plt.legend()
plt.savefig('AccVal_acc.png')
plt.show()

# 保存训练后的模型
torch.save(model.state_dict(), 'pneumonia_model_optimized.pth')
