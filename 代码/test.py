import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
testing_data = r'C:\Users\86172\Desktop\实训\Pneumonia-Detection-using-Deep-Learning-main\test'
test_dataset = ImageFolder(root=testing_data, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# 定义自定义CNN模型
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

    def forward(self, x):
        # 经过卷积层、池化层
        x = self.pool1(self.conv1_relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(self.conv2_relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(self.conv3_relu(self.conv3_bn(self.conv3(x))))

        # 展平并通过全连接层
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2(x)  # 直接输出类的得分
        return x


# 加载训练好的模型
model = CustomCNN(num_classes=2)  # 假设分类问题为二分类
checkpoint = torch.load('pneumonia_model_optimized.pth', weights_only=False)  # 加载模型权重
model.load_state_dict(checkpoint, strict=False)  # 使用strict=False来忽略不匹配的参数
model.eval()  # 设置模型为评估模式

test_loss = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loss = 0  # 可选，用于记录损失（如果需要）
all_labels = []
all_predictions = []

with torch.no_grad():  # 禁用梯度计算
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        # 收集所有的标签和预测结果
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 将标签和预测结果转换为NumPy数组（如果它们还不是的话）
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# 计算精确率、召回率和F1分数（使用'binary'模式，因为是二分类问题）
precision = precision_score(all_labels, all_predictions, average='binary')
recall = recall_score(all_labels, all_predictions, average='binary')
f1 = f1_score(all_labels, all_predictions, average='binary')

# 打印结果
print(f'准确率: {100 * precision:.2f}%')
print(f'召回率: {100 * recall:.2f}%')
print(f'F1 分数: {100 * f1:.2f}%')

"""
# 指定图像文件夹路径
img_folder_path = r'C:/Users/86172/Desktop/实训/Pneumonia-Detection-using-Deep-Learning-main/test/NORMAL'

# 获取文件夹中的所有图像文件（支持JPEG, PNG格式）
img_paths = glob.glob(os.path.join(img_folder_path, "*.jpeg")) + glob.glob(os.path.join(img_folder_path, "*.png"))

if not img_paths:
    raise FileNotFoundError(f"文件夹中没有图像文件：{img_folder_path}")

# 逐个加载图像并进行预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for img_path in img_paths:
    try:
        # 加载并预处理图像
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # 增加批量维度
        img_tensor = img_tensor.to(device)

        # 进行预测
        with torch.no_grad():  # 禁用梯度计算，以节省内存和加速推理
            output = model(img_tensor)

        # 获取Softmax输出，得到每个类的概率
        probabilities = torch.nn.Softmax(dim=1)(output)  # 使用Softmax获取每个类的概率
        prediction = probabilities.cpu().numpy()  # 转换为NumPy数组

        # 输出预测结果
        print(f"处理图像：{img_path}")

        if prediction[0][0] > prediction[0][1]:
            print('Person is safe.')
        else:
            print('Person is affected with Pneumonia.')

        print(f'Predictions: {prediction}')

    except Exception as e:
        print(f"处理图像 {img_path} 时发生错误：{e}")
"""
