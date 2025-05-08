import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
import os

# 设置环境变量以减少显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 卷积自编码器定义，用于情形一的数据污染检测
class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 更深的编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # [3,224,224]->[64,112,112]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [64,112,112]->[128,56,56]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [128,56,56]->[256,28,28]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [256,28,28]->[512,14,14]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # 对称的解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 训练自编码器（优化显存使用）
def train_autoencoder(data_loader, epochs=15, device="cuda"):
    autoencoder = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4, weight_decay=1e-5)
    
    autoencoder.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, _ in data_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader):.4f}')
    
    return autoencoder

# 计算信任分数（基于重构误差）
def compute_trust_score(autoencoder, data_loader, device="cuda"):
    autoencoder.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = autoencoder(images)
            mse = F.mse_loss(outputs, images, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(1)  # 按样本计算平均误差
            reconstruction_errors.extend(mse.cpu().numpy())
    
    errors = np.array(reconstruction_errors)
    # 更鲁棒的归一化方法
    median_error = np.median(errors)
    trust_scores = np.exp(-errors / (median_error + 1e-6))  # 基于中位数的归一化
    return trust_scores / trust_scores.max()  # 归一化到[0,1]

# 计算模型信任分数（基于权重余弦相似度，用于情形二）
def compute_model_trust(global_model, client_model, device="cuda"):
    global_weights = torch.cat([param.flatten() for param in global_model.state_dict().values()]).to(device)
    client_weights = torch.cat([param.flatten() for param in client_model.state_dict().values()]).to(device)
    cosine_sim = torch.nn.functional.cosine_similarity(global_weights, client_weights, dim=0)
    return cosine_sim.item()

# 自定义污染数据集类（情形一）
class PoisonedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True, poison_ratio=0.4, noise_std=0.1):
        super(PoisonedCIFAR10, self).__init__(root, train=train, transform=transform, download=download)
        self.poison_ratio = poison_ratio
        self.noise_std = noise_std
        self.transform = transform
        self.poison_indices = self.poison_data()

    def poison_data(self):
        n_samples = len(self.data)
        n_poison = int(n_samples * self.poison_ratio)  # 4000 条污染数据
        normal_indices = [i for i, label in enumerate(self.targets) if label < 5]  # 前5类为正常
        anomaly_indices = [i for i, label in enumerate(self.targets) if label >= 5]  # 后5类为异常

        # 标签翻转：2000 条正常数据改为异常
        poison_label_indices = random.sample(normal_indices, n_poison // 2)
        for idx in poison_label_indices:
            self.targets[idx] = 10  # 新的异常类别

        # 噪声添加：2000 条异常数据添加噪声（在变换后的空间）
        poison_noise_indices = random.sample(anomaly_indices, n_poison // 2)
        return poison_label_indices + poison_noise_indices

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if index in self.poison_indices[len(self.poison_indices)//2:]:  # 噪声污染
            noise = torch.normal(0, self.noise_std, img.shape).to(img.device)
            img = torch.clamp(img + noise, 0, 1)
        return img, target

# ResNet 基本块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# ResNet 瓶颈块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

# ResNet 模型
class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride,
                            groups=self.groups, width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

# 添加权重噪声（情形二）
def add_weight_noise(model, fault_prob=0.4, device="cuda"):
    state_dict = model.state_dict()
    for key in state_dict:
        r = random.random()
        if r < fault_prob and state_dict[key].dtype in [torch.float, torch.float32, torch.float64]:  # 只对浮点类型张量添加噪声
            noise = torch.normal(0, 1, state_dict[key].shape).to(device)
            state_dict[key] += noise
    model.load_state_dict(state_dict)
    return model

# 修改训练函数以支持权重噪声
def train_with_noise(model, train_loader, weights, epochs=5, device="cuda", fault_prob=0.4):
    model.train().to(device)
    model.load_state_dict(weights, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch}/{epochs}]: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        scheduler.step()
    
    # 添加权重噪声（情形二）
    model = add_weight_noise(model, fault_prob=fault_prob, device=device)
    # 清理显存
    torch.cuda.empty_cache()
    return model

def evaluate(model, test_loader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    # 清理显存
    torch.cuda.empty_cache()
    return correct / total

# 基于数据信任值过滤客户端（情形一，优化显存使用）
def filter_clients_by_trust(client_loaders, models, global_model, num_clients, device="cuda"):
    trusted_models = []
    for loader, model in zip(client_loaders, models):
        autoencoder = train_autoencoder(loader, epochs=5, device=device)  # 训练自编码器
        trust_scores = compute_trust_score(autoencoder, loader, device=device)  # 计算信任分数
        avg_trust = np.mean(trust_scores)  # 计算平均信任分数
        
        # 打印每个客户端的信任分数（修复变量名错误）
        print(f"Client with average trust score {avg_trust:.4f}")
        
        if avg_trust > 0.4:  # 信任阈值
            trusted_models.append(model)
        else:
            print(f"Client discarded due to low trust score: {avg_trust:.4f}")
        
        # 释放自编码器显存
        del autoencoder
        torch.cuda.empty_cache()
    
    if not trusted_models:
        print("No trusted models, using global model.")
        trusted_models.append(global_model)
    return trusted_models

# 基于模型信任值过滤客户端（情形二）
def filter_clients_by_model_trust(client_loaders, models, global_model, num_clients, device="cuda"):
    trusted_models = []
    for loader, model in zip(client_loaders, models):
        trust_score = compute_model_trust(global_model, model, device)
        if trust_score > 0.1:  # 信任阈值
            trusted_models.append(model)
            print(f"Client with trust score {trust_score:.4f} .")
        else:
            print(f"Client with trust score {trust_score:.4f} discarded.")
        # 清理显存
        torch.cuda.empty_cache()
    if not trusted_models:
        print("No trusted models, using global model.")
        trusted_models.append(global_model)
    return trusted_models

# 修改聚合函数以支持信任值过滤
def aggregate_with_trust(models, global_model, num_clients, client_loaders, use_data_trust=True, device="cuda"):
    if use_data_trust:
        trusted_models = filter_clients_by_trust(client_loaders, models, global_model, num_clients, device)
    else:
        trusted_models = filter_clients_by_model_trust(client_loaders, models, global_model, num_clients, device)
    total_weights_avg = {key: torch.zeros_like(value).to(device) for key, value in global_model.state_dict().items()}
    for key in total_weights_avg:
        for model in trusted_models:
            total_weights_avg[key] += model.state_dict()[key]
        total_weights_avg[key] += global_model.state_dict()[key]
        total_weights_avg[key] = total_weights_avg[key] / (len(trusted_models) + 1)
    global_model.load_state_dict(total_weights_avg)
    # 清理显存
    torch.cuda.empty_cache()
    return global_model

# 数据加载（支持污染数据）
def load_data(use_poisoned=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 更合理的归一化
    ])
    
    if use_poisoned:
        trainset = PoisonedCIFAR10(root='./data', train=True, download=True, transform=transform, poison_ratio=0.4)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    num_clients = 10  # 减少客户端数量以降低显存占用
    data_per_client = len(trainset) // num_clients
    client_data = [Subset(trainset, range(i * data_per_client, (i + 1) * data_per_client)) for i in range(num_clients)]
    client_loaders = [
        DataLoader(data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)  # 减少 batch_size
        for data in client_data
    ]
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return client_loaders, test_loader

# 主训练循环（支持两种情形）
rounds = 10


# 情形一：数据集污染
client_loaders, test_loader = load_data(use_poisoned=True)
global_model = resnet50(num_classes=11).to(device)  # 增加异常类别
clients = [resnet50(num_classes=11).to(device) for _ in range(10)]  # 减少  Daisuke
zero_pad_accs_scenario1 = []
local_models = []
for model in clients:
    local_models.append(model)
global_model = aggregate_with_trust(local_models, global_model, len(clients), client_loaders, use_data_trust=True, device=device)
acc = evaluate(global_model, test_loader)
print(f"Scenario 1 - Round 0: Test Accuracy = {acc:.4f}")
for round_num in range(1, rounds + 1):
    print(f"--- Scenario 1 - Round {round_num} ---")
    local_models = []
    for client, loader in tqdm(zip(clients, client_loaders), total=len(clients)):
        model = train_with_noise(client, loader, global_model.state_dict(), epochs=5, fault_prob=0.0)  # 无权重噪声
        local_models.append(model)
    global_model = aggregate_with_trust(local_models, global_model, len(clients), client_loaders, use_data_trust=True, device=device)
    acc = evaluate(global_model, test_loader)
    print(f"Scenario 1 - Round {round_num}: Test Accuracy = {acc:.4f}")
    zero_pad_accs_scenario1.append(acc)
print('Scenario 1 - zero_pad:', zero_pad_accs_scenario1)


# 情形二：权重噪声
client_loaders, test_loader = load_data(use_poisoned=False)
global_model = resnet50(num_classes=10).to(device)
clients = [resnet50(num_classes=10).to(device) for _ in range(10)]  # 减少客户端数量
zero_pad_accs_scenario2 = []
local_models = []
for model in clients:
    local_models.append(model)
global_model = aggregate_with_trust(local_models, global_model, len(clients), client_loaders, use_data_trust=False, device=device)
acc = evaluate(global_model, test_loader)
print(f"Scenario 2 - Round 0: Test Accuracy = {acc:.4f}")
for round_num in range(1, rounds + 1):
    print(f"--- Scenario 2 - Round {round_num} ---")
    local_models = []
    for client, loader in tqdm(zip(clients, client_loaders), total=len(clients)):
        model = train_with_noise(client, loader, global_model.state_dict(), epochs=5, fault_prob=0.4)  # 添加权重噪声
        local_models.append(model)
    global_model = aggregate_with_trust(local_models, global_model, len(clients), client_loaders, use_data_trust=False, device=device)
    acc = evaluate(global_model, test_loader)
    print(f"Scenario 2 - Round {round_num}: Test Accuracy = {acc:.4f}")
    zero_pad_accs_scenario2.append(acc)
print('Scenario 2 - zero_pad:', zero_pad_accs_scenario2)