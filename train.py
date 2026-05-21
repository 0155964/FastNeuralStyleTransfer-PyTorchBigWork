import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from PIL import Image

# ==========================================
# 1. 定义转换网络 (Image Transform Network)
# 用于将普通照片转换为艺术风格图
# ==========================================
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

# ==========================================
# 2. 定义 VGG 损失网络 (提取特征)
# ==========================================
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4): self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9): self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16): self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23): self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3

def calc_gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

# ==========================================
# 3. 核心训练逻辑
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    # 注意：ImageFolder 要求子目录，所以你把解压的 train2014 放进 dataset/ 目录下
    # 例如：dataset/train2014/xxx.jpg
    dataset_dir = "./dataset" 
    style_image_path = "./style_images/starry_night.jpg" # 你自己找一张《星空》放这里
    save_model_path = "./models/starry_night.pth"
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./style_images", exist_ok=True)

    # 超参数配置
    epochs = 2
    batch_size = 4
    content_weight = 1e5
    style_weight = 1e10
    lr = 1e-3

    # 数据集加载
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(dataset_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    transformer = TransformerNet().to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    
    # 处理风格图
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = Image.open(style_image_path).convert('RGB')
    style = style_transform(style).unsqueeze(0).to(device)
    
    # 预计算风格图的 Gram 矩阵
    features_style = vgg(normalize_batch(style))
    gram_style = [calc_gram_matrix(y) for y in features_style]

    print("开始训练...")
    for epoch in range(epochs):
        transformer.train()
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            x = x.to(device)
            optimizer.zero_grad()
            
            # 1. 通过转换网络生成图片
            y = transformer(x)

            # 2. 将生成图和原图传入 VGG 计算特征
            x = normalize_batch(x)
            y = normalize_batch(y)
            features_y = vgg(y)
            features_x = vgg(x)

            # 3. 计算内容损失 (取 relu2_2)
            content_loss = content_weight * mse_loss(features_y[1], features_x[1])

            # 4. 计算风格损失 (取四层特征的 Gram 矩阵差)
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = calc_gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s.expand_n_batch, -1, -1))
            style_loss *= style_weight

            # 5. 反向传播更新 生成器 (transformer)
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            if (batch_id + 1) % 500 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_id+1}/{len(train_loader)}] \t "
                      f"Content Loss: {content_loss.item():.2f} \t Style Loss: {style_loss.item():.2f}")

    # 保存训练好的模型 (.pth)
    transformer.eval()
    torch.save(transformer.state_dict(), save_model_path)
    print(f"模型已保存至 {save_model_path}")

if __name__ == "__main__":
    train()