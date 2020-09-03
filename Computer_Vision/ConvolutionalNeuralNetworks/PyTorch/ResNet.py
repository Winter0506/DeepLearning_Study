import torch
import torch.nn as nn
import torchvision

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            # 输入通道 输出通道
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            # 特征维度
            nn.BatchNorm2d(places),
            # 对原值进行操作，然后将得到的值又直接复制到该值中
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            # 下采样跳连接
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    # 直接看 forward  x 为输入
    def forward(self, x):
        residual = x
        # 去看bottleneck
        out = self.bottleneck(x)

        # 残差
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    # ResNet([3, 4, 6, 3])  blocks = [3,4,6,3]  
    '''
    这个expansion是为了实现在1*1卷积时，扩展channel数量，匹配下一层的输入channel数量。4*64=256
    self.downsampling = downsampling
    是否需要下采样跳连接部分，这主要取决于neck网络输出的大小，与原来的图像的大小。
    '''
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        # 输入通道3 输出通道64
        self.conv1 = Conv1(in_planes = 3, places= 64)

        # block 3 输入 64 输出 64
        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        # block 4 输入 256 输出 128
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        # block 6 输入 512 输出 256
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        # block 3 输入 1024 输出 512
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        # 池化层 卷积核 7 步长 1
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 全连接层
        self.fc = nn.Linear(2048,num_classes)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        # 添加第一层
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        # 添加后面的 block-1 层  去看Bottleneck
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__=='__main__':
    # model = torchvision.models.resnet50()
    # 使用ResNet50
    model = ResNet50()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)