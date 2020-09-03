import torch
import torch.nn as nn
import torchvision

def Conv3x3BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True) # inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出。省去了反复申请与释放内存的时间，直接代替原来的值。
    )

class VGGNet(nn.Module):
    def __init__(self, block_nums,num_classes=1000): # 接收block_nums数组，另外还有指定num_classes为1000
        super(VGGNet, self).__init__()

        # 第一层 输入通道为3 输出通道为64   卷积层个数为2
        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        # 第二层 输入通道为64 输出通道为128 卷积层个数为2
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        # 第三层 输入通道为128 输出通道为256 卷积层个数为3
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        # 第四层 输入通道为256 输出通道为512 卷积层个数为3
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        # 第五层 输入通道为512 输出通道为512 卷积层个数为3
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        # 分类层 三个全连接层 输入特征512*7*7->4096->4096->num_classes
        # dropout参数为0.2
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

        self._init_params()

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels,out_channels))
        for i in range(1,block_num):
            layers.append(Conv3x3BNReLU(out_channels,out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2, ceil_mode=False)) # 在卷积层结束后加上池化层  celi_model默认为地板模式
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # pass.默认为fan_in模式，fan_in可以保持前向传播的权重方差的数量级，fan_out可以保持反向传播的权重方差的数量级。
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0),-1) # x从[1,512,7,7]变成[1,512*7*7]
        out = self.classifier(x)
        return out

def VGG16():
    block_nums = [2, 2, 3, 3, 3]
    model = VGGNet(block_nums) # 将列表传入其中
    return model

def VGG19():
    block_nums = [2, 2, 4, 4, 4]
    model = VGGNet(block_nums)
    return model

if __name__ == '__main__':
    model = VGG16() # 使用的是VGG16
    print(model)

    input = torch.randn(1,3,224,224) # 一张图片，通道为3，宽和高为224
    out = model(input)
    print(out.shape) # torch.size([1,1000])

'''
VGGNet(
  (stage1): Sequential(
    (0): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (stage2): Sequential(
    (0): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): Sequential(
      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (stage3): Sequential(
    (0): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): Sequential(
      (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Sequential(
      (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (stage4): Sequential(
    (0): Sequential(
      (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): Sequential(
      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Sequential(
      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (stage5): Sequential(
    (0): Sequential(
      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): Sequential(
      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Sequential(
      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): Dropout(p=0.2, inplace=False)
    (2): Linear(in_features=4096, out_features=4096, bias=True)
    (3): Dropout(p=0.2, inplace=False)
    (4): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
torch.Size([1, 1000])
'''