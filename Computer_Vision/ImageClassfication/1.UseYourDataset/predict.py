#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torchvision 
import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn as nn

input_size = 224
names = ['backpack', 'butterfly', 'comet', 'monitor']


def pridict():

    # 设置device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置模型
    model=torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    # 更改模型
    model.fc = nn.Linear(num_ftrs, 4)  # make the change
    # 为模型加载参数
    model.load_state_dict(torch.load("resnet18.pth"))
    model = model.to(device)
    model.eval()  # 预测模式

    # 获取测试图片，并行相应的处理
    img = Image.open('1.jpg')
    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # 进行图像增广
    img = transform(img)

    '''
    torch.squeeze() 对数据的维度进行压缩。
    去掉维数为1的的维度，squeeze(a)将a中所有为1的维度删掉，不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度。
    还有一种形式就是b=torch.squeeze(a，N) ，在a中指定位置N去掉一个维数为1的维度

    torch.unsqueeze()对数据维度进行扩充。
    a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。
    还有一种形式就是b=torch.squeeze(a，N) ，在a中指定位置N加上一个维数为1的维度
    '''
    # 为模型第0维度加上一个维度
    #torch.Size([1, 3, 224, 224])
    img = img.unsqueeze(0)
    print(img.size())
    img = img.to(device)


    # 不需要梯度
    with torch.no_grad():
        py = model(img)
    _, predicted = torch.max(py, 1)  # 获取分类结果
    classIndex_ = predicted[0]

    print('预测结果', names[classIndex_])


if __name__ == '__main__':
    # 调用predict函数
    pridict()
