from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


#训练数据目录
data_dir = "./DataSets"

# 选择训练网络 [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

#权重文件输出目录及名称 --> 当前文件夹目录
output_path="./resnet18.pth"

#学习率
lr_rate=0.001

# 种类数目
num_classes = 4

# Batch Size取决你电脑内存大小
batch_size = 8

# 训练次数 
num_epochs = 5

# False时候更新所有参数，是特征提取，True时候只更新最后一层的参数，
feature_extract = False


# train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception")
# 如果模型名字为inception就为True，否则为False，这里默认为False
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    # 记录时间
    since = time.time()

    val_acc_history = []
    val_loss_history=[]
    
    train_acc_history = []
    train_loss_history=[]
    
    # 深拷贝模型中的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # 转换到设备上
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 将优化器参数梯度设置为0
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # 训练阶段
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    # 这段没看，写的是inception以及训练阶段
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        # outputs与loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # torch.max取最大置信度
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # 训练阶段的反向传播和优化器优化
                        loss.backward()
                        optimizer.step()

                # statistics
                # 计算running损失和正确率，进行加和
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 计算epoch损失，求取平均值
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # 验证集精度设置为最佳
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # 添加损失和精度记录    
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss) 
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)          

        # 打印空行
        print()

    # 计算所用时间
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # 打印精度
    print('Best val Acc: {:4f}'.format(best_acc))

    # 模型加载最佳时的参数
    # load best model weights
    # 将各种结果返回
    model.load_state_dict(best_model_wts)
    return model, train_acc_history,train_loss_history,val_acc_history,val_loss_history


# 设置参数是否需要梯度
def set_parameter_requires_grad(model, feature_extracting):
    # 如果是特征提取：需要梯度是False
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    # 初始化模型中的变量
    # model_ft 初始化为None，后面要用到
    model_ft = None
    # 初始化尺寸一开始设置为0
    input_size = 0

    # 这里只看了resnet
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        # 函数，在后面有定义
        set_parameter_requires_grad(model_ft, feature_extract)
        # 全连接层的输入参数
        num_ftrs = model_ft.fc.in_features
        # 添加一个全连接层
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # 输入尺寸
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    # 是无效模型，并退出
    else:
        print("Invalid model name, exiting...")
        exit()
    
    # 返回的是赋值并修改的模型，以及设置的输入尺寸
    return model_ft, input_size



if __name__ == '__main__':

    # 打印torch和torchvision的版本
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    # 对模型进行初始化操作
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
    # 输出模型结构
    print(model_ft) 
    
    
    # data_transforms的格式
    data_transforms = {
        'train': transforms.Compose([
            # 随机裁剪
            transforms.RandomResizedCrop(input_size),
            # 水平翻转
            transforms.RandomHorizontalFlip(),
            # 转换成Tensor
            transforms.ToTensor(),
            # 归一化 分别是三个通道的均值和方差
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print("Initializing Datasets and Dataloaders...")
    
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    print(image_datasets)

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    print(dataloaders_dict)
    # Detect if we have a GPU available  选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # Send the model to GPU
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    # 通过转换成列表打印出参数
    print(list(params_to_update))
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                # 如果是特征提取并且需要梯度，添加参数到列表中
                params_to_update.append(param)
                # 打印名字
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                # 打印名字
                print("\t",name)
    
    # Observe that all parameters are being optimized
    # 优化器
    optimizer_ft = optim.SGD(params_to_update, lr=lr_rate, momentum=0.9)
    
    # Setup the loss fxn
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练和评估
    # Train and evaluate
    # 得到微调后的模型  训练精度和损失  验证精度和损失
    model_ft, trainacc,trainloss,valacc,valloss = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    # 保存模型
    torch.save(model_ft.state_dict(), output_path)

    # 标题
    plt.title("Acc&Loss")
    # 横坐标
    plt.xlabel("Training Epochs")
    # 纵坐标
    plt.ylabel("Value")
    # 绘出结果
    plt.plot(range(1,num_epochs+1),trainacc,label="TrainAcc")
    plt.plot(range(1,num_epochs+1),trainloss,label="TrainLoss")
    plt.plot(range(1,num_epochs+1),valacc,label="ValAcc")
    plt.plot(range(1,num_epochs+1),valloss,label="ValLoss")
    
    # y轴尺度
    plt.ylim((0,1.))
    # x轴每个点及其间隔
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    # 给图像加上图例，四个颜色分别代表什么
    plt.legend()
    plt.show()
    
    
