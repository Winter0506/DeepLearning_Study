import torch
import torchvision
import torch.nn as nn
#model = torchvision.models.resnet50(pretrained=True)
model=torchvision.models.resnet18()
num_ftrs = model.fc.in_features
# 改变model的全连接层
model.fc = nn.Linear(num_ftrs, 4)  # make the change
# 加载文件中的resnet18.pth
model.load_state_dict(torch.load("resnet18.pth"))

model.eval()
example = torch.rand(1, 3, 224, 224)
# 跟踪一个函数并返回可执行文件  
# torch.jit.trace
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet18.pt")
