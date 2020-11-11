import torch
from torch_geometric.data import Data


print("PyTorch ==", torch.__version__)
print("CUDA available", torch.cuda.is_available())
print("CUDA ==", torch.version.cuda)

'''
PyTorch == 1.5.1
CUDA available True
CUDA == 10.1
'''

# 创建有向图 节点具有属性和标签
src = [0,1,2] # 起点
dst = [1,2,1] # 终点

edge_index = torch.tensor([src, dst], dtype=torch.long)  # 定义边

x0 = [1,2]
x1 = [3,4]
x2 = [5,6]
x = torch.tensor([x0,x1,x2], dtype=torch.float)  # 节点属性 

y0 = [1]
y1 = [0]
y2 = [1]
y = torch.tensor([y0,y1,y2], dtype=torch.float)  # 节点标签

data = Data(x=x, y=y, edge_index=edge_index)  # 创建图

print(data)  # Data(edge_index=[2, 3], x=[3, 2], y=[3, 1])  为什么这样子


