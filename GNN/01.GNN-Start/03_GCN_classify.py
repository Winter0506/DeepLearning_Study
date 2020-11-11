# GCN进行节点分类

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub

from torch_geometric.utils import to_networkx

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

dataset = KarateClub()

print("图数量:", len(dataset))
print("类数量:", dataset.num_classes)

data = dataset[0]


# 查看创建出来的图
def check_graph(data):
    print("图结构:", data)
    print("图中的键:", data.keys)
    print("图中节点数:", data.num_nodes)
    print("图中边数:", data.num_edges)
    print("图中节点属性特征数:", data.num_node_features)
    print("图中是否存在孤立节点", data.contains_isolated_nodes())
    print("图中是否存在自环", data.contains_self_loops())
    print("======节点特征======")
    print(data['x'])
    print("======节点标签======")
    print(data['y'])
    print("======图中边形状======")
    print(data['edge_index'])
    '''
    图数量: 1
    类数量: 2
    图结构: Data(edge_index=[2, 156], x=[34, 34], y=[34])
    图中的键: ['x', 'edge_index', 'y']
    图中节点数: 34
    图中边数: 156
    图中节点属性特征数: 34
    图中是否存在孤立节点 False
    图中是否存在自环 False
    ======节点特征======
    tensor([[1., 0., 0.,  ..., 0., 0., 0.],
            [0., 1., 0.,  ..., 0., 0., 0.],
            [0., 0., 1.,  ..., 0., 0., 0.],
            ...,
            [0., 0., 0.,  ..., 1., 0., 0.],
            [0., 0., 0.,  ..., 0., 1., 0.],
            [0., 0., 0.,  ..., 0., 0., 1.]])
    ======节点标签======
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    ======图中边形状======
    tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,
            3,  3,  3,  3,  3,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,
            7,  7,  8,  8,  8,  8,  8,  9,  9, 10, 10, 10, 11, 12, 12, 13, 13, 13,
            13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 19, 20, 20, 21,
            21, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 27, 27,
            27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 31,
            31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33,
            33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33],
            [ 1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 17, 19, 21, 31,  0,  2,
            3,  7, 13, 17, 19, 21, 30,  0,  1,  3,  7,  8,  9, 13, 27, 28, 32,  0,
            1,  2,  7, 12, 13,  0,  6, 10,  0,  6, 10, 16,  0,  4,  5, 16,  0,  1,
            2,  3,  0,  2, 30, 32, 33,  2, 33,  0,  4,  5,  0,  0,  3,  0,  1,  2,
            3, 33, 32, 33, 32, 33,  5,  6,  0,  1, 32, 33,  0,  1, 33, 32, 33,  0,
            1, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31, 29, 33,  2, 23,
            24, 33,  2, 31, 33, 23, 26, 32, 33,  1,  8, 32, 33,  0, 24, 25, 28, 32,
            33,  2,  8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33,  8,  9, 13, 14, 15,
            18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32]])
    '''
# check_graph(data)

def networkx(data):
    # 图形结构的可视化
    nxg = to_networkx(data)

    pr = nx.pagerank(nxg)
    pr_max = np.array(list(pr.values())).max()

    draw_pos = nx.spring_layout(nxg, seed=0)

    cmap = plt.get_cmap('tab10')
    labels = data.y.numpy()
    colors = [cmap(l) for l in labels]

    plt.figure(figsize=(10,10))

    nx.draw_networkx_nodes(nxg,
                           draw_pos,
                           node_size=[v/pr_max*1000 for v in pr.values()],
                           node_color=colors, alpha=0.5)
    nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
    nx.draw_networkx_labels(nxg, draw_pos, font_size=10)

    plt.title('KarateClub')
    plt.savefig("fig/fig1.jpg")

# networkx(data)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_size = 5
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, dataset.num_classes)

    def forword(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
print(model)
'''
Net(
  (conv1): GCNConv(34, 5)
  (conv2): GCNConv(5, 2)
)
'''
model.train()

# 输入图 进行优化器设置
data = dataset[0]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)  # NotImplementedError
    loss = F.nll_loss(out,data.y)
    loss.backward()
    optimizer.step()
    print('Epoch %d | Loss: %.4f'%(epoch,loss.item()))

# eval
model.eval()

_, pred = model(data).max(dim=1)

print("结果:",pred)
print("真值:",data[0]["y"])

print("========================")

# 测试数据创建  更改一些链接
test_dataset = KarateClub()
test_data = test_dataset[0]

x = test_data["x"]
edge_index = test_data['edge_index']

edge_index[1][61] = 0
edge_index[1][7] = 9
edge_index[1][56] = 9
edge_index[1][62] = 8
edge_index[1][140] = 2
edge_index[1][30] = 33

t_data = Data(x=x, edge_index=edge_index)
check_graph(t_data)

# 链接更改后的图,由训练模型执行节点分类(标签预测)
model.eval()
_, pred = model(t_data).max(dim=1)

print("======变更前======")
print(data["y"])
print("======变更后======")
print(pred)

