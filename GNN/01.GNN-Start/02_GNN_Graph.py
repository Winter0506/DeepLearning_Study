# 简短代码创建图

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0,1,2],
                          [1,2,1]],dtype=torch.long)

x = torch.tensor([[1,2],[3,4],[5,6]], dtype=torch.float)

y = torch.tensor([[1],[0],[1]], dtype=torch.float)

data = Data(x=x, y=y, edge_index=edge_index)


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
    图结构: Data(edge_index=[2, 3], x=[3, 2], y=[3, 1])
    图中的键: ['x', 'edge_index', 'y']
    图中节点数: 3
    图中边数: 3
    图中节点属性特征数: 2
    图中是否存在孤立节点 False
    图中是否存在自环 False
    ======节点特征======
    tensor([[1., 2.],
            [3., 4.],
            [5., 6.]])
    ======节点标签======
    tensor([[1.],
            [0.],
            [1.]])
    ======图中边形状======
    tensor([[0, 1, 2],
            [1, 2, 1]])
    '''
check_graph(data)