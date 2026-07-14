import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import numpy as np
import json
import ncutil  
import graphcast

def build_loss_weights(dynamic_vars={}, dynamic_weights={}, level_weights=1, **key):
    loss_weights = []
    for i in sorted(dynamic_vars):
        w = np.ones(dynamic_vars[i], dtype=np.float32) * dynamic_weights[i]
        if len(w) > 1: w *= level_weights
        loss_weights.append(w)
    return np.concatenate(loss_weights)

def count_grid_weight(m2g_idx):
    mesh_count = np.bincount(m2g_idx[:, 0])
    buf = np.zeros(m2g_idx[:, 1].max() + 1, dtype=np.float32)
    np.add.at(buf, m2g_idx[:, 1], mesh_count[m2g_idx[:, 0]])
    return buf.min() / buf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载网络结构与配置
root_dir = '../weights/lite_5mesh_13level_1deg'
graph_body = dict(np.load(root_dir + '/GraphBody.npz'))
norm_vector = dict(np.load(root_dir + '/NormVector.npz'))

with open(root_dir + '/DataConfig.json') as f:
    data_config = json.loads(f.read())

model = graphcast.GraphCastModel(graph_body, norm_vector).to(device)
model.train()

# 2. 准备各种变量的 key 列表与层级信息
const_keys = data_config['const_vars'].keys()
dynamic_keys = data_config['dynamic_vars'].keys()
force_keys = data_config['force_vars'].keys()
levels = data_config['levels']

def to_tensor(arr): 
    return torch.from_numpy(arr).float().to(device)

# 外部权重
vars_w = to_tensor(build_loss_weights(**data_config)).view(1, 1, -1)
grid_w = to_tensor(count_grid_weight(graph_body['m2g_idx'])).view(1, -1, 1)

# 加载数据集
ds_data = ncutil.nc.Dataset('../weights/testdata_and_normvector/input_data_20220101_10_step4.nc')
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# --- 3. 训练循环 ---
for epoch in range(10):
    optimizer.zero_grad(set_to_none=True)
    
    step_count = 2
    dynamic_list, force_list = [], []
    
    # 纯净读取时序上的 dynamic 和 force 变量
    for i in range(step_count + 2): 
        dy = ncutil.get_data(ds_data, i, keys=dynamic_keys, levels=levels)
        fr = ncutil.get_data(ds_data, i, keys=force_keys, levels=levels)
        
        dynamic_list.append(to_tensor(dy).view(1, -1, dy.shape[-1]))
        force_list.append(to_tensor(fr).view(1, -1, fr.shape[-1]))
    
    # 纯净读取静态变量 const
    const_np = ncutil.get_data(ds_data, 0, keys=const_keys, levels=levels)
    const = to_tensor(const_np).view(1, -1, const_np.shape[-1])

    # 展开初始时间步状态
    dynamic_t1, dynamic_t2 = dynamic_list[:2]
    force_t1, force_t2 = force_list[:2]
    
    total_loss = 0

    # --- 自回归循环 ---
    for t in range(step_count):
        target = dynamic_list[t + 2] - dynamic_list[t + 1]
        force_next = force_list[t + 2]
        
        # 使用 Gradient Checkpoint 节省显存
        dynamic_next, diff = checkpoint(
            model, const, dynamic_t1, dynamic_t2, 
            force_t1, force_t2, force_next, target,
            use_reentrant=False 
        )

        step_loss = torch.mean((diff ** 2) * vars_w * grid_w)
        total_loss += step_loss
        
        # 状态滚动更新
        dynamic_t1, dynamic_t2 = dynamic_t2, dynamic_next
        force_t1, force_t2 = force_t2, force_next

    avg_loss = total_loss / step_count
    avg_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Avg Loss: {avg_loss.item():.8f}")
