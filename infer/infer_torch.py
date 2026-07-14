import numpy as np
import json, pickle
from time import time
import torch

import sys; sys.path.append('../')
from graphcast import ncutil, forcefield, graphcast

# 1. 加载权重与配置
model_root = '../weights/lite_5mesh_13level_1deg'
# 1. 加载权重与配置
graph_body = dict(np.load(model_root + '/GraphBody.npz'))
norm_vector = dict(np.load(model_root + '/NormVector.npz'))

with open(model_root + '/DataConfig.json') as f:
    data_config = json.loads(f.read())


# 2. 提取需要的各种变量 Key 列表与层级
const_keys = list(data_config['const_vars'].keys())
dynamic_keys = list(data_config['dynamic_vars'].keys())
force_keys = list(data_config['force_vars'].keys())
all_keys = const_keys + dynamic_keys + force_keys

levels = data_config['levels']

# 3. 时间步解析
ds_data = ncutil.nc.Dataset('../weights/testdata_and_normvector/input_data_20220101_10_step4.nc')
t0, t1 = ncutil.get_times(ds_data)[:2]
ts = t1 + (np.arange(4) + 1) * (t1 - t0) # next 4 step

# 4. 纯净的数据提取 (对照 numpy 版本修改)
const = ncutil.get_data(ds_data, t0, keys=const_keys, levels=levels)
dynamic_t1 = ncutil.get_data(ds_data, t0, keys=dynamic_keys, levels=levels)
dynamic_t2 = ncutil.get_data(ds_data, t1, keys=dynamic_keys, levels=levels)

# 强迫场保持原逻辑
force_t1 = forcefield.force_5cn(t0, data_config['lats'], data_config['lons'])
force_t2 = forcefield.force_5cn(t1, data_config['lats'], data_config['lons'])

# 5. 初始化模型与加载权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = graphcast.GraphCastModel(graph_body, norm_vector)
net.load_state_dict(torch.load(model_root + '/GraphWeight.pth'))
net.to(device)
net.eval()

# 6. 自回归推理循环
start = time()
for step, t in enumerate(ts):
    with torch.no_grad():
        print(f"start step {step + 1} ")
        force_next = forcefield.force_5cn(t, data_config['lats'], data_config['lons'])
        x_input = const, dynamic_t1, dynamic_t2, force_t1, force_t2, force_next
        tensor_input = [torch.tensor(i, dtype=torch.float32, device=device) for i in x_input]
        dynamic_next = net.forward(*tensor_input)
        force_t1, force_t2 = force_t2, force_next
        dynamic_t1, dynamic_t2 = dynamic_t2, dynamic_next.cpu().numpy() # 滚回 NumPy 用于下一步循环或保持统一

print('cost:', time()-start)
print(dynamic_next.mean().item(), dynamic_next.min().item(), dynamic_next.max().item(),
      '\nshould be 13425.764 -2993.4424 204544.72')
