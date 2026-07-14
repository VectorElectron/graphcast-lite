import numpy as np
import json, pickle
from time import time

import sys; sys.path.append('../')
from graphcast import ncutil, forcefield, gcnumpy

model_root = '../weights/lite_5mesh_13level_1deg'
# 1. 加载权重与配置
graph_body = dict(np.load(model_root + '/GraphBody.npz'))
norm_vector = dict(np.load(model_root + '/NormVector.npz'))

with open(model_root + '/DataConfig.json') as f:
    data_config = json.loads(f.read())
    
with open(model_root + '/GraphWeight.pkl', 'rb') as f:
    weights = pickle.load(f)

# 2. 提取需要的各种变量 Key 列表
const_keys = list(data_config['const_vars'].keys())
dynamic_keys = list(data_config['dynamic_vars'].keys())
force_keys = list(data_config['force_vars'].keys())
all_keys = const_keys + dynamic_keys + force_keys

levels = data_config['levels']

# 3. 时间步解析
ds_data = ncutil.nc.Dataset('../weights/testdata_and_normvector/input_data_20220101_10_step4.nc')
t0, t1 = ncutil.get_times(ds_data)[:2]
ts = t1 + (np.arange(4) + 1) * (t1 - t0) # 后续的 4 个 step

# 4. 纯净的数据提取 (替代原有的三通道解构)
const = ncutil.get_data(ds_data, t0, keys=const_keys, levels=levels)
dynamic_t1 = ncutil.get_data(ds_data, t0, keys=dynamic_keys, levels=levels)
dynamic_t2 = ncutil.get_data(ds_data, t1, keys=dynamic_keys, levels=levels)

# 强迫场保持原逻辑
force_t1 = forcefield.force_5cn(t0, data_config['lats'], data_config['lons'])
force_t2 = forcefield.force_5cn(t1, data_config['lats'], data_config['lons'])

# 5. 硬件后端与数据初始化 (支持 Cupy 切换)
import numpy as np  # 如果需要 GPU 加速，这里可以换成 import cupy as np
gcnumpy.np = np
for i in graph_body: graph_body[i] = np.asarray(graph_body[i])
for i in norm_vector: norm_vector[i] = np.asarray(norm_vector[i])
for i in weights: weights[i] = [np.asarray(j) for j in weights[i]]

# 初始化模型
net = gcnumpy.GraphCastModel(graph_body, norm_vector, weights)

# 6. 自回归推理循环
start = time()
for step, t in enumerate(ts):
    print(f"start step {step + 1} ", end='')
    force_next = forcefield.force_5cn(t, data_config['lats'], data_config['lons'])

    # 这里的 x_input 顺序跟模型 forward 预留接口对齐
    x_input = const, dynamic_t1, dynamic_t2, force_t1, force_t2, force_next
    dynamic_next = net.forward(*[np.asarray(i) for i in x_input])
    
    # 状态滚动更新
    force_t1, force_t2 = force_t2, force_next
    dynamic_t1, dynamic_t2 = dynamic_t2, dynamic_next

print('cost:', time()-start)
print(dynamic_next.mean(), dynamic_next.min(), dynamic_next.max(),
      '\nshould be 13425.764 -2993.4424 204544.72 ')
