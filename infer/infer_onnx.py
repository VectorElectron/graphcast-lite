import onnxruntime as ort
import numpy as np
import json
from time import time

import sys; sys.path.append('../')
from graphcast import ncutil, forcefield  # 引入 forcefield 保持逻辑一致

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

# 3. 初始化 ONNX Session 与时间步解析
ds_data = ncutil.nc.Dataset('../weights/testdata_and_normvector/input_data_20220101_10_step4.nc')
net = ort.InferenceSession(model_root + '/GraphWeight.onnx')

t0, t1 = ncutil.get_times(ds_data)[:2]
ts = t1 + (np.arange(4) + 1) * (t1 - t0) # 后续的 4 个 step

# 4. 纯净的数据提取 (对照 numpy/torch 版本修改)
const = ncutil.get_data(ds_data, t0, keys=const_keys, levels=levels)
dynamic_t1 = ncutil.get_data(ds_data, t0, keys=dynamic_keys, levels=levels)
dynamic_t2 = ncutil.get_data(ds_data, t1, keys=dynamic_keys, levels=levels)

# 强迫场切换为 forcefield.force_5cn 算子计算
force_t1 = forcefield.force_5cn(t0, data_config['lats'], data_config['lons'])
force_t2 = forcefield.force_5cn(t1, data_config['lats'], data_config['lons'])

# 5. 自回归推理循环
start = time()
for step, t in enumerate(ts):
    print(f"start step {step + 1} ", end='')
    
    # 强迫场生成逻辑对齐
    force_next = forcefield.force_5cn(t, data_config['lats'], data_config['lons'])
    
    # ONNX 字典映射输入
    dynamic_next, = net.run(None, {
        'const': const,
        'dynamic_t1': dynamic_t1,
        'dynamic_t2': dynamic_t2,
        'force_t1': force_t1,
        'force_t2': force_t2,
        'force_next': force_next
    })
    
    # 状态滚动更新
    force_t1, force_t2 = force_t2, force_next
    dynamic_t1, dynamic_t2 = dynamic_t2, dynamic_next

print('cost:', time()-start)
print(dynamic_next.mean(), dynamic_next.min(), dynamic_next.max(),
      '\nshould be 13425.764 -2993.4424 204544.72')
