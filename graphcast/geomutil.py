import numpy as np
from scipy.spatial import cKDTree

# 笛卡尔转球面
def xyz2phitheta(xyz):
    x, y, z = xyz.T
    return np.array([np.arctan2(y, x), np.arccos(z)]).T

# 球面转经纬度
def phitheta2latlon(phi_theta):
    phi, theta = phi_theta.T
    lon = np.mod(np.rad2deg(phi), 360)
    lat = 90 - np.rad2deg(theta)
    return np.array([lat, lon]).T 

# 经纬度转球面
def latlon2phitheta(latlon):
    phi = np.deg2rad(latlon[:,1])
    theta = np.deg2rad(90 - latlon[:,0])
    return np.array([phi, theta]).T

# 球面转笛卡尔
def phitheta2xyz(phi_theta):
    phi, theta = phi_theta.T
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return np.array([x, y, z]).T

# 笛卡尔转经纬度
def xyz2latlon(xyz):
    return phitheta2latlon(xyz2phitheta(xyz))

# 经纬度转笛卡尔
def latlon2xyz(latlon):
    return phitheta2xyz(latlon2phitheta(latlon))

# 查找范围内的邻居索引
def radius_query_indices(src, des, radius):
    tree = cKDTree(des)
    idx = tree.query_ball_point(src, radius)
    indices = np.arange(len(idx))
    idx1 = np.repeat(indices, [len(i) for i in idx])
    return np.array([idx1, np.concatenate(idx)]).T

# 查找最邻居的n个邻居索引
def knn_query_indices(src, des, k):
    tree = cKDTree(des)
    dists, idx = tree.query(src, k=k)
    indices = np.arange(len(src))
    idx1 = np.repeat(indices, k)
    return np.column_stack([idx.ravel(), idx1])

# 计算端点特征
def count_node_feats(latlon):
    phi_theta = latlon2phitheta(latlon)
    feats = []
    feats.append(np.cos(phi_theta[:,1]))
    feats.append(np.cos(phi_theta[:,0]))
    feats.append(np.sin(phi_theta[:,0]))
    return np.stack(feats, axis=-1)

# 计算边特征
def count_edge_feats(src_latlon, des_latlon, pair, fact=None):
    src_phi_theta = latlon2phitheta(src_latlon)
    des_phi_theta = latlon2phitheta(des_latlon)
    src_xyz = phitheta2xyz(src_phi_theta)
    des_xyz = phitheta2xyz(des_phi_theta)
    # 转局部坐标系
    X = des_xyz
    Y = np.stack([-np.sin(des_phi_theta[:,0]),
                  np.cos(des_phi_theta[:,0]),
                  np.zeros(len(X), dtype=np.float32)], axis=-1)
    Z = np.cross(X, Y)
    XYZ = np.stack([X, Y, Z], axis=-1).transpose(0,2,1)
    dv = src_xyz[pair[:,0]] - des_xyz[pair[:,1]]
    tv = np.matmul(XYZ[pair[:,1]], dv[:,:,None]).squeeze(-1)

    dist = np.linalg.norm(dv, axis=1)
    fact = dist.max() if fact is None else fact
    dist /= fact; tv /= fact
    return np.concatenate([dist[:,None], tv], axis=-1)

# 构建网络结构
def build_graph(mesh_verts, mesh_faces, lats, lons, query_radius, edge_norm=None):
    grid_latlon = np.array(np.meshgrid(lats, lons, indexing='ij')).reshape(2,-1).T
    mesh_latlon = xyz2latlon(mesh_verts)
    grid_verts = latlon2xyz(grid_latlon)

    x = mesh_verts[mesh_faces[:,[0,1,1,2,2,0]].reshape(-1,2)]

    # 1. Grid to Mesh Graph (g2m)
    g2m_idx = radius_query_indices(
        src=grid_verts, des=mesh_verts, radius=query_radius)
    
    g2m_grid_feats = count_node_feats(grid_latlon)
    g2m_mesh_feats = count_node_feats(mesh_latlon)
    g2m_edge_feats = count_edge_feats(
        grid_latlon, mesh_latlon, g2m_idx, fact=None)
    
    # 2. Mesh to Mesh Graph (m2m)
    m2m_idx = mesh_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    m2m_mesh_feats = count_node_feats(mesh_latlon)
    m2m_edge_feats = count_edge_feats(
        mesh_latlon, mesh_latlon, m2m_idx, fact=None)
    
    # 3. Mesh to Grid Graph (m2g)
    m2g_idx = knn_query_indices(grid_verts, mesh_verts, k=3)
    
    # 这里官方用的trimesh，在每个三角形边界，与官方对不上, 但是微调训练不影响
    # m2g_idx = np.array([
    #      np.load('../data_check/mesh_indices.npy'), 
    #      np.load('../data_check/grid_indices.npy')
    # ]).T
    m2g_mesh_feats = count_node_feats(mesh_latlon)
    m2g_grid_feats = count_node_feats(grid_latlon)
    m2g_edge_feats = count_edge_feats(
        mesh_latlon, grid_latlon, m2g_idx, fact=edge_norm)
    
    return {
        'mesh_verts': mesh_verts,
        'mesh_faces': mesh_faces,
        'g2m_grid_feats': g2m_grid_feats,
        'g2m_mesh_feats': g2m_mesh_feats,
        'g2m_edge_feats': g2m_edge_feats,
        'g2m_idx': g2m_idx,
        'm2m_mesh_feats': m2m_mesh_feats,
        'm2m_edge_feats': m2m_edge_feats,
        'm2m_idx': m2m_idx,
        'm2g_mesh_feats': m2g_mesh_feats,
        'm2g_grid_feats': m2g_grid_feats,
        'm2g_edge_feats': m2g_edge_feats,
        'm2g_idx': m2g_idx
    }

if __name__ == '__main__':
    import json, meshutil
    
    with open('../weights/lite_5mesh_13level_1deg/DataConfig.json') as f:
        data_config = json.loads(f.read())
    
    grid_lats = np.linspace(*data_config['lats'], dtype=np.float32)
    grid_lons = np.linspace(*data_config['lons'], dtype=np.float32)
    
    mesh_verts, mesh_faces, finestlen = meshutil.multi_mesh(data_config['mesh_levels'])
    
    graphbody = build_graph(mesh_verts, mesh_faces, grid_lats, grid_lons,
        finestlen*data_config['query_radius_fact'], data_config['edge_norm_fact'])

    np.savez_compressed('../weights/GraphBody.npz', **graphbody)
