import numpy as np

def swish(x):
    return x / (1.0 + np.exp(-x))

def layer_norm(x, scale, offset, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + offset

def dense(x, w, b):
    return x @ w + b

def scatter(buf, idx, arr):
    np.add.at(buf,(slice(None), idx, slice(None)), arr)

def mlp_block(x, ws, norm=True):
    out = dense(x, ws[0], ws[1])
    out = swish(out)
    out = dense(out, ws[2], ws[3])
    if not norm: return out
    return layer_norm(out, ws[4], ws[5])

def graph_interaction(s_nodes, r_nodes, edges, idx, we, wn):
    senders, receivers = idx[:, 0], idx[:, 1]
    edge_in = np.concatenate([edges, s_nodes[:, senders], r_nodes[:, receivers]], axis=-1)
    edge_delta = mlp_block(edge_in, we)
    agg_msg = np.zeros_like(r_nodes)
    scatter(agg_msg, receivers, edge_delta)
    node_in = np.concatenate([r_nodes, agg_msg], axis=-1)
    node_delta = mlp_block(node_in, wn)
    return edge_delta, node_delta

def grid2mesh(x, g2m_gf, g2m_mf, g2m_ef, g2m_idx, wd):
    latent_grid = mlp_block(np.concatenate([x, g2m_gf], axis=-1), wd['g2m_enc_grid'])
    dummy = np.zeros((x.shape[0], g2m_mf.shape[1], x.shape[-1]), dtype=g2m_mf.dtype)
    latent_mesh = mlp_block(np.concatenate([dummy, g2m_mf], axis=-1), wd['g2m_enc_mesh'])
    
    latent_edge = mlp_block(g2m_ef, wd['g2m_enc_edge'])
    updated_edge, node_delta = graph_interaction(latent_grid, latent_mesh, latent_edge, g2m_idx, wd['g2m_int_edge'], wd['g2m_int_node'])
    grid_out = latent_grid + mlp_block(latent_grid, wd['g2m_grid_self'])
    return grid_out, latent_mesh + node_delta

def mesh2mesh(m2m_ef, m2m_idx, g2m_node_out, wd):
    cur_node = g2m_node_out
    cur_edge = mlp_block(m2m_ef, wd['m2m_enc_edge'])
    for i in range(16):
        e_delta, n_delta = graph_interaction(cur_node, cur_node, cur_edge, m2m_idx,
                wd[f'processor_{i}_edge_mlp'], wd[f'processor_{i}_node_mlp'])
        cur_edge += e_delta
        cur_node += n_delta
    return cur_node, cur_edge

def mesh2grid(m2g_ef, m2g_idx, mem_nf_out, g2m_gf_out, wd):
    latent_edge = mlp_block(m2g_ef, wd['m2g_enc_edge'])
    _, n_delta = graph_interaction(mem_nf_out, g2m_gf_out, latent_edge, m2g_idx, wd['m2g_int_edge'], wd['m2g_int_node'])
    final_grid = g2m_gf_out + n_delta
    return mlp_block(final_grid, wd['head'], norm=False)

class GraphCastModel:
    def __init__(self, graph_body, norm_vector, weights):
        
        self.wd = weights
        self.g2m_gf = np.array(graph_body['g2m_grid_feats'])[None, :]
        self.g2m_mf = np.array(graph_body['g2m_mesh_feats'])[None, :]
        self.g2m_ef = np.array(graph_body['g2m_edge_feats'])[None, :]
        self.g2m_idx = np.array(graph_body['g2m_idx'])
        self.m2m_ef = np.array(graph_body['m2m_edge_feats'])[None, :]
        self.m2m_idx = np.array(graph_body['m2m_idx'])
        self.m2g_ef = np.array(graph_body['m2g_edge_feats'])[None, :]
        self.m2g_idx = np.array(graph_body['m2g_idx'])

        self.b_const, self.k_const = norm_vector['b_const'], norm_vector['k_const']
        self.b_dynamic, self.k_dynamic = norm_vector['b_dynamic'], norm_vector['k_dynamic']
        self.b_force, self.k_force = norm_vector['b_force'], norm_vector['k_force']
        self.k_diff_dynamic = norm_vector['k_diff_dynamic']

    def forward(self, const, dynamic1, dynamic2, force1, force2, force3):
        x = np.concatenate([const, dynamic1, dynamic2, force1, force2, force3], axis=-1)
        norm_b = np.concatenate([self.b_const] + [self.b_dynamic]*2 + [self.b_force]*3)
        norm_k = np.concatenate([self.k_const] + [self.k_dynamic]*2 + [self.k_force]*3)
        x = x.reshape(1, -1, len(norm_b));
        x -= norm_b; x /= norm_k
        g2m_gf_out, g2m_mf_out = grid2mesh(x, self.g2m_gf, self.g2m_mf, self.g2m_ef, self.g2m_idx, self.wd)
        # return g2m_gf_out, g2m_mf_out
        m2m_mf_out, _ = mesh2mesh(self.m2m_ef, self.m2m_idx, g2m_mf_out, self.wd)
        dynamic_delta = mesh2grid(self.m2g_ef, self.m2g_idx, m2m_mf_out, g2m_gf_out, self.wd)
        dynamic_delta = dynamic_delta.reshape(dynamic1.shape)
        
        dynamic_delta *= self.k_diff_dynamic
        return np.add(dynamic_delta, dynamic2, out=dynamic_delta)

if __name__ == '__main__':
    import pickle

    model_dir = '../weights/lite_5mesh_13level_1deg'
    graph_body = dict(np.load(model_dir + '/GraphBody.npz'))
    
    norm_vector = dict(np.load(model_dir + '/NormVector.npz'))
    with open(model_dir + '/GraphWeight.pkl', 'rb') as f:
        weights = pickle.load(f)
    
    model = GraphCastModel(graph_body, norm_vector, weights)
    
    const = np.zeros([181, 360, 2], dtype=np.float32)
    dynamic_t1 = dynamic_t2 = np.zeros([181, 360, 83], dtype=np.float32)
    force_t1 = force_t2 = force_t3 = np.zeros([181, 360, 5], dtype=np.float32)

    y = model.forward(const, dynamic_t1, dynamic_t2, force_t1, force_t2, force_t3)
    print(y.mean(), y.max(), y.min(), '\n should be 9.505517 816.1084 -551.87756')

