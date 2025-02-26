import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import elu, softmax, dropout
import torch.nn.functional as F

from torch_geometric.nn import DenseGCNConv, GINConv,GCNConv,global_mean_pool as gep
from torch_geometric.utils import dropout_adj, degree

def normalize_adj(adj):
    """Normalize adjacency matrix for large-scale graphs."""
    # 确保 adj 是 coalesced 的
    adj = adj.coalesce()
    # 获取行和列索引
    row, col = adj.indices()
    values = adj.values()
    # 计算度矩阵
    deg = degree(row, num_nodes=adj.size(0), dtype=torch.float)
    # 计算度矩阵的逆平方根
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # 计算归一化的邻接矩阵的边权重
    row_deg_inv_sqrt = deg_inv_sqrt[row]
    col_deg_inv_sqrt = deg_inv_sqrt[col]
    norm_values = row_deg_inv_sqrt * col_deg_inv_sqrt * values
    # 构建新的稀疏矩阵
    adj_normalized = torch.sparse_coo_tensor(adj.indices(), norm_values, adj.size(), device=adj.device)
    return adj_normalized

# def normalize_adj(adj):
#     D = torch.sparse.sum(adj, dim=1).to_dense()  # 度矩阵
#     D_inv_sqrt = torch.diag(D.pow(-0.5))
#     adj_normalized = torch.sparse.mm(D_inv_sqrt, torch.sparse.mm(adj, D_inv_sqrt))
#     return adj_normalized

class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=128):
        super().__init__()
        # 初始化两个线性变换层，用于生成注意力映射
        # mk: 将输入特征从d_model维映射到S维，即降维到共享内存空间的大小
        self.mk = nn.Linear(d_model, S, bias=False)
        # mv: 将降维后的特征从S维映射回原始的d_model维
        self.mv = nn.Linear(S, d_model, bias=False)
        # 使用Softmax函数进行归一化处理
        self.softmax = nn.Softmax(dim=1)
        # 调用权重初始化函数
        self.init_weights()

    def init_weights(self):
        # 自定义权重初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层的权重进行Kaiming正态分布初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    # 如果有偏置项，则将其初始化为0
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 对批归一化层的权重和偏置进行常数初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对线性层的权重进行正态分布初始化，偏置项（如果存在）初始化为0
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, batch_size):
        # 前向传播函数
        attn = self.mk(queries)  # 使用mk层将输入特征降维到S维
        attn = self.softmax(attn)  # 对降维后的特征进行Softmax归一化处理
        # 对归一化后的注意力分数进行标准化，使其和为1
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)  # 使用mv层将注意力特征映射回原始维度
        out = out.squeeze(0)
        Linear = nn.Linear(in_features=out.shape[1], out_features=256, device='cuda:0')
        out = gep(Linear(out), batch_size)
        return out

class GINModel(nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, dropout=0.5):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU()
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ELU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ELU()
        ))
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, out_channels),
            nn.ELU()
        ))

        # 添加MLP
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels*2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear( out_channels*2, out_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.ELU()
        )

    def forward(self, data):
        x, edge_index, batch = data[0].x, data[0].edge_index, data[0].batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = gep(x, batch)  # 对每个图进行全局平均池化

        # 通过MLP进一步处理
        x = self.mlp(x)
        return x


class Attntopo(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dropout: float = 0.3, alpha: float = 0.2, act=F.elu):
        super(Attntopo, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.act = act

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.mm(input, self.W)
        if self.bias is not None:
            h = h + self.bias

        # Compute attention coefficients
        attention = self._prepare_attentional_mechanism_input(h, edge_index)
        # Apply dropout to attention coefficients
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        # Normalize attention coefficients
        attention = F.softmax(attention, dim=1)

        # Apply attention to the nodes
        h_prime = self._apply_attention(h, edge_index, attention)

        return self.act(h_prime)

    def _prepare_attentional_mechanism_input(self, h, edge_index):
        Wh1 = torch.matmul(h, self.a[:self.out_features, :])
        Wh2 = torch.matmul(h, self.a[self.out_features:, :])
        e = Wh1[edge_index[0]] + Wh2[edge_index[1]]
        return self.leakyrelu(e)

    def _apply_attention(self, h, edge_index, attention):
        # Create a sparse adjacency matrix with attention coefficients
        row, col = edge_index
        values = attention.squeeze()
        adj = torch.sparse_coo_tensor(edge_index, values, size=(h.size(0), h.size(0)), device=h.device)

        # Apply attention to the nodes
        h_prime = torch.sparse.mm(adj, h)
        return h_prime

class TopoGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_hop, bias=True, dropout=0.3, alpha=0.2, act=F.elu, nheads=4):
        super(TopoGCN, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_hop = n_hop
        self.dropout = dropout
        self.alpha = alpha
        self.act = act
        self.nheads = nheads

        # 使用 GCNConv 层
        self.gc1 = GCNConv(in_features, hidden_features)
        self.gc2 = GCNConv(hidden_features*4, out_features)
        self.linear = nn.Linear(hidden_features*4, out_features)

        # 注意力机制
        self.attentions_1 = [Attntopo(in_features=hidden_features, out_features=hidden_features) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention_1_{}'.format(i), attention)

        self.attentions_2 = [Attntopo(in_features=out_features, out_features=out_features) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention_2_{}'.format(i), attention)

    def forward(self, data):
            input, edge_index, batch = data[0].x, data[0].edge_index, data[0].batch
            # 第一层图卷积
            input = self.act(self.gc1(input, edge_index))

            # 多次应用邻接矩阵，加强图特征
            adj = normalize_adj((torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)),
                                                        size=(input.size(0), input.size(0)),  device='cuda:0')))
            for i in range(self.n_hop):
                input = torch.spmm(adj, input)

            # 应用多头注意力机制
            input = torch.cat([att(input, edge_index) for att in self.attentions_1], dim=1)
            input = F.dropout(input, p=self.dropout, training=self.training)

            # 第二层图卷积
            input = self.act(self.gc2(input, edge_index))

            # 再次应用多头注意力机制
            input = torch.cat([att(input, edge_index) for att in self.attentions_2], dim=1)
            input = F.dropout(input, p=self.dropout, training=self.training)

            # 使用线性层调整输出形状
            input = self.linear(input)

            # 使用 global_mean_pool 聚合输出
            output = gep(input, batch)

            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.hidden_features) + ' -> ' \
               + str(self.out_features) + ')'


class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):  #3-1=2
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings

class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1  #3-1=2
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings


class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings

class ContrastLoss(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam):
        super(ContrastLoss, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        # 使用余弦相似度代替原始的相似度计算
        # 计算z1和z2之间的余弦相似度矩阵
        cosine_sim_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        sim_matrix = torch.exp(cosine_sim_matrix / self.tau)
        return sim_matrix

    def forward(self, za, zb, pos):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)

        # 计算余弦相似度矩阵
        cosine_sim_matrix = self.sim(za_proj, zb_proj)

        # 归一化相似度矩阵
        sim_matrix = cosine_sim_matrix / (torch.sum(cosine_sim_matrix, dim=-1, keepdim=True) + 1e-8)

        # 获取正样本位置的索引
        pos_matrix = pos.to_dense()
        pos_matrix = pos_matrix.to(sim_matrix.device)

        # 提取正样本位置的相似度得分
        indices = pos.coalesce().indices()
        pos_scores = sim_matrix[indices[0], indices[1]]

        # 计算损失
        log_pos_scores = torch.log(pos_scores)
        loss_contrastive = -log_pos_scores.mean()

        # 返回损失值和投影后的张量
        return loss_contrastive, torch.cat((za_proj, zb_proj), 1)

class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        return sim_matrix

    def forward(self, za, zb, pos):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        matrix_a2b = self.sim(za_proj, zb_proj)
        matrix_b2a = matrix_a2b.t()

        matrix_a2b = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-8)
        lori_a = -torch.log(matrix_a2b.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_b2a = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-8)
        lori_b = -torch.log(matrix_b2a.mul(pos.to_dense()).sum(dim=-1)).mean()

        return self.lam * lori_a + (1 - self.lam) * lori_b, torch.cat((za_proj, zb_proj), 1)


class AGCL(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, t_ms_dims, embedding_dim=128, dropout_rate=0.1):
        super(AGCL, self).__init__()
        self.output_dim = embedding_dim * 2

        self.affinity_graph_conv = DenseGCNModel(ns_dims, dropout_rate)
        self.feature_graph_attention1 = TopoGCN(d_ms_dims[0], 256, 256, 3)
        self.feature_graph_attention2 = GINModel(t_ms_dims[0], 64, 256)
        # self.semantic_graph_attention1 = TopoGCN(d_ms_dims[0], 256, 256, 3)
        # self.semantic_graph_attention2 = GINModel(t_ms_dims[0], 64, 256)
        self.drug_graph_attention = ExternalAttention(d_ms_dims[0], 64)
        self.target_graph_attention = ExternalAttention(t_ms_dims[0], 64)
        self.drug_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        self.target_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)


    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_pos, target_pos):
        num_d = affinity_graph.num_drug

        #药物-靶标网络结构特征
        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]

        # #药物分子特征
        drug_graph_embedding = self.feature_graph_attention1(drug_graph_batchs)
        # 蛋白质分子特征
        target_graph_embedding = self.feature_graph_attention2(target_graph_batchs)

        # # 药物分子特征
        # drug_graph_embedding = self.semantic_graph_attention1(drug_graph_batchs)
        # # 蛋白质分子特征
        # target_graph_embedding = self.semantic_graph_attention2(target_graph_batchs)


        # 外部注意力获取药物特征
        d_x = torch.unsqueeze(drug_graph_batchs[0].x, 0)
        drug_attention_graph_embedding = self.drug_graph_attention(d_x, drug_graph_batchs[0].batch)
        # 外部注意力获取蛋白质特征
        x = torch.unsqueeze(target_graph_batchs[0].x, 0)
        target_attention_graph_embedding = self.target_graph_attention(x, target_graph_batchs[0].batch)

        # 融合特征
        final_drug_graph_embedding = drug_graph_embedding + drug_attention_graph_embedding
        final_target_graph_embedding = target_graph_embedding + target_attention_graph_embedding

        dru_loss, drug_embedding = self.drug_contrast(affinity_graph_embedding[:num_d], final_drug_graph_embedding,
                                                      drug_pos)
        tar_loss, target_embedding = self.target_contrast(affinity_graph_embedding[num_d:],
                                                          final_target_graph_embedding, target_pos)

        return dru_loss + tar_loss, drug_embedding, target_embedding
        # return final_drug_graph_embedding, final_target_graph_embedding



class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()

        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 4 * dim)
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings

