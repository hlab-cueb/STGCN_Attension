import torch
import torch.nn as nn

from attention import MultiHeadAttention
from stgcn import STGCNBlock, TimeBlock


# 定义cross attention的gcn模块
class STGCNCatAttBlock(nn.Module):
    def __init__(self, node_size, feature_dim1, feature_dim2, temp_dim, time_dim):
        super(STGCNCatAttBlock, self).__init__()
        self.temp_out_channels = temp_dim
        self.temp_spatial_channels = 16
        self.head_num = 8
        self.time_dim = time_dim
        self.d_model = self.time_dim * self.temp_out_channels
        # 定义相关的操作 stgcn的block
        self.stb1 = STGCNBlock(in_channels=feature_dim1, out_channels=self.temp_out_channels,
                               spatial_channels=self.temp_spatial_channels, num_nodes=node_size)

        self.stb2 = STGCNBlock(in_channels=feature_dim2, out_channels=self.temp_out_channels,
                               spatial_channels=self.temp_spatial_channels, num_nodes=node_size)

        self.att1 = MultiHeadAttention(self.d_model, self.head_num)
        self.att2 = MultiHeadAttention(self.d_model, self.head_num)
        # 对输出进行降维度
        self.out = nn.Linear(self.d_model * 2, self.d_model)

    def forward(self, input1, input2, a_hat, prod_emb=None):
        """
        d_model = T' * out_channel
        """
        s1 = self.stb1(input1, a_hat)  # [B*P, N, T', out_channels]
        s2 = self.stb2(input2, a_hat)  # [B*P, N, T', out_channels]

        # [B*P, N, T', out_channels]-view→ [B*P, N, T'*out_channels]
        s1 = s1.view(s1.shape[0], -1, self.d_model)
        s2 = s2.view(s2.shape[0], -1, self.d_model)

        if prod_emb is not None:
            a1 = self.att1(prod_emb, context=s1)
            a2 = self.att2(prod_emb, context=s2)

        else:
            # attetnion的feature
            a1 = self.att1(s1, s2)  # [B*P, N, T'*out_channels]
            a2 = self.att2(s2, s1)  # [B*P, N, T'*out_channels]

        # 合并两个特征
        c1 = torch.cat([a1, a2], dim=2)  # [B*P, N, 2T'*out_channels]
        # 转化输出
        o1 = self.out(c1)  # [B*P, N, T'*out_channels]
        o1 = o1.view(o1.shape[0], o1.shape[1], self.head_num, -1)  # [B*P, N, H=T'=8, out_channels]

        return o1


# 定义self attention的gcn模块
class STGCNAttBlock(nn.Module):
    def __init__(self, node_size, feature_dim1, time_dim):
        super(STGCNAttBlock, self).__init__()
        self.temp_out_channels = 64
        self.temp_spatial_channels = 16
        self.head_num = 4
        self.time_dim = time_dim
        self.d_model = self.temp_out_channels * self.time_dim
        # 定义相关的操作 stgcn的block
        self.stb1 = STGCNBlock(in_channels=feature_dim1, out_channels=self.temp_out_channels,
                               spatial_channels=self.temp_spatial_channels, num_nodes=node_size)

        self.att1 = MultiHeadAttention(self.d_model, self.head_num)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, input1, a_hat):
        # [B*P, N, T', out_channels]
        s1 = self.stb1(input1, a_hat)  # [B*P, N, T'', out_channels'=out_channels]

        s1 = s1.view(s1.shape[0], -1, self.d_model)  # [B*P, N, T''*out_channels]

        # attetnion的feature
        a1 = self.att1(s1)  # [B*P, N, T''*out_channels]
        # 转化输出
        o1 = self.out(a1)  # [B*P, N, T''*out_channels]
        o1 = o1.view(o1.shape[0], o1.shape[1], self.head_num, -1)  # [B*P, N, T'', out_channels]

        return o1


### 定义模型
class STGCNAttModel(nn.Module):
    def __init__(self, node_size, feature_dim1, feature_dim2, emb_dim,
                 num_timesteps_input, num_timesteps_output):
        super(STGCNAttModel, self).__init__()
        self.temp_dim = 64
        self.time_dim = num_timesteps_input - 4
        self.d_model = self.temp_dim * self.time_dim

        # 主体模块

        '''FiLM'''
        # self.emb_proj = nn.Sequential(
        #     nn.Linear(768, emb_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(emb_dim)
        # )
        #
        # self.res_weight = nn.Parameter(torch.tensor(0.1))
        # self.film_gamma = nn.Linear(emb_dim, feature_dim1)
        # self.film_beta = nn.Linear(emb_dim, feature_dim1)

        '''Attention'''
        # self.emb_proj = nn.Sequential(
        #     nn.Linear(768, self.d_model),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.d_model)
        # )

        self.att_block1 = STGCNCatAttBlock(node_size, feature_dim1, feature_dim2, self.temp_dim,
                                           self.time_dim)
        self.att_block2 = STGCNAttBlock(node_size, self.temp_dim, self.time_dim - 4)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=76)
        self.fully = nn.Linear(self.time_dim - 6, num_timesteps_output)

    def forward(self, input1, input2, prod_emb, a_hat):
        """
        input1: [B, P, X, N, F1](贸易流特征)
        input2: [B, P, X, N, F2](节点特征，比如天气等)
        prod_emb: [P, 768](对商品名称进行的word-embedding向量)
        a_hat: [B, P, N, N](每个商品、每个样本的邻接矩阵)
        """

        B, P, X, N, _ = input1.shape

        '''FiLM'''
        # prod_emb = self.emb_proj(prod_emb)
        # prod_emb = prod_emb.unsqueeze(0).expand(B, -1, -1)  # [B, P, D]
        # prod_emb_flat = prod_emb.reshape(B * P, -1)  # [B*P, D]
        #
        # gamma = self.film_gamma(prod_emb_flat).unsqueeze(-1).unsqueeze(-1)  # [B*P, F1, 1, 1]
        # beta = self.film_beta(prod_emb_flat).unsqueeze(-1).unsqueeze(-1)  # [B*P, F1, 1, 1]
        #
        # # [B*P, F1, N, X] → [B*P, X, N, F1] → [B,P,X,N,F1]
        # input1 = input1.view(B * P, X, N, -1).permute(0, 3, 2, 1)  # [B*P, F1, N, X]
        # input1_mod = (input1 * torch.sigmoid(gamma) + beta) + self.res_weight * input1  # [B*P, F1, N, X]
        # input1_flat = input1_mod.permute(0, 2, 3, 1)

        '''Attention'''
        # prod_emb = self.emb_proj(prod_emb)  # [P, d_model]
        # prod_emb = prod_emb.unsqueeze(0).expand(B, -1, -1)  # [B,P,d_model]
        # prod_emb = prod_emb.reshape(B * P, 1, -1).expand(-1, N, -1)  # [B*P,N,d_model]

        # [B,P,X,N,F] → [B*P, X, N, F] → permute → [B*P, N, X, F]
        input1_flat = input1.reshape(B * P, X, N, -1).permute(0, 2, 1, 3)
        input2_flat = input2.reshape(B * P, X, N, -1).permute(0, 2, 1, 3)
        a_hat_flat = a_hat.reshape(B * P, N, N)

        # 模型主体
        # STGCNBlock input: (batch_size, num_nodes, num_timesteps, num_features=in_channels)

        # at1 = self.att_block1(input1_flat, input2_flat, a_hat_flat, prod_emb=prod_emb)  # [B*P, N, T', out_channels']
        at1 = self.att_block1(input1_flat, input2_flat, a_hat_flat)
        at2 = self.att_block2(at1, a_hat_flat)  # [B*P, N, T'', out_channels]

        last_temp = self.last_temporal(at2)  # [B*P, N, T'''=2, out_channels=N]

        last_temp = last_temp.permute(0, 3, 1, 2)  # [B*P, N, N, T'''=2]
        out_flat = self.fully(last_temp)  # [B*P, N, N, 1]

        out = out_flat.reshape(B, P, N, N, -1).permute(0, 1, 4, 3, 2)  # [B, P, 1, N, N]

        return out
