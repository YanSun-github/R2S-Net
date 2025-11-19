# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as data
# from torch.nn.parameter import Parameter
# import math, os
# import numpy as np
#
#
#
# ##########################################
# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     Param:
#         in_features, out_features, bias
#     Input:
#         features: N x C (n = # nodes), C = in_features
#         adj: adjacency matrix (N x N)
#     """
#
#     def __init__(self, in_features, out_features, mat_path, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#         adj_mat = np.load(mat_path)
#         self.register_buffer('adj', torch.from_numpy(adj_mat))
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input):
#         b, n, c = input.shape
#         # b 12 2 b 2 16->b 12 16
#         support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
#         # b 12 12 b 12 16-> b12 16
#         output = torch.bmm(self.adj.unsqueeze(0).repeat(b, 1, 1), support)
#         # output = SparseMM(adj)(support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'
#
#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
#         self.bn1 = nn.BatchNorm1d(nhid)  #
#         self.gc2 = GraphConvolution(nhid, nout, mat_path)
#         self.bn2 = nn.BatchNorm1d(nout)
#         # self.dropout = dropout
#
#     def forward(self, x):
#         x = self.gc1(x)
#         x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
#         x = self.bn1(x).transpose(1, 2).contiguous()
#         x = F.relu(x)
#
#         # x = F.dropout(x, self.dropout, training=self.training)
#
#         # x = self.gc2(x)
#         #
#         # x = x.transpose(1, 2).contiguous()
#         # x = self.bn2(x).transpose(1, 2).contiguous()
#         # x = F.relu(x)
#
#         # x = F.relu(self.gc2(x))
#         # x = F.dropout(x, self.dropout, training=self.training)
#         return x
#
#
# class AUwGCN(torch.nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         mat_path = os.path.join(
#             'assets',
#             '{}.npy'.format(opt['dataset'])
#         )
#         self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))
#         # self.graph_embedding = torch.nn.Sequential(GCN(2, 32, 32, mat_path))
#         in_dim = 192  # 24
#
#         # self.memory = Reliable_Memory(2, 24)
#         # self.encoder = Encoder(opt)
#
#         # self._sequential = torch.nn.Sequential(
#         #     torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0,
#         #                     bias=False),
#         #     torch.nn.BatchNorm1d(64),
#         #     torch.nn.ReLU(inplace=True),
#         #
#         #     # # receptive filed: 7
#         #     torch.nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2,
#         #                     bias=False),
#         #     torch.nn.BatchNorm1d(64),
#         #     torch.nn.ReLU(inplace=True),
#         #
#         #     torch.nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2,
#         #                     bias=False),
#         #     torch.nn.BatchNorm1d(64),
#         #     torch.nn.ReLU(inplace=True),
#         # )
#         self._new_sequential = torch.nn.Sequential(
#             torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#
#             torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2,
#                             bias=False),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#         )
#         self._new_clshead=torch.nn.Sequential(
#             torch.nn.Conv1d(192, 64, kernel_size=1, stride=1, padding=0, bias=False),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#
#             torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
#                             ),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv1d(64, 10, kernel_size=3, stride=1, padding=1, bias=False,
#                             ),
#         )
#         self._new_reghead = torch.nn.Sequential(
#             torch.nn.Conv1d(192, 64, kernel_size=1, stride=1, padding=0, bias=False),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#
#             torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
#                             ),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv1d(64, 4*7, kernel_size=3, stride=1, padding=1, bias=False,
#                             ),
#         )
#         # 0:micro(start,end,None),    3:macro(start,end,None),
#         # 6:micro_apex,7:macro_apex,  8:micro_action, macro_action
#         self._classification = torch.nn.Conv1d(
#             64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
#
#         self._init_weight()
#
    # def forward(self, x):
    #     b, t, n, c = x.shape
    #     # x=x.reshape(b,t,n*c)    #[B,T,F]
    #     # embeded_feature = self.encoder(x, self.memory.proto_vectors)#[B,F,T]
    #     x = x.reshape(b * t, n, c)  # (b*t, n, c)
    #     x = self.graph_embedding(x).reshape(b, t, -1).transpose(1,
    #                                                             2)  # (b, C=384=12*32, t) bt 12 2-> bt 12 16 reshape b t 192-> b 192 t
    #     # x = self.graph_embedding(x).reshape(b, t, n, 16)
    #     cls=self._new_clshead(x)
    #     reg=self._new_reghead(x)
    #     x = self._new_sequential(x)  # 79 64 270
    #     # # x = self._classification(x)#79 10 270
    #     # # out1 = self._new_sequential(x)
    #     # # out2 = self._new_sequential(x)
    #     #
    #     # # Concatenate outputs from both sequences along the channel dimension
    #     # # out = torch.cat((out1, out2), dim=1)
    #     #
    #     x = self._classification(x)
    #     return cls,reg,x
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv1d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             if isinstance(m, torch.nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#
#
# if __name__ == "__main__":
#     import yaml
#
#     # load config & params.
#     with open("./config.yaml", encoding="UTF-8") as f:
#         yaml_config = yaml.safe_load(f)
#         dataset = yaml_config['dataset']
#         opt = yaml_config[dataset]
#
#     x = torch.randn((16, 270, 12, 2))  # (b, t, n, c)
#     model = AUwGCN(opt)
#
#     out,reg,x = model(x)
#     print(reg.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn.parameter import Parameter
import math, os
import numpy as np



##########################################mutil v1
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=3):
        """
        初始化编码器.

        参数:
        - input_dim: 输入特征维度 (F)
        - hidden_dims: 每层的特征维度 (例如: [d0, d1, d2, d3, d4, d5, d6])
        - kernel_size: 卷积核大小
        """
        super(TemporalEncoder, self).__init__()

        # 初始投影层 Φ(0)
        self.initial_proj = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=kernel_size, padding=1)

        # 顺序编码器层 {Φ(u) : u ≤ 6}
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dims[u], hidden_dims[u + 1], kernel_size=kernel_size, padding=1),
                nn.MaxPool1d(kernel_size=2)  # 每层时间维度减半
            )
            for u in range(len(hidden_dims) - 1)
        ])

    def forward(self, x):
        """
        前向传播.

        参数:
        - x: 输入张量, 形状为 (batch_size, F, T_in)

        返回:
        - 编码后的特征, 形状为 (batch_size, d6, T6)
        """
        # 初始投影
        x = self.initial_proj(x)  # 形状: (batch_size, d0, T_in)
        features=[]
        features.append(x)
        # 顺序编码器层
        for layer in self.encoder_layers:
            x = layer(x)  # 每层都会降低时间维度并增加特征维度
            features.append(x)
        # 最终编码器输出 x 具有形状 (batch_size, d6, T6)
        return x,features


class TemporalPyramidPooling(nn.Module):
    def __init__(self, input_dim, num_levels=4, kernel_sizes=[2, 4, 8, 16]):
        """
        初始化时间金字塔池层。

        参数:
        - input_dim: 输入特征维度
        - num_levels: 金字塔级别数
        - kernel_sizes: 不同级别的池化窗口大小
        """
        super(TemporalPyramidPooling, self).__init__()
        self.pooling_layers = nn.ModuleList([
            nn.MaxPool1d(kernel_size=k, stride=k) for k in kernel_sizes
        ])
        self.conv1x1 = nn.Conv1d(input_dim, 1, kernel_size=1)  # 用于折叠特征

    def forward(self, x):
        """
        前向传播.

        参数:
        - x: 输入张量, 形状为 (batch_size, input_dim, T_en)

        返回:
        - 拼接后的金字塔特征, 形状为 (batch_size, T_en, 4 + input_dim)
        """
        features = []
        T_en = x.size(2)

        for pool in self.pooling_layers:
            pooled = pool(x)  # 每个池化窗口下的特征
            upsampled = F.interpolate(pooled, size=T_en, mode='linear', align_corners=True)  # 上采样回原始时间维度
            features.append(self.conv1x1(upsampled))  # 维度折叠

        features.append(x)  # 将原始输入特征 f_en 加入到瓶颈层输出
        return torch.cat(features, dim=1)  # 在潜在维度上连接特征


class TemporalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        """
        初始化解码器。

        参数:
        - input_dim: 输入特征维度
        - hidden_dims: 每层解码器的特征维度
        """
        super(TemporalDecoder, self).__init__()
        self.upconvs = nn.ModuleList([
            nn.Conv1d(input_dim if u == 0 else hidden_dims[u - 1], hidden_dims[u], kernel_size=3, padding=1)
            for u in range(len(hidden_dims))
        ])

    def forward(self, x, encoder_outputs):
        """
        前向传播.

        参数:
        - x: 瓶颈层输出, 形状为 (batch_size, input_dim, T_en)
        - encoder_outputs: 编码器的中间输出, 用于跳跃连接

        返回:
        - 解码器的最终输出, 形状为 (batch_size, T, 128)
        """

        for u, upconv in enumerate(self.upconvs):
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)  # 上采样
            x = upconv(x)  # 卷积
            x = x +  encoder_outputs[-(u +2)] # 跳跃连接

        return x
class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale>0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale >0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1+self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x
class mGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Param:
        in_features, out_features, bias
    Input:
        features: N x C (n = # nodes), C = in_features
        adj: adjacency matrix (N x N)
    """

    def __init__(self, in_features, out_features, num_nodes, bias=True):
        super(mGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.adj = Parameter(torch.FloatTensor(num_nodes, num_nodes))
        torch.nn.init.eye_(self.adj)  # 使用单位矩阵初始化邻接矩阵


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        # b 12 2 b 2 16->b 12 16
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        # b 12 12 b 12 16-> b12 16
        adj = (self.adj + self.adj.T) / 2
        adj = adj.to(input.device)  # 迁移到与输入相同的设备
        output = torch.bmm(adj.unsqueeze(0).repeat(b, 1, 1), support)
        # output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SkeletalPooling(nn.Module):
    """骨架池化，将节点成对池化"""
    def __init__(self):
        super(SkeletalPooling, self).__init__()

    def forward(self, x):
        # 池化操作：将输入节点按成对池化（假设输入是 [B, N, F]，N 是节点数）
        pooled_x = F.max_pool1d(x.permute(0, 2, 1), kernel_size=2).permute(0, 2, 1)
        return pooled_x

class SkeletalUnpooling(nn.Module):
    """骨架反池化，将低分辨率的特征还原为高分辨率"""
    def __init__(self):
        super(SkeletalUnpooling, self).__init__()

    def forward(self, x):
        # 反池化操作：将每个节点的特征复制到两个节点上
        unpooled_x = x.repeat_interleave(2, dim=1)
        return unpooled_x

class HourglassModule(nn.Module):
    """沙漏模块，包含下采样和上采样"""
    def __init__(self, in_channels, num_nodes, num_nodes1, num_nodes2):
        super(HourglassModule, self).__init__()


        # # 下采样路径
        # self.conv1 = mGraphConvolution(in_channels, 96,num_nodes)
        # self.bn1 = nn.BatchNorm1d(96)  #
        # self.pool = SkeletalPooling()
        # self.conv2 = mGraphConvolution(96, 128, num_nodes1)
        # self.bn2 = nn.BatchNorm1d(128)  #
        # self.pool1 = SkeletalPooling()
        # self.conv3 = mGraphConvolution(128, 128, num_nodes2)
        # self.bn3 = nn.BatchNorm1d(128)  #
        # # 上采样路径
        # self.unpool = SkeletalUnpooling()
        # self.conv4 = mGraphConvolution(128, 96, num_nodes1)
        # self.bn4 = nn.BatchNorm1d(96)  #
        # self.unpool1 = SkeletalUnpooling()
        # self.conv5 = mGraphConvolution(96, in_channels, num_nodes)
        # self.bn5 = nn.BatchNorm1d(in_channels)  #
        # 残差连接
        # 下采样路径
        self.conv1 = mGraphConvolution(in_channels, 96,num_nodes)
        self.bn1 = nn.BatchNorm1d(96)  #
        self.pool = SkeletalPooling()

        self.conv3 = mGraphConvolution(96, 96, num_nodes1)
        self.bn3 = nn.BatchNorm1d(96)  #
        # 上采样路径

        self.unpool = SkeletalUnpooling()
        self.conv5 = mGraphConvolution(96, in_channels, num_nodes)
        self.bn5 = nn.BatchNorm1d(in_channels)  #


    def forward(self, x):
        # 下采样
        # x0=x
        # x = self.conv1(x)
        # x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        # x = self.bn1(x).transpose(1, 2).contiguous()
        # x1 = F.relu(x)
        #
        # x=self.pool(x1)
        # x = self.conv2(x)
        # x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        # x = self.bn2(x).transpose(1, 2).contiguous()
        # x2 = F.relu(x)
        #
        # x = self.pool1(x2)
        # x = self.conv3(x)
        # x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        # x = self.bn3(x).transpose(1, 2).contiguous()
        # x = F.relu(x)
        # x = self.unpool(x)
        #
        # x=x+x2
        # x = self.conv4(x)
        # x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        # x = self.bn4(x).transpose(1, 2).contiguous()
        # x = F.relu(x)
        # x = self.unpool1(x)
        #
        # x=x+x1
        # x = self.conv5(x)
        # x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        # x = self.bn5(x).transpose(1, 2).contiguous()
        # x = F.relu(x)
        #
        # x=x+x0
        x0 = x
        x = self.conv1(x)
        x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        x = self.bn1(x).transpose(1, 2).contiguous()
        x1 = F.relu(x)
        x = self.pool(x1)



        x = self.conv3(x)
        x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        x = self.bn3(x).transpose(1, 2).contiguous()
        x = F.relu(x)
        x = self.unpool(x)



        x = x + x1
        x = self.conv5(x)
        x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        x = self.bn5(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        x = x + x0


        return x
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Param:
        in_features, out_features, bias
    Input:
        features: N x C (n = # nodes), C = in_features
        adj: adjacency matrix (N x N)
    """

    def __init__(self, in_features, out_features, mat_path, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.adj = nn.Parameter(torch.Tensor(12, 12))
        self.reset_parameters()

        # adj_mat = np.load(mat_path)
        # self.register_buffer('adj', torch.from_numpy(adj_mat))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        stdva = 1. / math.sqrt(self.adj.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.adj.data.uniform_(-stdva, stdva)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        # b 12 2 b 2 16->b 12 16
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        # b 12 12 b 12 16-> b12 16
        output = torch.bmm(self.adj.unsqueeze(0).repeat(b, 1, 1), support)
        # output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCN, self).__init__()
        self.connorm = ContraNorm(dim=12, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False,
                                  positive=False,
                                  identity=False)
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.bn1 = nn.BatchNorm1d(nhid)  #
        self.gc2 = GraphConvolution(nhid, nout, mat_path)
        self.bn2 = nn.BatchNorm1d(nout)
        # self.dropout = dropout

    def forward(self, x):
        x = self.gc1(x)
        x = x.transpose(1, 2).contiguous()  # 21330 16 12 对21330 12 求均值和标准差
        x = self.connorm(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        # x = F.dropout(x, self.dropout, training=self.training)

        # x = self.gc2(x)
        #
        # x = x.transpose(1, 2).contiguous()
        # x = self.bn2(x).transpose(1, 2).contiguous()
        # x = F.relu(x)

        # x = F.relu(self.gc2(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x


class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        mat_path = os.path.join(
            'assets',
            '{}.npy'.format(opt['dataset'])
        )
        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))
        # self.graph_embedding = torch.nn.Sequential(GCN(2, 32, 32, mat_path))
        in_dim = 128  # 24

        # self.memory = Reliable_Memory(2, 24)
        # self.encoder = Encoder(opt)

        # self._sequential = torch.nn.Sequential(
        #     torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0,
        #                     bias=False),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.ReLU(inplace=True),
        #
        #     # # receptive filed: 7
        #     torch.nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2,
        #                     bias=False),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.ReLU(inplace=True),
        #
        #     torch.nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2,
        #                     bias=False),
        #     torch.nn.BatchNorm1d(64),
        #     torch.nn.ReLU(inplace=True),
        # )
        # self.hourglass = HourglassModule(64, 12, 6, 3)
        self._new_sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )
        self._new_clshead=torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
                            ),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(64, 10, kernel_size=3, stride=1, padding=1, bias=False,
                            ),
        )
        self._new_reghead = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False,
                            ),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(64, 4*7, kernel_size=3, stride=1, padding=1, bias=False,
                            ),
        )
        # 0:micro(start,end,None),    3:macro(start,end,None),
        # 6:micro_apex,7:macro_apex,  8:micro_action, macro_action
        self.encoder = TemporalEncoder(input_dim=12*16, hidden_dims=[128, 256, 512])
        self.tpp = TemporalPyramidPooling(input_dim=512)
        self.decoder = TemporalDecoder(input_dim=516, hidden_dims=[256, 128])
        self.deconv = nn.ConvTranspose1d(10, 128, kernel_size=3, stride=1, padding=1)

        # 连接后的降维卷积
        self.conv1x1 = nn.Conv1d(2 * 128, 128, kernel_size=1)
        self._classification = torch.nn.Sequential(
            torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

        )
        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape
        # x=x.reshape(b,t,n*c)    #[B,T,F]
        # embeded_feature = self.encoder(x, self.memory.proto_vectors)#[B,F,T]
        x = x.reshape(b * t, n, c)  # (b*t, n, c)
        x = self.graph_embedding(x).reshape(b, t, -1).transpose(1, 2)  # (b, C=384=12*32, t) bt 12 2-> bt 12 16 reshape b t 192-> b 192 t ##bt 12 64
        # x = self.hourglass(x)#b 12*64 t

        x, encoderouputs = self.encoder(x)
        bottleneck_output = self.tpp(x)
        x = self.decoder(bottleneck_output, encoderouputs)
        #SRM
        cls = self._new_clshead(x)  # b 128 272->b 10 272
        x = self.conv1x1(torch.cat((x, self.deconv(cls)), dim=1))# b 10 272->b 128 272->b 256 272->b 128 272
        # x = self.graph_embedding(x).reshape(b, t, n, 16)

        reg=self._new_reghead(x)
        x = self._new_sequential(x)  # 79 64 270
        # # x = self._classification(x)#79 10 270
        # # out1 = self._new_sequential(x)
        # # out2 = self._new_sequential(x)
        #
        # # Concatenate outputs from both sequences along the channel dimension
        # # out = torch.cat((out1, out2), dim=1)
        #
        x = self._classification(x)
        return cls,reg,x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    import yaml

    # load config & params.
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]

    x = torch.randn((16, 272, 12, 2))  # (b, t, n, c)
    model = AUwGCN(opt)

    out,reg,x = model(x)
    print(x.shape)
