from einops import rearrange
from torch import nn, einsum
import torch

# 定义一个pair函数，用于确保输入是一个元组（如果输入是单一的值，转换为元组）
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 定义PreNorm类，实现预处理的LayerNorm操作，作为网络中的每层前向传播的一部分
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 初始化LayerNorm层，归一化维度为dim
        self.fn = fn  # 函数fn是网络层，如Attention或FeedForward

    def forward(self, x, **kwargs):
        # 在输入数据x上应用LayerNorm，然后通过指定的fn函数（Attention或FeedForward）
        return self.fn(self.norm(x), **kwargs)

# 定义FeedForward类，实现前馈神经网络层（通常用于Transformer的MLP部分）
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # 定义一个包含两个线性层的序列，中间使用GELU激活和Dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 输入到隐藏层
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),  # Dropout操作
            nn.Linear(hidden_dim, dim),  # 隐藏层到输出层
            nn.Dropout(dropout)  # Dropout操作
        )

    def forward(self, x):
        # 前向传播，直接通过网络
        return self.net(x)

# 定义SummaryMixing类，用于计算输入特征的摘要
class SummaryMixing(nn.Module):
    def __init__(self, input_dim, dimensions_f, dimensions_s, dimensions_c):
        super().__init__()

        self.local_norm = nn.LayerNorm(dimensions_f)
        self.summary_norm = nn.LayerNorm(dimensions_s)

        self.s = nn.Linear(input_dim, dimensions_s)
        self.f = nn.Linear(input_dim, dimensions_f)
        self.c = nn.Linear(dimensions_s + dimensions_f, dimensions_c)

    def forward(self, x):
        local_summ = torch.nn.GELU()(self.local_norm(self.f(x)))  # 本地特征摘要
        time_summ = self.s(x)  # 时间维度的特征摘要
        time_summ = torch.nn.GELU()(self.summary_norm(torch.mean(time_summ, dim=1)))  # 计算全局时间摘要
        time_summ = time_summ.unsqueeze(1).repeat(1, x.shape[1], 1)  # 扩展时间摘要
        out = torch.nn.GELU()(self.c(torch.cat([local_summ, time_summ], dim=-1)))  # 将本地和全局摘要拼接后计算输出
        return out

# 定义MultiHeadSummary类，实现多头摘要
class MultiHeadSummary(nn.Module):
    def __init__(self, nheads, input_dim, dimensions_f, dimensions_s, dimensions_c, dimensions_projection):
        super().__init__()

        self.mixers = nn.ModuleList([])
        for _ in range(nheads):
            self.mixers.append(SummaryMixing(input_dim=input_dim, dimensions_f=dimensions_f, dimensions_s=dimensions_s,
                                             dimensions_c=dimensions_c))

        self.projection = nn.Linear(nheads * dimensions_c, dimensions_projection)

    def forward(self, x):
        outs = []
        for mixer in self.mixers:
            outs.append(mixer(x))

        outs = torch.cat(outs, dim=-1)
        out = self.projection(outs)

        return out

# 定义Transformer类，实现Transformer架构，采用线性化自注意力机制
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        # 定义Transformer的层：由多个PreNorm（含注意力机制和前馈网络）组成
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            # 使用PreNorm（含注意力机制和前馈网络）
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadSummary(heads, dim, dim_head, dim_head, mlp_dim, dim)),  # 使用多头摘要
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 使用前馈网络
            ]))

    def forward(self, x):
        # 前向传播：遍历每层Transformer
        for attn, ff in self.layers:
            x = attn(x) + x  # 添加残差连接
            x = ff(x) + x  # 添加残差连接
        return x  # 返回最终的输出