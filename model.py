import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import *
from dataset import *
from utils import *
from transformer import Transformer
from abmil import BatchedABMIL


class Feed(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Feed, self).__init__()
        # 初始化各层
        self.fc6 = nn.Linear(X_dim, 1024)
        self.fc6_bn = nn.BatchNorm1d(1024)
        self.fc7 = nn.Linear(1024, 2048)
        self.fc7_bn = nn.BatchNorm1d(2048)
        self.fc8 = nn.Linear(2048, 2048)
        self.fc8_bn = nn.BatchNorm1d(2048)
        self.fc9 = nn.Linear(2048, gene_number)

    def forward(self, z, relu):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        if relu:
            return F.relu(self.fc9(h8))
        else:
            return self.fc9(h8)








class ABMILEmbedder(nn.Module):
    def __init__(
            self,
            pre_attention_params: dict = None,
            attention_params: dict = None,
            aggregation: str = 'regular',
    ) -> None:
        super(ABMILEmbedder, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre_attention_params = pre_attention_params
        if pre_attention_params is not None:
            self._build_pre_attention_params(params=pre_attention_params)

        self.attention_params = attention_params
        if attention_params is not None:
            self._build_attention_params(
                attn_model=attention_params['model'],
                params=attention_params['params']
            )

        self.agg_type = aggregation  #

    def _build_pre_attention_params(self, params):
        self.pre_attn = nn.Sequential(
            nn.Linear(params['input_dim'], params['hidden_dim']),
            nn.LayerNorm(params['hidden_dim']),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.LayerNorm(params['hidden_dim']),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _build_attention_params(self, attn_model='ABMIL', params=None):
        if attn_model == 'ABMIL':
            self.attn = BatchedABMIL(**params)
        else:
            raise NotImplementedError('Attention model not implemented -- Options are ABMIL, PatchGCN and TransMIL.')

    def forward(
            self,
            bags: torch.Tensor,
            return_attention: bool = False,
    ) -> torch.tensor:

        if self.pre_attention_params is not None:
            embeddings = self.pre_attn(bags)
        else:
            embeddings = bags

        if self.attention_params is not None:
            if return_attention:
                attention, raw_attention = self.attn(embeddings, return_raw_attention=True)
            else:
                attention = self.attn(embeddings)

        if self.agg_type == 'regular':
            embeddings = embeddings * attention
            if self.attention_params["params"]["activation"] == "sigmoid":
                slide_embeddings = torch.mean(embeddings, dim=1)
            else:
                slide_embeddings = torch.sum(embeddings, dim=1)

        else:
            raise NotImplementedError('Agg type not supported. Options are "additive" or "regular".')

        if return_attention:
            return slide_embeddings, raw_attention

        return slide_embeddings


class TriHRGE(nn.Module):
    def __init__(self, in_features, depth, heads, n_genes=1000, dropout=0.):
        super(TriHRGE, self).__init__()
        self.x_embed = nn.Embedding(512, in_features)
        self.y_embed = nn.Embedding(512, in_features)
        self.trans = Transformer(dim=in_features, depth=depth, heads=heads, dim_head=64, mlp_dim=in_features,
                                 dropout=dropout)

        pre_params = {
            "input_dim": 1024,
            "hidden_dim": 1024,
        }

        attention_params = {
            "model": "ABMIL",
            "params": {
                "input_dim": 1024,
                "hidden_dim": 1024,
                "dropout": True,
                "activation": "softmax",
                "n_classes": 1,
            },
        }
        self.wsi_embedder = ABMILEmbedder(
            pre_attention_params=pre_params,
            attention_params=attention_params,
        )

        # 基因预测头
        self.gene_head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, n_genes)
        )
        self.layer_norm = nn.LayerNorm(in_features)
        self.feed = Feed(1000, 1024)

    def forward(self, image, centers):
        centers_x = self.x_embed(centers[:, :, 0].long())
        centers_y = self.y_embed(centers[:, :, 1].long())
        image = image.squeeze(0)
        wsi_emb = self.wsi_embedder(image)
        y = self.feed(wsi_emb, True)

        return wsi_emb, y
