import numpy as np
import torch
from scipy.spatial import distance


def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='NA'):
    """

    参数:
        coord: 空间坐标矩阵，形状为 (n, 2)，n 为样本数，2 为坐标维度
        k: 每个点选择的邻居数（默认值为 4）
        distanceType: 距离度量方法（默认是欧几里得距离 'euclidean'）
        pruneTag: 用于过滤邻居的标签（'NA'：不做修剪；'STD'：根据标准差修剪；'Grid'：根据网格大小修剪）

    返回:
        Adj: 邻接矩阵，形状为 (n, n)，表示节点之间的连通性
    """

    spatialMatrix = coord
    nodes = spatialMatrix.shape[0]
    Adj = torch.zeros((nodes, nodes))

    for i in np.arange(spatialMatrix.shape[0]):
        tmp = spatialMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, spatialMatrix, distanceType)

        if k == 0:
            k = spatialMatrix.shape[0] - 1

        res = distMat.argsort()[:k + 1]

        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)

        for j in np.arange(1, k + 1):
            if pruneTag == 'NA':
                Adj[i][res[0][j]] = 1.0

            elif pruneTag == 'STD':
                if distMat[0, res[0][j]] <= boundary:
                    Adj[i][res[0][j]] = 1.0

            elif pruneTag == 'Grid':
                if distMat[0, res[0][j]] <= 2.0:
                    Adj[i][res[0][j]] = 1.0

    return Adj
