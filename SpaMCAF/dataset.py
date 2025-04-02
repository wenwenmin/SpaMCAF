import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from graph_construction import calcADJ


class MyDatasetTrans(Dataset):

    def __init__(self, train, normed_data, coor_df, image, transform=None):
        """

        Args:
            train: 是否为训练模式，影响特征键的选择
            normed_data: 从原始 AnnData 对象中提取的归一化数据
            coor_df: 从原始 AnnData 对象中提取的空间坐标
            image: 包含图像特征的字典
            transform (callable, optional): 可选的转换函数，将应用于样本
        """
        self.data = normed_data.values.T
        self.coor_df = coor_df.values
        self.image = image
        self.transform = transform

        self.coord = np.array_split(self.coor_df, np.ceil(len(self.coor_df) / 50))
        self.exp = np.array_split(self.data, np.ceil(len(self.data) / 50))

        if train:
            self.image_feature_224 = np.array_split(
                self.image['sample_features_224'],
                np.ceil(len(self.image['sample_features_224']) / 50),
            )
            self.image_feature_672 = np.array_split(
                self.image['sample_features_672'],
                np.ceil(len(self.image['sample_features_672']) / 50),
            )
        else:
            self.image_feature_224 = np.array_split(
                self.image['fill_features_224'],
                np.ceil(len(self.image['fill_features_224']) / 50),
            )
            self.image_feature_672 = np.array_split(
                self.image['fill_features_672'],
                np.ceil(len(self.image['fill_features_672']) / 50),
            )

        self.adj = [calcADJ(coord=i, k=4, pruneTag='NA') for i in self.coord]

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        exp = torch.tensor(self.exp[idx])
        coord = torch.tensor(self.coord[idx])
        image_224 = torch.tensor(self.image_feature_224[idx]).unsqueeze(1)
        image_672 = torch.tensor(self.image_feature_672[idx]).unsqueeze(1)

        image = torch.concat((image_224, image_672), dim=1)

        sample = (exp, coord, image)
        return sample


class MyDatasetTrans2(Dataset):

    def __init__(self, coor_df, image, transform=None):
        """
        Args:
            coor_df: 从原始 AnnData 对象中提取的空间坐标
            image: 包含图像特征的字典
            transform (callable, optional): 可选的转换函数，将应用于样本
        """
        self.coor_df = coor_df.values
        self.image = image
        self.transform = transform

        required_keys = ['fill_features_224', 'fill_features_672']
        for key in required_keys:
            if key not in self.image:
                raise KeyError(f"Missing required key '{key}' in image dictionary.")

        self.coord = np.array_split(self.coor_df, np.ceil(len(self.coor_df) / 50))
        self.image_feature_224 = np.array_split(
            self.image['fill_features_224'],
            np.ceil(len(self.image['fill_features_224']) / 50),
        )
        self.image_feature_672 = np.array_split(
            self.image['fill_features_672'],
            np.ceil(len(self.image['fill_features_672']) / 50),
        )

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        coord = torch.tensor(self.coord[idx])
        image_224 = torch.tensor(self.image_feature_224[idx]).unsqueeze(1)
        image_672 = torch.tensor(self.image_feature_672[idx]).unsqueeze(1)

        image = torch.concat((image_224, image_672), dim=1)

        sample = (coord, image)
        return sample