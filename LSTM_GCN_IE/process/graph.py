"""
创建图节点链接
"""
import sys

import networkx
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

sys.path.append('..')
from utils import *
from config import *


class Graph:
    # 创建链接
    @staticmethod
    def connect(csv_path):
        """
        创建图节点链接
        
        Args:
            csv_path(str): OCR输出的包含节点位置信息的csv文件的路径

        Returns:
            节点链接字典和孤立节点列表
        """
        graph_dict = {}
        df = pd.read_csv(csv_path, index_col=0)
        for src_idx, src_row in df.iterrows():
            neighbor_x = []  # 同一行节点
            neighbor_y = []  # 同一列节点
            # 再次遍历，两两比较
            for dst_idx, dst_row in df.iterrows():
                if src_idx == dst_idx:
                    continue
                # 右边的节点
                if src_row.x2 < dst_row.x1 and src_row.y1 < dst_row.y2 and src_row.y2 > dst_row.y1:
                    # (距离, 节点id)，距离在前方便比较大小
                    neighbor_x.append((dst_row.x1 - src_row.x2, dst_idx))
                # 下边的节点
                if src_row.y2 < dst_row.y1 and src_row.x1 < dst_row.x2 and src_row.x2 > dst_row.x1:
                    neighbor_y.append((dst_row.y1 - src_row.y2, dst_idx))

            # 取最近的节点，其他的忽略
            # 节点右边或下边可能没有节点，取最小值前要做判断
            min_x = [min(neighbor_x)[1]] if neighbor_x else []
            min_y = [min(neighbor_y)[1]] if neighbor_y else []
            graph_dict[src_idx] = min_x + min_y

        # 过滤空节点
        graph_dict = {k: v for k, v in graph_dict.items() if v}

        # 找出孤立点，键和值中都未出现过
        node_idx = set(graph_dict.keys())
        node_idx.update([i for v in graph_dict.values() for i in v])
        isolated_idx = set(df.index) - node_idx

        return graph_dict, list(isolated_idx)

    @staticmethod
    def get_adjacency_norm(graph_dict):
        """
        计算标准化的邻接矩阵

        Args:
            graph_dict(dict): 节点链接字典

        Returns:
            标准化的邻接矩阵
        """
        G = nx.from_dict_of_lists(graph_dict)
        A = nx.adjacency_matrix(G)
        A_new = A + np.eye(*A.shape)
        D = np.array(A_new.sum(1)).flatten()
        # D^(-0.5) A D^(-0.5)
        return np.diag(D ** (-0.5)) @ A_new @ np.diag(D ** (-0.5))

    def connect_get_adjacency_norm(self, csv_path):
        """
        创建图节点链接并计算标准化的邻接矩阵

        Args:
            csv_path(str): OCR输出的包含节点位置信息的csv文件的路径

        Returns:
            标准化的邻接矩阵和孤立节点列表
        """
        graph_dict, isolated_idx = self.connect(csv_path)
        return self.get_adjacency_norm(graph_dict), isolated_idx


if __name__ == '__main__':
    # graph = Graph()
    # graph_dict, isolated_idx = graph.connect(os.path.join(TRAIN_CSV_DIR, '34908612.jpeg.csv'))
    # A = graph.get_adjacency_norm(graph_dict)
    # print(A.shape)

    # 画图
    # G = nx.from_dict_of_lists(graph_dict)
    # fig, ax = plt.subplots()
    # nx.draw(G, ax=ax, with_labels=True)  # 显示节点标签
    # plt.show()

    graph = Graph()
    for file_path in tqdm(glob(os.path.join(TRAIN_CSV_DIR, '*.csv'))):
        adj, isolated_idx = graph.connect_get_adjacency_norm(file_path)
        file_name = os.path.split(file_path)[1][:-3] + 'pkl'
        dump_file([adj, isolated_idx], os.path.join(TRAIN_GRAPH_DIR, file_name))

    for file_path in tqdm(glob(os.path.join(TEST_CSV_DIR, '*.csv'))):
        adj, isolated_idx = graph.connect_get_adjacency_norm(file_path)
        file_name = os.path.split(file_path)[1][:-3] + 'pkl'
        dump_file([adj, isolated_idx], os.path.join(TEST_GRAPH_DIR, file_name))
