"""
配置文件
"""
import os
import re
import torch

from glob import glob

ROOT_PATH = os.path.dirname(__file__)

TRAIN_CSV_DIR = os.path.join(ROOT_PATH, 'output/train/csv_label/')
TRAIN_GRAPH_DIR = os.path.join(ROOT_PATH, 'output/train/graph/')

TEST_CSV_DIR = os.path.join(ROOT_PATH, 'output/test/csv_label/')
TEST_GRAPH_DIR = os.path.join(ROOT_PATH, 'output/test/graph/')

WORD_UNK = '<UNK>'
WORD_UNK_ID = 0
VOCAB_SIZE = 50

VOCAB_PATH = os.path.join(ROOT_PATH, 'output/vocab.txt')
LABEL_PATH = os.path.join(ROOT_PATH, 'output/label.txt')

EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 10
LR = 1e-3
EPOCH = 100

MODEL_DIR = os.path.join(ROOT_PATH, 'output/models/')
MODEL_PATHS = sorted(glob(MODEL_DIR + 'model_*.pth'), key=lambda x: int(re.findall(r'\d+', x)[0]))


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(TEST_CSV_DIR)
