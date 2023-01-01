"""
配置文件
"""
import os
import re
import torch

from glob import glob

TRAIN_SAMPLE_PATH = './data/input/train.txt'
TEST_SAMPLE_PATH = './data/input/test.txt'
DEV_SAMPLE_PATH = './data/input/dev.txt'

LABEL_PATH = './data/input/class.txt'

BERT_PAD_ID = 0
TEXT_LEN = 30

BERT_MODEL = 'bert-base-chinese'
# BERT_MODEL = 'hfl/chinese-macbert-base'

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 10
FILTER_SIZES = [2, 3, 4]

EPOCH = 1
LR = 1e-3
BATCH_SIZE = 100

MODEL_DIR = f'./data/output/models/lr={LR}, bs={BATCH_SIZE}/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_PATHS = sorted(glob(os.path.join(MODEL_DIR, 'model_*.pth')), key=lambda x: int(re.findall(r'\d+', x)[-1]))

FIGURE_DIR = f'./figures/lr={LR}, bs={BATCH_SIZE}'
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
