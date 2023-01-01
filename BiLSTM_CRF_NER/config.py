"""
配置文件
"""
import os
import re
import torch
from glob import glob

ORIGIN_DIR = './input/origin/'
ANNOTATION_DIR = './output/annotation/'

TRAIN_SAMPLE_PATH = './output/train_sample.txt'
TEST_SAMPLE_PATH = './output/test_sample.txt'
TEMP_PATH = '/output/temp.txt'

VOCAB_PATH = './output/vocab.txt'
LABEL_PATH = './output/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 3000
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 31
LR = 1e-3
EPOCH = 40
BATCH_SIZE = 64

MODEL_DIR = './output/models/1e-3_bs64/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_PATHS = sorted(glob(MODEL_DIR + 'model_*.pth'), key=lambda x: int(re.findall(r'\d+', x)[0]))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
