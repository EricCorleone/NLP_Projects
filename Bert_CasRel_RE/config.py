"""
配置文件
"""
import os
import re
from glob import glob
from torch import cuda


REL_PATH = './data/output/rel.csv'
if not os.path.exists('./data/output/'):
    os.makedirs('./data/output/')

REL_SIZE = 48
SCHEMA_PATH = './data/input/duie/duie_schema.json'

TRAIN_JSON_PATH = './data/input/duie/duie_train.json'
TEST_JSON_PATH = './data/input/duie/duie_test.json'
DEV_JSON_PATH = './data/input/duie/duie_dev.json'

BERT_MODEL_NAME = 'bert-base-chinese'

BERT_PAD_ID = 0

DEVICE = 'cuda' if cuda.is_available() else 'cpu'

BATCH_SIZE = 8
BERT_DIM = 768
LR = 1e-5
EPOCH = 5

MODEL_DIR = f'./data/output/models/(kaggle) lr={LR}, bs={BATCH_SIZE}/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_PATHS = sorted(glob(MODEL_DIR + 'model_*.pth'), key=lambda x: int(re.findall(r'\d+', x)[-1]))

FIGURE_DIR = f'./figures/(kaggle) lr={LR}, bs={BATCH_SIZE}'
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

SUB_HEAD_BAR = 0.5
SUB_TAIL_BAR = 0.5
OBJ_HEAD_BAR = 0.5
OBJ_TAIL_BAR = 0.5

EPS = 1e-10

CLS_WEIGHT_COEF = [0.3, 1.]
SUB_WEIGHT_COEF = 3

PREDICT_RESULT_PATH = './predict_results.log'