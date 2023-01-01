"""
配置文件
"""
import os
import re
import torch
from glob import glob

TRAIN_FILE_PATH = './output/process/atepc.train.csv'
TEST_FILE_PATH = './output/process/atepc.test.csv'

BIO_O_ID, BIO_B_ID, BIO_I_ID = 0, 1, 2
BIO_MAP = {'O': BIO_O_ID, 'B-ASP': BIO_B_ID, 'I-ASP': BIO_I_ID}
ENT_SIZE = 3

POLA_N_LABEL, POLA_O_LABEL, POLA_P_LABEL = '-1', '0', '1'
POLA_LABEL2ID = {POLA_N_LABEL: 0, POLA_P_LABEL: 1, POLA_O_LABEL: -1}
POLA_MAP = ['Negative', 'Positive']
POLA_DIM = 2

BERT_PAD_ID = 0
BERT_MODEL_NAME = 'bert-base-chinese'
BERT_DIM = 768

SRD = 3  # Semantic-Relative Distance

BATCH_SIZE = 50
EPOCH = 5
LR = 1e-4

MODEL_DIR = f'./output/models/lr={LR}, bs={BATCH_SIZE}/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_PATHS = sorted(
    glob(os.path.join(MODEL_DIR, 'model_*.pth')),
    key=lambda x: int(re.findall(r'\d+', x)[-1])
)

FIGURE_DIR = f'./figures/lr={LR}, bs={BATCH_SIZE}/'
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
SCALE_RATE = 20

EPS = 1e-10
LCF = 'cdw'  # cdw cdm fusion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(POLA_MAP[-1])
