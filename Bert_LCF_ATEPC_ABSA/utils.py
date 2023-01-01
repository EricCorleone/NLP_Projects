"""
工具函数
"""
import torch
import random
import pandas as pd
import torch.utils.data as data
import matplotlib.pyplot as plt

from config import *
from model import Model
from transformers import BertTokenizer
from sklearn.metrics import classification_report


def get_ent_pos(lst):
    """
    获取实体位置

    Args:
        lst(list[int]): 实体抽取id列表

    Returns:
        一个列表，其中包含每个实体在句中从开始到结束每个字的索引列表
    """
    items = []
    for i in range(len(lst)):
        if lst[i] == BIO_B_ID:
            item = [i]
            while True:
                i += 1
                if i >= len(lst) or lst[i] != BIO_I_ID:
                    items.append(item)
                    break
                item.append(i)

    return items


# [CLS]这个手机外观时尚，美中不足的是拍照像素低。[SEP]
# 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 1 2 2 2 0 0 0
# print(get_ent_pos([0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,1,2,2,2,0,0,0]))

def get_ent_weight(max_len, ent_pos):
    """
    获取实体的 CDM 和 CDW

    Args:
        max_len(int): 最大句子长度
        ent_pos(list[int]): 实体的索引列表

    Returns:
        上下文特征动态掩码（CDM）和上下文特征动态权重（CDW）列表
    """
    cdm = []
    cdw = []

    for i in range(max_len):
        dst = min(abs(i - ent_pos[0]), abs(i - ent_pos[-1]))
        if dst <= SRD:
            cdm.append(1)
            cdw.append(1.)
        else:
            cdm.append(0)
            cdw.append(1 / (dst - SRD + 1))

    return cdm, cdw


# print(get_ent_weight(23, [5, 6]))
# exit()

def get_pola(model, input_ids, mask, ent_label):
    """
    根据从一个样本中预测出的实体 label，解析出所有实体位置，并预测相应的情感分类

    Args:
        model(Model): 预测模型
        input_ids(torch.Tensor): 字符 ID
        mask(torch.Tensor): 掩码
        ent_label(list[int]): 模型预测出的实体标签

    Returns:
        该样本的所有实体位置和对应的情感分类（线性层输出，未经过激活函数）
    """
    # 变量初始化
    b_input_ids = []
    b_mask = []
    b_ent_cdm = []
    b_ent_cdw = []
    b_ent_pos = []

    # 根据label解析实体位置
    ent_pos = get_ent_pos(ent_label)
    n = len(ent_pos)
    if n == 0:
        return None, None

    # n个实体一起预测，同一个句子复制n份，作为一个“batch”
    b_input_ids.extend([input_ids] * n)
    b_mask.extend([mask] * n)
    b_ent_pos.extend(ent_pos)
    for sg_ent_pos in ent_pos:
        cdm, cdw = get_ent_weight(len(input_ids), sg_ent_pos)
        b_ent_cdm.append(cdm)
        b_ent_cdw.append(cdw)

    # 列表转tensor
    b_input_ids = torch.stack(b_input_ids, dim=0)
    b_mask = torch.stack(b_mask, dim=0)
    b_ent_cdm = torch.tensor(b_ent_cdm, device=DEVICE)
    b_ent_cdw = torch.tensor(b_ent_cdw, device=DEVICE)
    b_ent_pola = model.get_pola(b_input_ids, b_mask, b_ent_cdm, b_ent_cdw)

    return b_ent_pos, b_ent_pola


def plot(iters, losses, f1_scores, last_epoch_count):
    plt.xlabel('iter')
    plt.ylabel(f'loss/f1_score({SCALE_RATE}x)')
    plt.plot(iters, losses, c='red', label='loss')
    plt.plot(iters, f1_scores, c='green', label=f'f1_score({SCALE_RATE}x)')
    plt.legend()
    plt.grid()
    plt.title(f'lr={LR}, bs={BATCH_SIZE}')
    plt.savefig(os.path.join(FIGURE_DIR, f'epoch={last_epoch_count + 1}~{last_epoch_count + EPOCH}.png'))
    plt.show()


class Dataset(data.Dataset):
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()
        if mode == 'train':
            file_path = TRAIN_FILE_PATH
        elif mode == 'test':
            file_path = TEST_FILE_PATH
        else:
            raise ValueError(f'{mode} is not a valid mode, please select one of ["train", "test"]')

        self.df = pd.read_csv(file_path)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # 相邻两个句子拼接
        text1, bio1, pola1 = self.df.loc[index]
        text2, bio2, pola2 = self.df.loc[index + 1]
        text = f'{text1} ; {text2}'
        bio = f'{bio1} O {bio2}'
        pola = f'{pola1} {POLA_O_LABEL} {pola2}'

        # 按自己的规则分词，防止文本存在英文导致BERT自动分词错乱
        tokens = ['[CLS]'] + text.split(' ') + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # BIO标签转id
        bio_arr = ['O'] + bio.split(' ') + ['O']
        bio_label = [BIO_MAP[l] for l in bio_arr]

        # 情感值转数字
        pola_arr = [POLA_O_LABEL] + pola.split(' ') + [POLA_O_LABEL]
        # ['-1', '0', '1'] -> [0, -1, 1]
        pola_label = list(map(lambda l: POLA_LABEL2ID[l], pola_arr))

        input_ids = torch.tensor(input_ids, device=DEVICE)
        bio_label = torch.tensor(bio_label, device=DEVICE)
        pola_label = torch.tensor(pola_label, device=DEVICE)

        return input_ids, bio_label, pola_label

    @staticmethod
    def collate_fn(batch):
        """
        批处理函数

        Args:
            batch(list[tuple[list[int]]]): 一个批次的数据

        Returns:
            一个元组，包含一个批次内每个样本的经过等长填充的字符 ID、掩码、CDM、CDW、实体标签、情感极性标签、实体情感对。
        """
        # 统计最大句子长度
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        max_len = len(batch[0][0])

        # 变量初始化
        batch_input_ids = []
        batch_mask = []
        batch_bio_label = []
        batch_ent_cdm = []
        batch_ent_cdw = []
        batch_pola_label = []
        batch_pairs = []

        for input_ids, bio_label, pola_label in batch:  # type:list[int]
            # 获取实体位置，没有实体则跳过
            ent_pos = get_ent_pos(bio_label)
            if len(ent_pos) == 0:
                continue

            # 填充句子长度
            pad_len = max_len - len(input_ids)
            batch_input_ids.append(input_ids + [BERT_PAD_ID] * pad_len)
            batch_mask.append([True] * len(input_ids) + [False] * pad_len)
            batch_bio_label.append(bio_label + [BIO_O_ID] * pad_len)

            # 实体和情感分类对应
            pairs = []
            for pos in ent_pos:
                pola = pola_label[pos[0]]
                # 异常值替换（不需要，已在 process.py 中处理干净）
                # pola = POLA_N_ID if pola == POLA_O_LABEL else pola
                pairs.append((pos, pola))
            batch_pairs.append(pairs)

            # 随机取一个实体
            sg_ent_pos = random.choice(ent_pos)
            # 计算加权参数
            cdm, cdw = get_ent_weight(max_len, sg_ent_pos)
            batch_ent_cdm.append(cdm)
            batch_ent_cdw.append(cdw)
            # 实体第一个字的情感极性
            pola = pola_label[sg_ent_pos[0]]
            # pola = POLA_N_ID if pola == POLA_O_LABEL else pola
            batch_pola_label.append(pola)

        return (
            torch.tensor(batch_input_ids, device=DEVICE),
            torch.tensor(batch_mask, device=DEVICE),
            torch.tensor(batch_ent_cdm, device=DEVICE),
            torch.tensor(batch_ent_cdw, device=DEVICE),
            torch.tensor(batch_bio_label, device=DEVICE),
            torch.tensor(batch_pola_label, device=DEVICE),
            batch_pairs
        )


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2, collate_fn=Dataset.collate_fn)
    print(iter(loader).__next__())
