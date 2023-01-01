"""
工具函数
"""
import torch
from torch.utils import data
from seqeval.metrics import classification_report
from config import *
import pandas as pd
import matplotlib.pyplot as plt


def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    # id2word, word2id
    return list(df['word']), dict(df.values)


def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    # id2label, label2id
    return list(df['label']), dict(df.values)


class Dataset(data.Dataset):
    def __init__(self, type='train', base_len=50):
        """
        Args:
            type: 数据集类型（'train'/'test'）
            base_len: 句子长度
        """
        super().__init__()
        self.base_len = base_len
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        self.df = pd.read_csv(sample_path, names=['word', 'label'])
        self.points = self.get_points()
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()

    def get_points(self):
        """
        计算分割点
        """
        points = [0]
        i = 0
        while True:
            if i + self.base_len >= len(self.df):
                points.append(len(self.df))
                break
            if self.df.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                points.append(i)
            else:
                # 如果i+50处的标签不为O, 则i+=1并进入下一个循环
                # 以查询i+51处标签是否为O
                i += 1

        return points

    def __len__(self):
        return len(self.points) - 1

    def __getitem__(self, index):
        df = self.df[self.points[index]:self.points[index + 1]]
        word_unk_id = self.word2id[WORD_UNK]
        label_o_id = self.label2id['O']
        input = [self.word2id.get(w, word_unk_id) for w in df['word']]
        target = [self.label2id.get(l, label_o_id) for l in df['label']]
        return input, target


def collate_fn(batch):
    """
    批处理函数

    Args:
        batch(list): 一个批次的数据（batch_size * (input, target)）

    Returns:
        经过批处理后的数据
    """
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = len(batch[0][0])
    input, target, mask = [], [], []
    for item in batch:
        pad_len = max_len - len(item[0])
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        mask.append([1] * len(item[0]) + [0] * pad_len)
    return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool()


def plot(iters, losses, title):
    """
    用Matplotlib生成损失函数图
    """
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.plot(iters, losses)
    plt.grid()
    plt.title(title)
    plt.show()


def report(y_true, y_pred):
    """
    seqeval评估函数
    """
    return classification_report(y_true, y_pred)


def extract(label, text):
    i = 0
    res = []
    while i < len(label):
        if label[i] != 'O':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and label[i] == 'I-' + name:
                end = i
                i += 1
            res.append([name, text[start:end + 1]])
        else:
            i += 1
    return res


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=100, collate_fn=collate_fn)
    print(iter(loader).__next__())
    d = Dataset()
    print(len(d))