"""
工具函数
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from config import *


def dump_file(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))


def load_file(file_path):
    return pickle.load(open(file_path, 'rb'))


def replace_text(text):
    text = re.sub('[1-9]', '0', text)
    text = re.sub('[a-zA-Z]', 'A', text)
    return text


def get_vocab():
    """
    获取词汇表

    Returns:
        id2word 和 word2id
    """
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)


def get_label():
    """
    获取标签表

    Returns:
        id2label 和 label2id
    """
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)


def load_data(csv_path):
    """
    加载数据集

    Args:
        csv_path: 标注好的CSV格式样本文件路径

    Returns:
        格式化后的数据集和标签
    """
    _, word2id = get_vocab()
    _, label2id = get_label()
    df = pd.read_csv(csv_path, usecols=['text', 'label'])
    inputs = []
    targets = []
    for text, label in df.values:
        text = replace_text(text)
        inputs.append([word2id.get(w, WORD_UNK_ID) for w in text])
        targets.append(label2id[label])

    return inputs, targets


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


def report(y_true, y_pred, target_names):
    return classification_report(y_true, y_pred, target_names=target_names)


if __name__ == '__main__':
    res = load_data(os.path.join('./output/train/csv_label/34908612.jpeg.csv'))
    print(res)
