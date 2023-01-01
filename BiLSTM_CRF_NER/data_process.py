"""
数据预处理
"""
from glob import glob
import os
import random
import pandas as pd
from config import *


def get_annotation(ann_path):
    """
    根据标注文件生成对应关系

    Args:
        ann_path: 标注文件路径

    Returns:
        关系字典
    """
    with open(ann_path, encoding='utf8') as file:
        anns = {}
        for line in file.readlines():
            arr = line.split('\t')[1].split()
            name = arr[0]
            start, end = int(arr[1]), int(arr[-1])
            # 标注太长，可能有问题
            if end - start > 50:
                continue
            anns[start] = 'B-' + name
            for i in range(start + 1, end):
                anns[i] = 'I-' + name
        return anns


def get_text(txt_path):
    with open(txt_path, encoding='utf8') as file:
        return file.read()


def generate_annotation():
    """
    建立文字和标签对应关系
    """
    for txt_path in glob(ORIGIN_DIR + '*.txt'):
        ann_path = txt_path[:-3] + 'ann'
        anns = get_annotation(ann_path)
        text = get_text(txt_path)
        # 建立文字和标注对应
        df = pd.DataFrame({'word': list(text),
                           'label': ['O'] * len(text)})
        df.loc[anns.keys(), 'label'] = list(anns.values())
        # 导出文件
        file_name = os.path.split(txt_path)[1]
        if not os.path.exists(ANNOTATION_DIR):
            os.makedirs(ANNOTATION_DIR)
        df.to_csv(ANNOTATION_DIR + file_name, header=False, index=False)


def split_sample(test_size=0.3):
    """
    拆分训练集和测试集

    Args:
        test_size: 测试集比例
    """
    files = glob(ANNOTATION_DIR + '*.txt')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    test_files = files[:n]
    train_files = files[n:]
    # 合并文件
    merge_file(train_files, TRAIN_SAMPLE_PATH)
    merge_file(test_files, TEST_SAMPLE_PATH)


def merge_file(files, target_path):
    """
    合并文件

    Args:
        files: 要合并的文件
        target_path: 目标路径
    """
    with open(target_path, 'w', encoding='utf8') as file:
        text = ''
        for f in files:
            text += open(f, encoding='utf8').read()
        file.write(text)


def generate_vocab():
    """
    生成词表
    """
    df = pd.read_csv(TRAIN_SAMPLE_PATH,
                     usecols=[0],  # 指定读取第0列(word列)
                     names=['word'])
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().tolist()
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab = pd.DataFrame(list(vocab_dict.items()))
    vocab.to_csv(VOCAB_PATH, header=False, index=False)


def generate_label():
    """
    生成标签表
    """
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().tolist()
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=False, index=False)


if __name__ == '__main__':
    # 建立文字和标签对应关系
    generate_annotation()
    # 拆分训练集和测试集
    split_sample()
    # 生成词表
    generate_vocab()
    # 生成标签
    generate_label()
