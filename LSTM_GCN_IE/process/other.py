"""
生成词表和标签表
"""
import sys
from collections import Counter

sys.path.append('..')
from utils import *
from config import *


def generate_vocab():
    vocab_list = []
    for file_path in glob(os.path.join(TRAIN_CSV_DIR, '*.csv')):
        df = pd.read_csv(file_path, usecols=['text'])
        for text in df['text'].values:
            text = replace_text(text)
            vocab_list += list(text)
    vocab_list = pd.Series(vocab_list)
    # vocab_list.value_counts().to_csv('vocab_counts.csv', header=False)
    vocab = [WORD_UNK] + vocab_list.value_counts().keys().tolist()
    vocab = vocab[:VOCAB_SIZE]
    vocab_dict = {v: k for k, v in enumerate(vocab)}
    vocab_df = pd.DataFrame(vocab_dict.items())
    vocab_df.to_csv(VOCAB_PATH, header=False, index=False)


def generate_label():
    label_list = []
    for file_path in glob(os.path.join(TRAIN_CSV_DIR, '*.csv')):
        df = pd.read_csv(file_path, usecols=['label'])
        for label in df['label'].values:
            label_list.append(label)
    labels = list(Counter(label_list).keys())
    label_dict = {v: k for k, v in enumerate(labels)}
    label_df = pd.DataFrame(label_dict.items())
    label_df.to_csv(LABEL_PATH, header=False, index=False)


if __name__ == '__main__':
    generate_vocab()
    generate_label()
