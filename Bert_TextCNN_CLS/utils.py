"""
工具函数
"""
import os
import torch

from config import *
from torch.utils import data
import matplotlib.pyplot as plt
from transformers import logging
from transformers import BertTokenizer
from sklearn.metrics import classification_report

logging.set_verbosity_error()


class Dataset(data.Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        if mode == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif mode == 'test':
            sample_path = TEST_SAMPLE_PATH
        elif mode == 'dev':
            sample_path = DEV_SAMPLE_PATH
        else:
            raise ValueError(f'{mode} is not a valid mode, please select one of ["train", "test", "dev"]')

        self.lines = open(sample_path, encoding='utf8').readlines()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text, label = self.lines[index].split('\t')
        tokenized = self.tokenizer(text)
        input_ids = tokenized['input_ids']
        mask = tokenized['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = TEXT_LEN - len(input_ids)
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        target = int(label)

        inputs_ids = torch.tensor(input_ids[:TEXT_LEN], device=DEVICE)
        mask = torch.tensor(mask[:TEXT_LEN], device=DEVICE)
        target = torch.tensor(target, device=DEVICE)

        return inputs_ids, mask, target


def get_label():
    text = open(LABEL_PATH, encoding='utf8').read()
    id2label = text.split()
    return id2label, {v: k for k, v in enumerate(id2label)}


def evaluate(pred, true, target_names=None, output_dict=False):
    return classification_report(
        y_true=true,
        y_pred=pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )


def plot(iters, losses, dev_accs, last_epoch_count, title):
    plt.xlabel('iter')
    plt.ylabel('loss/acc')
    plt.plot(iters, losses, c='red', label='loss')
    plt.plot(iters, dev_accs, c='blue', label='dev_acc')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.savefig(os.path.join(FIGURE_DIR, f'epoch={last_epoch_count + 1}~{last_epoch_count + EPOCH}.png'))
    plt.show()


if __name__ == '__main__':
    # dataset = Dataset()
    # print(len(dataset))count
    #
    # loader = data.DataLoader(dataset, batch_size=2)
    # ipt_ids, msk, trgt = iter(loader).__next__()
    # print(ipt_ids)
    # print(msk)
    # print(trgt)

    print(get_label())
    print(evaluate([1, 2, 3], [1, 1, 1]))
