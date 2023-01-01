"""
工具函数
"""
import sys
import json
import time

import torch
import random
import numpy as np
import pandas as pd
import torch.utils.data as data
import matplotlib.pyplot as plt

from config import *
from model import *
from transformers import BertTokenizerFast


def get_rel():
    """
    获取 id2rel 和 rel2id

    Returns:
        tuple[list[str], dict[str, str]]: id2rel列表和rel2id字典
    """
    df = pd.read_csv(REL_PATH, names=['rel', 'id'])
    return df['rel'].tolist(), dict(df.values)


def multi_hot(length, hot_pos):
    """
    根据位置索引生成独热编码

    Args:
        length(int): 序列长度
        hot_pos(list[int]): 位置索引

    Returns:
        根据位置索引生成的独热编码
    """
    return [1 if i in hot_pos else 0 for i in range(length)]


def timer(name=''):
    """
    计时器
    """

    def _timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            print(f"{name}耗时: {(end_time - start_time):.2f} s\n")

        return wrapper

    return _timer


def plot(f1_iters, f1_scores, last_epoch_count):
    """
    绘图

    Args:
        f1_iters(list[int]): F1迭代次数列表
        f1_scores(list[float]): F1得分列表
        last_epoch_count(int): 训练轮次
    """
    plt.title(f'LR={LR}, BATCH SIZE={BATCH_SIZE}')
    plt.xlabel('iter')
    plt.ylabel('f1_score')
    plt.plot(f1_iters, f1_scores, c='green')
    plt.grid()

    figure_save_path = os.path.join(FIGURE_DIR, f'epoch={last_epoch_count + 1}~{EPOCH}.png')
    plt.savefig(figure_save_path)
    print(f'损失图/F1得分图已保存到：{figure_save_path}')
    # plt.show()


def report(model, encoded_text, pred_y, batch_text, batch_mask):
    """
    评估报告

    Args:
        model(CasRel): 模型
        encoded_text(torch.Tensor): 一个批次的转化成 BERT 词向量的文本
        pred_y(tuple[torch.Tensor]): 一个批次的模型输出结果
        batch_text(dict[str, list]): 一个批次的原始信息
        batch_mask(list[list[bool]]): 一个批次的掩码

    Returns:
        float: F1得分
    """
    # 计算三元结构和统计指标
    # shape (batch_size, max_len, 1)
    pred_sub_head, pred_sub_tail, _, _ = pred_y
    true_triple_list = batch_text['triple_list']
    # pred_triple_list = []

    correct_num, predict_num, gold_num = 0, 0, 0

    # 遍历 batch
    for i in range(len(pred_sub_head)):
        text = batch_text['text'][i]

        mask = batch_mask[i]
        offset_mapping = batch_text['offset_mapping'][i]
        true_triple_item = true_triple_list[i]

        # 将大于阈值（默认为0.5）的结果的索引取出
        sub_head_ids = torch.where(pred_sub_head[i] > SUB_HEAD_BAR)[0]  # size: (num_pred_head, )
        sub_tail_ids = torch.where(pred_sub_tail[i] > SUB_TAIL_BAR)[0]  # size: (num_pred_tail, )

        pred_triple_item = get_triple_list(
            model, sub_head_ids, sub_tail_ids, encoded_text[i], text, mask, offset_mapping
        )

        # 统计个数
        correct_num += len(set(true_triple_item) & set(pred_triple_item))
        predict_num += len(set(pred_triple_item))
        gold_num += len(set(true_triple_item))

        # pred_triple_list.append(pred_triple_item)

    precision = correct_num / (predict_num + EPS)
    recall = correct_num / (gold_num + EPS)
    f1_score = 2 * precision * recall / (precision + recall + EPS)
    print(f'\tcorrect_num: {correct_num}, '
          f'predict_num: {predict_num}, '
          f'gold_num: {gold_num}')
    print(f'\tprecision: {precision:.3f}, '
          f'recall: {recall:.3f}, '
          f'f1_score: {f1_score:.3f}')
    return f1_score


def get_triple_list(model, sub_head_ids, sub_tail_ids, encoded_text, text, mask, offset_mapping):
    """
    获取三元组列表

    Args:
        model(CasRel): 模型
        sub_head_ids(torch.Tensor): 一个样本中预测出的所有主体首字索引
        sub_tail_ids(torch.Tensor): 一个样本中预测出的所有主体尾字索引
        encoded_text(torch.Tensor): 一个样本的转化成 BERT 词向量的文本
        text(str): 一个样本的原始文本
        mask(list[bool]): 一个样本的掩码
        offset_mapping(list[tuple[int]]): 一条原始文本中每个字的偏移映射

    Returns:
         list[tuple[str, str, str]]: 三元组列表
    """
    id2rel, _ = get_rel()
    triple_list = []
    # 遍历所有 head_id，找到离每个 head_id 最近的 tail_id
    for sub_head_id in sub_head_ids:
        sub_tail_ids = sub_tail_ids[sub_tail_ids >= sub_head_id]
        if len(sub_tail_ids) == 0:
            continue
        sub_tail_id = sub_tail_ids[0]
        if mask[sub_head_id] == 0 or mask[sub_tail_id] == 0:
            continue
        # 根据位置信息反推出 subject 文本内容
        sub_head_pos_id = offset_mapping[sub_head_id][0]
        sub_tail_pos_id = offset_mapping[sub_tail_id][1]
        subject_text = text[sub_head_pos_id:sub_tail_pos_id]

        # 根据 subject 计算出对应 object 和 relation
        sub_head_seq = torch.tensor(multi_hot(len(mask), sub_head_id), device=DEVICE)
        sub_tail_seq = torch.tensor(multi_hot(len(mask), sub_tail_id), device=DEVICE)

        pred_obj_head, pred_obj_tail = model.get_objs_for_specific_sub(
            # (max_len, rel_size) -> (1, max_len, rel_size)
            encoded_text.unsqueeze(0),
            sub_head_seq.unsqueeze(0),
            sub_tail_seq.unsqueeze(0)
        )

        # 按分类找对应关系
        # (1, max_len, rel_size) -> (rel_size, max_len)
        pred_obj_head = pred_obj_head[0].T
        pred_obj_tail = pred_obj_tail[0].T
        # 遍历每个 relation
        for j in range(len(pred_obj_head)):
            obj_head_ids = torch.where(pred_obj_head[j] > OBJ_HEAD_BAR)[0]  # size: (num_pred_head, )
            obj_tail_ids = torch.where(pred_obj_tail[j] > OBJ_TAIL_BAR)[0]  # size: (num_pred_tail, )
            for obj_head_id in obj_head_ids:
                obj_tail_ids = obj_tail_ids[obj_tail_ids >= obj_head_id]
                if len(obj_tail_ids) == 0:
                    continue
                obj_tail_id = obj_tail_ids[0]
                if mask[obj_head_id] == 0 or mask[obj_tail_id] == 0:
                    continue
                # 根据位置信息反推出object文本内容，mapping中已有移位，不需要再加一
                obj_head_pos_id = offset_mapping[obj_head_id][0]
                obj_tail_pos_id = offset_mapping[obj_tail_id][1]
                object_text = text[obj_head_pos_id:obj_tail_pos_id]
                triple_list.append((subject_text, id2rel[j], object_text))

    return list(set(triple_list))


class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Dataset(data.Dataset):
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()
        _, self.rel2id = get_rel()
        # 加载文件
        if mode == 'train':
            file_path = TRAIN_JSON_PATH
        elif mode == 'test':
            file_path = TEST_JSON_PATH
        elif mode == 'dev':
            file_path = DEV_JSON_PATH
        else:
            raise ValueError(f'{mode} is not a valid mode, please select from one of ["train", "test", "dev"]')
        with open(file_path, encoding='utf8') as f:
            self.lines = f.readlines()

        # 加载bert
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        info = json.loads(line)
        tokenized = self.tokenizer(info['text'], return_offsets_mapping=True)
        info['input_ids'] = tokenized['input_ids']
        info['offset_mapping'] = tokenized['offset_mapping']

        return self.parse_json(info)

    def parse_json(self, info):
        text = info['text']
        input_ids = info['input_ids']
        dct = {
            'text': text,
            'input_ids': input_ids,
            'offset_mapping': info['offset_mapping'],
            'sub_head_ids': [],
            'sub_tail_ids': [],
            'triple_list': [],
            'triple_id_list': []
        }
        spo_list = info['spo_list']
        for spo in spo_list:
            subject = spo['subject']
            predicate = spo['predicate']
            object = spo['object']['@value']
            dct['triple_list'].append((subject, predicate, object))

            # 计算 subject 实体位置
            tokenized = self.tokenizer(subject, add_special_tokens=False)
            s_token = tokenized['input_ids']
            s_pos_id = self.get_pos_id(input_ids, s_token)
            if not s_pos_id:
                continue
            s_head, s_tail = s_pos_id

            # 计算 object 实体位置
            tokenized = self.tokenizer(object, add_special_tokens=False)
            o_token = tokenized['input_ids']
            o_pos_id = self.get_pos_id(input_ids, o_token)
            if not o_pos_id:
                continue
            o_head, o_tail = o_pos_id

            # 数据组装
            dct['sub_head_ids'].append(s_head)
            dct['sub_tail_ids'].append(s_tail)
            dct['triple_id_list'].append(([s_head, s_tail], self.rel2id[predicate], [o_head, o_tail]))

        return dct

    @staticmethod
    def get_pos_id(source, elem):
        """
        获取首尾字索引

        Args:
            source: 句子
            elem: 目标词

        Returns:
            目标词在句中的首尾字索引
        """
        for head_id in range(len(source)):
            tail_id = head_id + len(elem)
            if source[head_id:tail_id] == elem:
                return head_id, tail_id - 1

    @staticmethod
    def collate_fn(batch):
        """
        批处理函数

        Args:
            batch(list[dict]): 一个批次的数据

        Returns:
            batch_mask, (batch_text, batch_sub_rnd),
                (batch_sub, batch_obj_rel)
        """
        batch.sort(key=lambda x: len(x['input_ids']), reverse=True)
        max_len = len(batch[0]['input_ids'])
        batch_text = {
            'text': [],
            'input_ids': [],
            'offset_mapping': [],
            'triple_list': []
        }
        batch_mask = []
        batch_sub = {
            'heads_seq': [],
            'tails_seq': []
        }
        batch_sub_rnd = {
            'head_seq': [],
            'tail_seq': []
        }
        batch_obj_rel = {
            'heads_mx': [],
            'tails_mx': []
        }

        # 循环解析内容
        for item in batch:
            input_ids = item['input_ids']
            item_len = len(input_ids)
            pad_len = max_len - item_len
            input_ids += [BERT_PAD_ID] * pad_len
            mask = [True] * item_len + [False] * pad_len

            # 填充subject位置
            sub_heads_seq = multi_hot(max_len, item['sub_head_ids'])
            sub_tails_seq = multi_hot(max_len, item['sub_tail_ids'])

            # 随机选择一个subject
            if len(item['triple_id_list']) == 0:
                continue
            sub_rnd = random.choice(item['triple_id_list'])[0]
            sub_rnd_head_seq = multi_hot(max_len, [sub_rnd[0]])
            sub_rnd_tail_seq = multi_hot(max_len, [sub_rnd[1]])

            # 根据随机subject计算relations矩阵
            obj_head_mx = [[0] * REL_SIZE for _ in range(max_len)]
            obj_tail_mx = [[0] * REL_SIZE for _ in range(max_len)]
            for triple in item['triple_id_list']:
                sub_id, rel_id, (obj_head_id, obj_tail_id) = triple
                if sub_id == sub_rnd:
                    obj_head_mx[obj_head_id][rel_id] = 1
                    obj_tail_mx[obj_tail_id][rel_id] = 1

            # 重新组装
            batch_text['text'].append(item['text'])
            batch_text['input_ids'].append(item['input_ids'])
            batch_text['offset_mapping'].append(item['offset_mapping'])
            batch_text['triple_list'].append(item['triple_list'])
            batch_mask.append(mask)
            batch_sub['heads_seq'].append(sub_heads_seq)
            batch_sub['tails_seq'].append(sub_tails_seq)
            batch_sub_rnd['head_seq'].append(sub_rnd_head_seq)
            batch_sub_rnd['tail_seq'].append(sub_rnd_tail_seq)
            batch_obj_rel['heads_mx'].append(obj_head_mx)
            batch_obj_rel['tails_mx'].append(obj_tail_mx)

        # 结构太复杂，没有转tensor
        return batch_mask, (batch_text, batch_sub_rnd), (batch_sub, batch_obj_rel)


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    print(iter(loader).__next__())
