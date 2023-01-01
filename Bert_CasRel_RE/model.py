"""
模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from config import *

# 忽略 transformers 警告
from transformers import logging

logging.set_verbosity_error()


class CasRel(nn.Module):
    def __init__(self):
        super(CasRel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.sub_head_linear = nn.Linear(BERT_DIM, 1)
        self.sub_tail_linear = nn.Linear(BERT_DIM, 1)
        self.obj_head_linear = nn.Linear(BERT_DIM, REL_SIZE)
        self.obj_tail_linear = nn.Linear(BERT_DIM, REL_SIZE)

    def get_encoded_text(self, input_ids, mask):
        """
        获取 BERT Embedding

        Args:
            input_ids(list): 经过 BERT 编码的文本
            mask(list[bool]): BERT 掩码

        Returns:
            torch.Tensor: 转化成 BERT 词向量的文本
        """
        return self.bert(input_ids, attention_mask=mask)[0]

    def get_subs(self, encoded_text):
        """
        预测主体的首尾字索引

        Args:
            encoded_text(torch.Tensor): 一个批次的转化成 BERT 词向量的文本

        Returns:
            一个批次的主体的首/尾字索引预测结果（batch_size * 2 * 1个二分类）
        """
        pred_sub_head = torch.sigmoid(self.sub_head_linear(encoded_text))
        pred_sub_tail = torch.sigmoid(self.sub_tail_linear(encoded_text))
        return pred_sub_head, pred_sub_tail

    def get_objs_for_specific_sub(self, encoded_text, sub_head_seq, sub_tail_seq):
        """
        预测特定主体对应的客体首尾字索引

        Args:
            encoded_text(torch.Tensor): 一个批次的转化成 BERT 词向量的文本
            sub_head_seq(torch.Tensor): 一个批次的主体首字的独热编码序列
            sub_tail_seq(torch.Tensor): 一个批次的主体尾字的独热编码序列

        Returns:
            一个批次的客体的首/尾字索引预测结果（batch_size * 2 * 48个二分类）
        """
        # sub_head_seq.shape (batch_size, max_len) -> (batch_size, 1, max_len)
        sub_head_seq = sub_head_seq.unsqueeze(1).float()
        sub_tail_seq = sub_tail_seq.unsqueeze(1).float()

        # encoded_text.shape (batch_size, max_len, 768)
        sub_head = sub_head_seq @ encoded_text
        sub_tail = sub_tail_seq @ encoded_text
        encoded_text = encoded_text + (sub_head + sub_tail) / 2

        pred_obj_head = torch.sigmoid(self.obj_head_linear(encoded_text))
        pred_obj_tail = torch.sigmoid(self.obj_tail_linear(encoded_text))

        return pred_obj_head, pred_obj_tail

    def forward(self, input, mask):
        input_ids, sub_head_seq, sub_tail_seq = input
        encoded_text = self.get_encoded_text(input_ids, mask)

        # 预测subject首尾序列
        pred_sub_head, pred_sub_tail = self.get_subs(encoded_text)
        # (batch_size, max_len, 1)

        # 预测relation-object矩阵
        pred_obj_head, pred_obj_tail = self.get_objs_for_specific_sub(encoded_text, sub_head_seq, sub_tail_seq)
        # (batch_size, max_len, 48)

        return encoded_text, (pred_sub_head, pred_sub_tail, pred_obj_head, pred_obj_tail)

    def loss_fn(self, true_y, pred_y, mask):
        def calc_loss(pred, true, mask):
            true = true.float()
            # pred.shape (batch_size, max_len, 1) -> (batch_size, max_len)
            pred = pred.squeeze(-1)
            weight = torch.where(true > 0, CLS_WEIGHT_COEF[1], CLS_WEIGHT_COEF[0])
            loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')
            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            return torch.sum(loss * mask) / torch.sum(mask)

        pred_sub_head, pred_sub_tail, pred_obj_head, pred_obj_tail = pred_y
        true_sub_head, true_sub_tail, true_obj_head, true_obj_tail = true_y

        sub_head_loss = calc_loss(pred_sub_head, true_sub_head, mask) * SUB_WEIGHT_COEF
        sub_tail_loss = calc_loss(pred_sub_tail, true_sub_tail, mask) * SUB_WEIGHT_COEF
        obj_head_loss = calc_loss(pred_obj_head, true_obj_head, mask)
        obj_tail_loss = calc_loss(pred_obj_tail, true_obj_tail, mask)

        return sub_head_loss + sub_tail_loss + obj_head_loss + obj_tail_loss
