"""
模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

from torchcrf import CRF
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertPooler

from transformers import logging

logging.set_verbosity_error()

config = BertConfig.from_pretrained(BERT_MODEL_NAME)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        # 无需冻结BERT参数，否则BERT的优势发挥不出来
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False
        self.ent_linear = nn.Linear(BERT_DIM, ENT_SIZE)
        self.crf = CRF(ENT_SIZE, batch_first=True)

        self.pola_linear2 = nn.Linear(BERT_DIM * 2, BERT_DIM)
        self.pola_linear3 = nn.Linear(BERT_DIM * 3, BERT_DIM)
        self.pola_linear = nn.Linear(BERT_DIM, POLA_DIM)
        self.attention = BertAttention(config)
        self.pooler = BertPooler(config)
        # self.dropout = nn.Dropout(p=0.2)

    def get_text_encoded(self, input_ids, mask):
        return self.bert(input_ids, attention_mask=mask)[0]

    def get_entity_fc(self, text_encoded):
        # return self.dropout(self.ent_linear(text_encoded))
        return self.ent_linear(text_encoded)

    def get_entity_crf(self, entity_fc, mask):
        return self.crf.decode(entity_fc, mask)

    def get_entity(self, input_ids, mask):
        """
        获取实体标签
        """
        text_encoded = self.get_text_encoded(input_ids, mask)
        entity_fc = self.get_entity_fc(text_encoded)
        pred_ent_label = self.get_entity_crf(entity_fc, mask)
        return pred_ent_label

    def get_pola(self, input_ids, mask, ent_cdm, ent_cdw) -> torch.Tensor:
        """
        获取情绪正负分类（线性层输出，未经过激活函数）
        """
        text_encoded = self.get_text_encoded(input_ids, mask)

        # shape [b, c] -> [b, c, 768]
        ent_cdm_weight = ent_cdm.unsqueeze(-1).repeat(1, 1, BERT_DIM)
        ent_cdw_weight = ent_cdw.unsqueeze(-1).repeat(1, 1, BERT_DIM)
        cdm_features = ent_cdm_weight * text_encoded
        # cdm_features = torch.mul(text_encoded, ent_cdm_weight)
        cdw_features = ent_cdw_weight * text_encoded
        # cdw_features = torch.mul(text_encoded, ent_cdw_weight)

        # 根据配置，使用不同策略，重新组合特征，再降到768维
        if LCF == 'fusion':
            out = torch.cat([text_encoded, cdm_features, cdw_features], dim=-1)
            out = self.pola_linear3(out)
        elif LCF == 'cdw':
            out = cdw_features
        elif LCF == 'cdm':
            out = cdm_features
        else:
            raise ValueError(f'{LCF} is not a valid LCF mode, please select one of ["cdw", "cdm", "fusion"]')

        # self-attention 结合上下文信息，增强语义
        out = self.attention(out, None)
        # pooler 取[CLS]标记位，作为整个句子的特征
        out = self.pooler(torch.tanh(out[0]))
        # return torch.sigmoid(self.pola_linear(out))
        return self.pola_linear(out)

    def ent_loss_fn(self, input_ids, ent_label, mask):
        text_encoded = self.get_text_encoded(input_ids, mask)
        entity_fc = self.get_entity_fc(text_encoded)
        return -self.crf.forward(entity_fc, ent_label, mask, reduction='mean')

    def pola_loss_fn(self, pred_pola, pola_label):
        return F.cross_entropy(pred_pola, pola_label)

    def loss_fn(self, input_ids, ent_label, mask, pred_pola, pola_label):
        return self.ent_loss_fn(input_ids, ent_label, mask) + \
               self.pola_loss_fn(pred_pola, pola_label)


if __name__ == '__main__':
    input_ids = torch.randint(0, 3000, (2, 30))
    mask = torch.ones((2, 30)).bool()
    ent_cdm = torch.rand((2, 30))
    ent_cdw = torch.rand((2, 30))
    model = Model()
    # print(model.get_entity(input_ids, mask))
    print(model.get_pola(input_ids, mask, ent_cdm, ent_cdw))
