"""
模型
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from config import *
from transformers import BertModel


class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)

        # 固定BERT参数，不参与训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (i, EMBEDDING_DIM)) for i in FILTER_SIZES])
        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)

    def conv_and_pool(self, conv, input):
        # [batch_size, 1, 30, 768] -conv1-> [batch_size, 256, 29/28/27, 1]
        out = conv(input)
        out = F.relu(out)
        # [batch_size, 256, 29, 1] -max_pool-> [batch_size, 256, 1, 1] -squeeze-> [batch_size, 256]
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def forward(self, input, mask):
        # [batch_size, 30, 768] -unsqueeze(1)-> [batch_size, 1, 30, 768]
        out = self.bert(input, mask)[0].unsqueeze(1)
        out = torch.cat([self.conv_and_pool(conv, out) for conv in self.convs], dim=1)
        return self.linear(out)


if __name__ == '__main__':
    model = TextCNN()
    input = torch.randint(0, 3000, (2, TEXT_LEN))
    mask = torch.ones_like(input)
    print(input.shape)
    print(model(input, mask).shape)
