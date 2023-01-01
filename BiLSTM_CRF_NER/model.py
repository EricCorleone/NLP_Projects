"""
模型
"""
from torch import nn
from config import *
from torchcrf import CRF


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=WORD_PAD_ID
        )
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(
            in_features=2 * HIDDEN_SIZE,
            out_features=TARGET_SIZE
        )
        self.crf = CRF(
            num_tags=TARGET_SIZE,
            batch_first=True
        )

    def _get_lstm_feature(self, input):
        out = self.embed(input)
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')


if __name__ == '__main__':
    mymodel = Model()
    input = torch.randint(0, 3000, size=(100, 50))
    print(mymodel)
    print(torch.tensor(mymodel(input, None)).shape)

