"""
预测
"""
from utils import *
from model import *
from config import *


def predict(model_name, text):
    _, word2id = get_vocab()
    input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in text]])
    mask = torch.tensor([[1] * len(text)]).bool()

    model_path = os.path.join(MODEL_DIR, model_name)
    model = torch.load(model_path).cpu()
    y_pred = model(input, mask)
    id2label, _ = get_label()
    label = [id2label[l] for l in y_pred[0]]
    # print(text)
    # print(label)
    print(extract(label, text))


if __name__ == '__main__':
    text = '每个糖尿病患者,无论是病情轻重,不论是注射胰岛素,还是口服降糖药,都必须合理地控制饮食。'
    predict('model_40.pth', text)
