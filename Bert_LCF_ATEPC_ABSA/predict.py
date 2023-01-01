"""
预测
"""
from utils import *
from model import *
from transformers import BertTokenizer


def predict(model_name, text):
    model_path = os.path.join(MODEL_DIR, model_name)
    model = torch.load(model_path, map_location=DEVICE)

    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        tokens = list(text)
        input_ids = tokenizer.encode(tokens)
        mask = [True] * len(input_ids)

        # 实体部分
        input_ids = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)
        mask = torch.tensor(mask, device=DEVICE).unsqueeze(0)
        pred_ent_label = model.get_entity(input_ids, mask)

        # 情感分类
        b_ent_pos, b_ent_pola = get_pola(model, input_ids[0], mask[0], pred_ent_label[0])

        if not b_ent_pos:
            print('\t', 'no result.')
        else:
            pred_pair = []
            for ent_pos, pola in zip(b_ent_pos, torch.argmax(b_ent_pola, dim=1)):
                aspect = text[ent_pos[0] - 1:ent_pos[-1]]
                pred_pair.append({'aspect': aspect, 'sentiment': POLA_MAP[pola], 'position': ent_pos})

            print('\t', text)
            print('\t', pred_pair)


if __name__ == '__main__':
    predict('model_70.pth', text='手机运行速度很快，拍照效果挺不错，可是充电有点慢')
