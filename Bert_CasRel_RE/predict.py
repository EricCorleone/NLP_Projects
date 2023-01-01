"""
预测
"""
import sys
import json
import random
from config import *

DEVICE = 'cpu'

from utils import *
from model import *
from transformers import BertTokenizerFast


@timer('预测')
def predict(model_name, random_index=True, index=None):
    """
    预测

    Args:
        model_name(str): 模型名称
        random_index(bool): 是否随机生成索引，默认为 True
        index(int): 手动指定索引，仅当 random_index 为 False 时有效
    """
    with open(TEST_JSON_PATH, encoding='utf8') as f:
        texts = f.readlines()

    if index and index > len(texts):
        raise IndexError(f'list index out of range, please specify the index between 1 and {len(texts)}')
    idx = random.randint(1, len(texts)) if random_index else index
    text = json.loads(texts[idx - 1])['text']

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    tokenized = tokenizer(text, return_offsets_mapping=True)
    info = {'input_ids': tokenized['input_ids'],
            'offset_mapping': tokenized['offset_mapping'],
            'mask': tokenized['attention_mask']}

    input_ids = torch.tensor([info['input_ids']], device=DEVICE)
    batch_mask = torch.tensor([info['mask']], device=DEVICE)

    model_path = os.path.join(MODEL_DIR, model_name)
    model = torch.load(model_path, map_location=DEVICE)

    encoded_text = model.get_encoded_text(input_ids, batch_mask)
    pred_sub_head, pred_sub_tail = model.get_subs(encoded_text)

    sub_head_ids = torch.where(pred_sub_head[0] > SUB_HEAD_BAR)[0]
    sub_tail_ids = torch.where(pred_sub_tail[0] > SUB_TAIL_BAR)[0]
    mask = batch_mask[0]
    encoded_text = encoded_text[0]

    offset_mapping = info['offset_mapping']

    pred_triple_item = get_triple_list(
        model, sub_head_ids, sub_tail_ids, encoded_text, text, mask, offset_mapping)

    print(f'No.{idx}\n{text}\n{pred_triple_item}')


if __name__ == '__main__':
    sys.stdout = Logger(PREDICT_RESULT_PATH, sys.stdout)
    # predict('model_10.pth', random_index=False, index=101239)
    predict('model_10.pth')
