"""
预测
"""
from config import *
from utils import *
from model import *


def predict(model_name, texts: list):
    id2label, _ = get_label()

    model = torch.load(os.path.join(MODEL_DIR, model_name), map_location=DEVICE)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    batch_input_ids = []
    batch_mask = []
    for text in texts:
        tokenized = tokenizer(text)
        input_ids = tokenized['input_ids']
        mask = tokenized['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        batch_input_ids.append(input_ids[:TEXT_LEN])
        batch_mask.append(mask[:TEXT_LEN])

    batch_input_ids = torch.tensor(batch_input_ids, device=DEVICE)
    batch_mask = torch.tensor(batch_mask, device=DEVICE)
    out = model(batch_input_ids, batch_mask)
    pred = torch.argmax(out, dim=1)

    print([id2label[l] for l in pred])


if __name__ == '__main__':
    predict('model_1.pth', texts=['小城不大，风景如画：边境小镇室韦的蝶变之路',
                                  '天问一号发射两周年，传回火卫一高清影像',
                                  '林志颖驾驶特斯拉自撞路墩起火，车头烧成废铁',
                                  '英国女王伊丽莎白二世去世',
                                  'GTA6正式发售',
                                  '阿根廷获得2022年卡塔尔世界杯冠军',
                                  '华南理工大学迎来百年校庆']
            )
    # 预测结果：['房产', '科技', '娱乐', '政治', '游戏', '体育', '教育']
