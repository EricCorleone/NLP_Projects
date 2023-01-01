"""
测试
"""
import math
import time

from utils import *
from model import Model
from tqdm import tqdm

import torch.utils.data as data


def test(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    model = torch.load(model_path)
    print(f'载入模型 {model_path} 成功。')

    test_set = Dataset('test')
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test_set.collate_fn)

    total_batches = math.ceil(len(test_set) / BATCH_SIZE)

    with torch.no_grad():
        correct_cnt = pred_cnt = gold_cnt = 0

        for b, batch in enumerate(tqdm(test_loader, desc='正在评估模型', unit='batches')):
            input_ids, mask, ent_cdm, ent_cdw, ent_label, pola_label, pairs = batch

            # 实体部分
            pred_ent_label = model.get_entity(input_ids, mask)

            # 计算准确率（实体和情感都判断正确才算对）
            for i in range(len(input_ids)):
                # 累加真实值数量
                gold_cnt += len(pairs[i])

                # 根据预测的实体label，解析出实体位置，并预测情感分类
                b_ent_pos, b_ent_pola = get_pola(model, input_ids[i], mask[i], pred_ent_label[i])
                if not b_ent_pos:
                    continue

                # 解析实体和情感，并和真实值对比
                pred_pair = []
                cnt = 0
                for ent, pola in zip(b_ent_pos, torch.argmax(b_ent_pola, dim=1)):
                    pair_item = (ent, pola.item())
                    pred_pair.append(pair_item)
                    # 判断正确，正确数量加1
                    if pair_item in pairs[i]:
                        cnt += 1

                # 累加数值
                correct_cnt += cnt
                pred_cnt += len(pred_pair)

        # 指标计算
        precision = round(correct_cnt / (pred_cnt + EPS), 3)
        recall = round(correct_cnt / (gold_cnt + EPS), 3)
        f1_score = round(2 / (1 / (precision + EPS) + 1 / (recall + EPS)), 3)
        print(f'\tcorrect_count: {correct_cnt}, '
              f'pred_count: {pred_cnt}, '
              f'gold_count: {gold_cnt}')

        print(f'\tprecision: {precision}, '
              f'recall: {recall}, '
              f'f1_score: {f1_score}\n')


if __name__ == '__main__':
    test(model_name='model_100.pth')

    """bert frozen, lr=1e-4, bs=50, epoch=100
    
    correct_count: 1890, pred_count: 2647, gold_count: 2552
    precision: 0.714, recall: 0.741, f1_score: 0.727
    """

    """bert unfrozen, lr=1e-05, bs=50, epoch=70
    
    correct_count: 2056, pred_count: 2577, gold_count: 2552
    precision: 0.798, recall: 0.806, f1_score: 0.802
    """
