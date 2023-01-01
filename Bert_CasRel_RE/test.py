"""
测试
"""
import math
import time

from utils import *
from model import *
from tqdm import tqdm
from torch.utils import data


@timer('测试')
def test(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    casrel = torch.load(model_path, map_location=DEVICE)
    print(f'加载模型: {model_path} 成功。')

    test_set = Dataset('dev')

    with torch.no_grad():

        test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_set.collate_fn)

        correct_num, predict_num, gold_num = 0, 0, 0

        for b, (batch_mask, batch_x, batch_y) in enumerate(tqdm(test_loader, desc='正在评估模型', unit='batches')):
            batch_text, batch_sub_rnd = batch_x

            # 整理input数据并预测
            input_mask = torch.tensor(batch_mask, device=DEVICE)
            test_input = (
                torch.tensor(batch_text['input_ids'], device=DEVICE),
                torch.tensor(batch_sub_rnd['head_seq'], device=DEVICE),
                torch.tensor(batch_sub_rnd['tail_seq'], device=DEVICE)
            )
            encoded_text, pred_y = casrel(test_input, input_mask)

            # 计算关系三元组，和统计指标
            pred_sub_head, pred_sub_tail, _, _ = pred_y
            true_triple_list = batch_text['triple_list']

            # 遍历batch
            for i in range(len(pred_sub_head)):
                text = batch_text['text'][i]
                true_triple_item = true_triple_list[i]
                mask = batch_mask[i]
                offset_mapping = batch_text['offset_mapping'][i]

                sub_head_ids = torch.where(pred_sub_head[i] > SUB_HEAD_BAR)[0]
                sub_tail_ids = torch.where(pred_sub_tail[i] > SUB_TAIL_BAR)[0]

                pred_triple_item = get_triple_list(
                    casrel, sub_head_ids, sub_tail_ids, encoded_text[i], text, mask, offset_mapping)

                # 统计个数
                correct_num += len(set(true_triple_item) & set(pred_triple_item))
                predict_num += len(set(pred_triple_item))
                gold_num += len(set(true_triple_item))

        precision = correct_num / (predict_num + EPS)
        recall = correct_num / (gold_num + EPS)
        f1_score = 2 * precision * recall / (precision + recall + EPS)
        print(f'\tcorrect_num: {correct_num}, '
              f'predict_num: {predict_num}, '
              f'gold_num: {gold_num}')
        print(f'\tprecision: {precision:.4f}, '
              f'recall: {recall:.4f}, '
              f'f1_score: {f1_score:.4f}')


"""10 epochs
    正在评估模型: 100%|██████████| 2582/2582 [17:09<00:00,  2.51batches/s]
        correct_num: 28567, predict_num: 43617, gold_num: 37754
        precision: 0.6550, recall: 0.7567, f1_score: 0.7021
    测试耗时：1031.25s
    """

if __name__ == '__main__':
    test('model_10.pth')
