"""
训练
"""
import math

from utils import *
from model import Model

import torch.utils.data as data


def train():
    if MODEL_PATHS:
        model = torch.load(MODEL_PATHS[-1], map_location=DEVICE)
        last_epoch_count = int(re.findall(r'\d+', MODEL_PATHS[-1])[-1])
        print(f'载入增量模型 epoch({last_epoch_count}) 成功。')
    else:
        last_epoch_count = 0
        model = Model().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_set = Dataset()
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_set.collate_fn)

    count = 0
    iters = []
    losses = []
    f1_scores = []
    total_batches = math.ceil(len(train_set) / BATCH_SIZE)
    # print(len(train_set))  # 5121
    # exit()

    for e in range(1, EPOCH + 1):
        for b, batch in enumerate(train_loader):
            count += 1
            input_ids, mask, ent_cdm, ent_cdw, ent_label, pola_label, pairs = batch

            # # 实体部分
            # pred_ent_label = model.get_entity(input_ids, mask)

            # 极性部分
            pred_pola = model.get_pola(input_ids, mask, ent_cdm, ent_cdw)

            # 损失计算
            loss = model.loss_fn(input_ids, ent_label, mask, pred_pola, pola_label)  # type: torch.Tensor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (b + 1) % 10 != 0:
                continue

            iters.append(count)
            losses.append(loss.item())
            print(f'>> epoch: {last_epoch_count + e}/{last_epoch_count + EPOCH}, '
                  f'batch: {b + 1}/{total_batches}, '
                  f'loss: {round(loss.item(), 3)}'
                  )

            if (b + 1) % 100 != 0:
                continue

            # 实体部分
            pred_ent_label = model.get_entity(input_ids, mask)

            # 计算准确率（实体和情感都判断正确才算对）
            correct_cnt = pred_cnt = gold_cnt = 0
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
            f1_scores.extend([f1_score * SCALE_RATE] * 10)
            print(f'\tcorrect_count: {correct_cnt}, '
                  f'pred_count: {pred_cnt}, '
                  f'gold_count: {gold_cnt}')

            print(f'\tprecision: {precision}, '
                  f'recall: {recall}, '
                  f'f1_score: {f1_score}\n')

        if e % 5 == 0:
            torch.save(model, os.path.join(MODEL_DIR, f'model_{last_epoch_count + e}.pth'))

    plot(iters, losses, f1_scores, last_epoch_count)


if __name__ == '__main__':
    train()
