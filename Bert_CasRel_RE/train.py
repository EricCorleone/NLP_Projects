"""
训练
"""
import math
import time

from utils import *
from model import *
from torch.utils import data


def load_model(model_paths):
    if model_paths:
        model = torch.load(model_paths[-1], map_location=DEVICE)
        last_epoch_count = int(re.findall(r'\d+', model_paths[-1])[-1])
        print(f'载入增量模型 epoch({last_epoch_count}) 成功。')
    else:
        last_epoch_count = 0
        model = CasRel().to(DEVICE)

    return model, last_epoch_count


def load_dataset(mode='train'):
    dataset = Dataset(mode)
    data_loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    return dataset, data_loader


def train():
    time_before_training = time.time()
    print('开始训练...')
    casrel, last_epoch_count = load_model(MODEL_PATHS)
    train_set, train_loader = load_dataset('train')
    _, dev_loader = load_dataset('dev')
    optimizer = torch.optim.AdamW(casrel.parameters(), lr=LR)

    count = 0
    loss_iters = []
    f1_iters = []
    losses = []
    f1_scores = []
    total_batches = math.ceil(len(train_set) / BATCH_SIZE)

    for e in range(1, EPOCH + 1 - last_epoch_count):
        time_before_epoch = time.time()
        for b, (batch_mask, batch_x, batch_y) in enumerate(train_loader):
            count += 1
            batch_text, batch_sub_rnd = batch_x
            batch_sub, batch_obj_rel = batch_y

            # 整理 input 数据并预测
            input_mask = torch.tensor(batch_mask, device=DEVICE)
            input = (
                torch.tensor(batch_text['input_ids'], device=DEVICE),
                torch.tensor(batch_sub_rnd['head_seq'], device=DEVICE),
                torch.tensor(batch_sub_rnd['tail_seq'], device=DEVICE)
            )
            encoded_text, pred_y = casrel(input, input_mask)

            # 整理target数据并计算损失
            true_y = (
                torch.tensor(batch_sub['heads_seq'], device=DEVICE),
                torch.tensor(batch_sub['tails_seq'], device=DEVICE),
                torch.tensor(batch_obj_rel['heads_mx'], device=DEVICE),
                torch.tensor(batch_obj_rel['tails_mx'], device=DEVICE)
            )
            loss = casrel.loss_fn(true_y, pred_y, input_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (b + 1) % 50 == 0:
                loss_iters.append(count)
                losses.append(loss.item())
                print(f'>> epoch: {e + last_epoch_count}/{EPOCH}, '
                      f'batch: {b + 1}/{total_batches}, '
                      f'loss: {loss.item()}')

            # 验证集评估
            if (b + 1) % 500 == 0:
                dev_mask, (dev_text, dev_sub_rnd), _ = iter(dev_loader).__next__()

                # 整理 input 数据并预测
                dev_input_mask = torch.tensor(dev_mask, device=DEVICE)
                dev_input = (
                    torch.tensor(dev_text['input_ids'], device=DEVICE),
                    torch.tensor(dev_sub_rnd['head_seq'], device=DEVICE),
                    torch.tensor(dev_sub_rnd['tail_seq'], device=DEVICE)
                )
                dev_encoded_text, dev_pred_y = casrel(dev_input, dev_input_mask)

                f1_score = report(casrel, dev_encoded_text, dev_pred_y, dev_text, dev_mask)
                f1_scores.append(f1_score)
                f1_iters.append(count)

        # if e % 2 == 0:
        model_save_path = os.path.join(MODEL_DIR, f'model_{last_epoch_count + e}.pth')
        torch.save(casrel, model_save_path)
        print(f'模型已保存到：{model_save_path}')

        print(f'---------此epoch耗时：{time.time() - time_before_epoch}s---------')

    plot(f1_iters=f1_iters,
         f1_scores=f1_scores,
         last_epoch_count=last_epoch_count)

    print(f'------------训练结束，总耗时：{time.time() - time_before_training}s------------')


if __name__ == '__main__':
    train()
