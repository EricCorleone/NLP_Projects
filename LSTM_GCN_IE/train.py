"""
训练
"""
from glob import glob
from utils import *
from model import *


def train():
    if MODEL_PATHS:
        model = torch.load(MODEL_PATHS[-1], map_location=DEVICE)
        last_epoch_count = int(re.findall(r'\d+', MODEL_PATHS[-1])[0])
        print(f'载入增量模型epoch({last_epoch_count})成功。')
    else:
        last_epoch_count = 0
        model = Model().to(DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    iters = []
    iter = 0
    losses = []

    for e in range(1, EPOCH + 1):
        for b, csv_path in enumerate(glob(os.path.join(TRAIN_CSV_DIR, '*.csv'))):
            iter += 1

            # 加载数据
            inputs, targets = load_data(csv_path)

            # 加载邻接矩阵
            _, file_name = os.path.split(csv_path)
            graph_path = os.path.join(TRAIN_GRAPH_DIR, file_name[:-3] + 'pkl')
            adj, isolated_idx = load_file(graph_path)

            # 移除孤立点
            for i in isolated_idx:
                inputs.pop(i)
                targets.pop(i)

            # 模型训练
            y_pred = model(inputs, adj)
            y_true = torch.tensor(targets).to(DEVICE)
            loss = criterion(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 10 == 0:
                print(f'>> epoch: {last_epoch_count + e}/{last_epoch_count + EPOCH}, loss: {loss.item()}')
                iters.append(iter)
                losses.append(loss.item())

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if e % 20 == 0:
            torch.save(model, os.path.join(MODEL_DIR, f'model_{last_epoch_count + e}.pth'))

    plot(iters, losses, 'Training')


if __name__ == '__main__':
    train()
