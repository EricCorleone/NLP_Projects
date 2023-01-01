"""
训练
"""
import os

from utils import *
from model import *
from config import *


def train():
    dataset = Dataset()
    train_loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    if MODEL_PATHS:
        model = torch.load(MODEL_PATHS[-1], map_location=DEVICE)
        last_epoch_count = int(re.findall(r'\d+', MODEL_PATHS[-1])[0])
        print(f'载入增量模型epoch({last_epoch_count})成功。')
    else:
        last_epoch_count = 0
        model = Model().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    iters = []
    iter = 0
    losses = []
    total_steps = len(dataset) // BATCH_SIZE + 1

    for epoch in range(1, EPOCH + 1):
        for step, (input, target, mask) in enumerate(train_loader):
            iter += 1
            input, target, mask = input.to(DEVICE), target.to(DEVICE), mask.to(DEVICE)
            loss = model.loss_fn(input, target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                iters.append(iter)
                losses.append(loss.item())
                print('>> epoch:', last_epoch_count + epoch, 'step:', f'{step + 1}/{total_steps}', 'loss:', loss.item())

        model_path = os.path.join(MODEL_DIR, f'model_{last_epoch_count + epoch}.pth')
        torch.save(model, model_path)

    plot(iters, losses, 'Training')


if __name__ == '__main__':
    train()
