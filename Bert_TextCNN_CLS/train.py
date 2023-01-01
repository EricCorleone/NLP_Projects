"""
训练
"""
import math
from config import *
from utils import *
from model import *


def train():
    train_set = Dataset()
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    dev_set = Dataset('dev')
    dev_loader = data.DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)

    if MODEL_PATHS:
        text_cnn = torch.load(MODEL_PATHS[-1], map_location=DEVICE)
        last_epoch_count = int(re.findall(r'\d+', MODEL_PATHS[-1])[-1])
        print(f'载入增量模型 epoch({last_epoch_count}) 成功。')
    else:
        last_epoch_count = 0
        text_cnn = TextCNN().to(DEVICE)

    optimizer = torch.optim.Adam(text_cnn.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    count = 0
    iters = []
    losses = []
    dev_accs = []
    total_batches = math.ceil(len(train_set) / BATCH_SIZE)

    for e in range(1, EPOCH + 1):
        for b, (input_ids, mask, target) in enumerate(train_loader):
            count += 1
            out = text_cnn(input_ids, mask)
            loss = loss_fn(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if b != 0 and (b + 1) % 50 != 0:
            #     continue

            iters.append(count)
            losses.append(loss.item())

            y_pred = torch.argmax(out, dim=1)
            report = evaluate(
                pred=y_pred.cpu().data.numpy(),
                true=target.cpu().data.numpy(),
                output_dict=True
            )
            train_acc = report['accuracy']

            with torch.no_grad():
                dev_input, dev_mask, dev_target = iter(dev_loader).__next__()
                dev_out = text_cnn(dev_input, dev_mask)
                dev_pred = torch.argmax(dev_out, dim=1)
                dev_report = evaluate(
                    pred=dev_pred.cpu().data.numpy(),
                    true=dev_target.cpu().data.numpy(),
                    output_dict=True
                )
                dev_acc = dev_report['accuracy']
                dev_accs.append(dev_acc)

            print(f'>> epoch: {e + last_epoch_count}/{EPOCH + last_epoch_count}, '
                  f'batch: {b + 1}/{total_batches}, '
                  f'loss: {round(loss.item(), 5)}, '
                  f'train_acc: {train_acc}, '
                  f'dev_acc: {dev_acc}')

        torch.save(text_cnn, os.path.join(MODEL_DIR, f'model_{last_epoch_count + e}.pth'))

    plot(iters, losses, dev_accs, last_epoch_count, 'Training')


if __name__ == '__main__':
    train()
