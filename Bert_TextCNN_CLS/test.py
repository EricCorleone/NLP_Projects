"""
测试
"""
import math
from tqdm import tqdm
from config import *
from utils import *
from model import *


def test():
    id2label, _ = get_label()

    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    text_cnn = torch.load(os.path.join(MODEL_DIR, 'model_1.pth'), map_location=DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    y_pred = []
    y_true = []
    total_batches = math.ceil(len(test_dataset) / BATCH_SIZE)

    with torch.no_grad():
        for b, (input, mask, target) in enumerate(tqdm(test_loader, desc='正在评估模型', unit=' batches')):
            out = text_cnn(input, mask)
            # loss = loss_fn(out, target)

            # print(f'>> batch: {b + 1}/{total_batches}, loss: {round(loss.item(), 5)}')

            test_pred = torch.argmax(out, dim=1)

            y_pred += test_pred.data.tolist()
            y_true += target.data.tolist()

    print(evaluate(y_pred, y_true, id2label))

    """lr=1e-3, bs=100, epoch=1
                  precision    recall  f1-score   support

              金融       0.93      0.91      0.92      1000
              房产       0.94      0.94      0.94      1000
              股票       0.83      0.92      0.87      1000
              教育       0.95      0.96      0.96      1000
              科技       0.88      0.88      0.88      1000
              社会       0.96      0.88      0.92      1000
              政治       0.93      0.89      0.91      1000
              体育       0.96      0.99      0.97      1000
              游戏       0.95      0.94      0.94      1000
              娱乐       0.92      0.95      0.93      1000
    
        accuracy                           0.93     10000
       macro avg       0.93      0.93      0.93     10000
    weighted avg       0.93      0.93      0.93     10000
    """


if __name__ == '__main__':
    test()
