"""
测试
"""
import os.path

from utils import *
from model import *
from config import *


def test(model_name):
    dataset = Dataset('test')
    test_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    total_steps = len(dataset) // BATCH_SIZE + 1

    with torch.no_grad():
        model_path = os.path.join(MODEL_DIR, model_name)
        model = torch.load(model_path)
        y_true_list = []
        y_pred_list = []
        id2label, _ = get_label()

        for step, (input, target, mask) in enumerate(test_loader):
            input, target, mask = input.to(DEVICE), target.to(DEVICE), mask.to(DEVICE)
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)

            if step % 10 == 0:
                print('>> step:', f'{step}/{total_steps}', 'loss:', loss.item())

            for lst in y_pred:
                y_pred_list.append([id2label[i] for i in lst])
            for y, m in zip(target, mask):
                y_true_list.append([id2label[i] for i in y[m == True].tolist()])

        print(report(y_true_list, y_pred_list))


if __name__ == '__main__':
    test('model_40.pth')
    """
                  precision    recall  f1-score   support
    
          Amount       0.59      0.58      0.58       255
         Anatomy       0.68      0.58      0.63      8670
         Disease       0.72      0.79      0.76     10129
            Drug       0.67      0.68      0.68      3710
        Duration       0.61      0.48      0.54       223
       Frequency       0.57      0.59      0.58        78
           Level       0.51      0.38      0.44       424
          Method       0.50      0.34      0.40       187
       Operation       0.63      0.58      0.60       115
          Reason       0.44      0.30      0.36      1323
         SideEff       0.37      0.43      0.40       190
         Symptom       0.59      0.32      0.42      1253
            Test       0.72      0.72      0.72     13989
      Test_Value       0.59      0.52      0.55      2411
       Treatment       0.47      0.34      0.40       203
    
       micro avg       0.69      0.66      0.67     43160
       macro avg       0.58      0.51      0.54     43160
    weighted avg       0.68      0.66      0.67     43160
    """
