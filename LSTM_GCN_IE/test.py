"""
测试
"""
from glob import glob
from utils import *
from model import *


def test(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    model = torch.load(model_path, map_location=DEVICE)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    with torch.no_grad():
        y_true_list = []
        y_pred_list = []
        id2label, _ = get_label()

        for b, csv_path in enumerate(glob(TEST_CSV_DIR + '*.csv')):

            inputs, targets = load_data(csv_path)

            _, file_name = os.path.split(csv_path)
            adj_path = TEST_GRAPH_DIR + file_name[:-3] + 'pkl'
            adj, loss_idx = load_file(adj_path)

            for i in loss_idx:
                inputs.pop(i)
                targets.pop(i)

            y_pred = model(inputs, adj)
            y_true = torch.tensor(targets).to(DEVICE)
            loss = criterion(y_pred, y_true)

            print('>> batch:', b, 'loss:', loss.item())

            y_pred_list += y_pred.argmax(dim=1).tolist()
            y_true_list += y_true.tolist()

        print(report(y_true_list, y_pred_list, id2label))


if __name__ == '__main__':
    test('model_100.pth')

    """
    100 epochs
                             precision    recall  f1-score   support
    
             ticket_num       1.00      0.90      0.95        10
       starting_station       0.89      0.80      0.84        10
    destination_station       0.82      0.90      0.86        10
              train_num       0.90      0.90      0.90        10
                  other       0.96      0.99      0.97        75
                   date       1.00      0.90      0.95        10
            seat_number       1.00      1.00      1.00        10
           ticket_grade       1.00      1.00      1.00        10
           ticket_price       1.00      1.00      1.00        10
                   name       1.00      1.00      1.00        10
    
               accuracy                           0.96       165
              macro avg       0.96      0.94      0.95       165
           weighted avg       0.96      0.96      0.96       165
    """
