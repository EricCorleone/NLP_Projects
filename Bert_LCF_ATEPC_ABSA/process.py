"""
数据预处理
"""
import pandas as pd

from config import *


def format_sample(file_paths, output_path):
    text = bio = pola = ''
    items = []
    for file_path in file_paths:
        with open(file_path, encoding='utf8') as f:
            for line in f.readlines():
                # 单独的空行，表示句子间隔
                if line == '\n':
                    items.append({'text': text.strip(),
                                  'bio': bio.strip(),
                                  'pola': pola.strip()})
                    text = bio = pola = ''
                    continue
                # 文本、bio标记、情感极性
                t, b, p = line.split(' ')  # type: str
                text += t + ' '
                bio += b + ' '
                # # 情感极性修正
                p = str(1) if p.strip() == str(-1) else p.strip()
                p = str(-1) if p.strip() == str(0) else p.strip()
                p = str(0) if p.strip() == str(1) else p.strip()
                p = str(1) if p.strip() == str(2) else p.strip()
                pola += p + ' '
    df = pd.DataFrame(items)
    df.to_csv(output_path, index=False)


def check_label(file_path):
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        for b, p in zip(row['bio'].split(), row['pola'].split()):
            # 删除异常值
            if b in ['B-ASP', 'I-ASP'] and p == POLA_O_LABEL:
                print(index, row)
                df.drop(index=index, inplace=True)
                break

    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    format_sample([
        './input/origin/camera/camera.atepc.train.dat',
        './input/origin/car/car.atepc.train.dat',
        './input/origin/notebook/notebook.atepc.train.dat',
        './input/origin/phone/phone.atepc.train.dat',
    ], TRAIN_FILE_PATH)

    format_sample([
        './input/origin/camera/camera.atepc.test.dat',
        './input/origin/car/car.atepc.test.dat',
        './input/origin/notebook/notebook.atepc.test.dat',
        './input/origin/phone/phone.atepc.test.dat',
    ], TEST_FILE_PATH)

    check_label(TRAIN_FILE_PATH)
    check_label(TEST_FILE_PATH)
