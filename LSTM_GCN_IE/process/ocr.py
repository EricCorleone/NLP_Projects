"""
OCR识别
"""
import os
import cv2
import logging
import pandas as pd

from glob import glob
from tqdm import tqdm
from paddleocr import paddleocr, PaddleOCR

# 屏蔽调试错误
paddleocr.logging.disable(logging.DEBUG)


class OCR:
    def __init__(self):
        self.ocr = PaddleOCR()

    def scan(self, file_path, output_path, marked_path=None):
        # 文字识别
        # info = self.ocr.ocr(file_path, cls=False)
        # PaddleOCR包更新，ocr输出多了一层[]
        info = self.ocr.ocr(file_path, cls=False)[0]
        df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'text'])
        for i, item in enumerate(info):
            # 保留左上和右下坐标
            ((x1, y1), _, (x2, y2), _), (text, _) = item
            df.loc[i] = list(map(int, [x1, y1, x2, y2])) + [text]
        # 保存识别结果
        df.to_csv(output_path)
        # 判断是否需要保存标记文件
        if marked_path:
            self.marked(df, file_path, marked_path)

    # 导出带标记的图片
    def marked(self, df, file_path, marked_path):
        # 加载图片
        img = cv2.imread(file_path)
        for x1, y1, x2, y2, text in df.values:
            # 画矩形（坐标值必须为整数）
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)
        cv2.imwrite(marked_path, img)


if __name__ == '__main__':
    ocr = OCR()
    for file_path in tqdm(glob('../input/imgs/train/' + '*.*')):
        _, file_name = os.path.split(file_path)
        output_path = f'../output/train/csv/{file_name}.csv'
        marked_path = f'../output/train/imgs_marked/{file_name}'
        ocr.scan(file_path, output_path, marked_path)

    for file_path in tqdm(glob('../input/imgs/test/' + '*.*')):
        _, file_name = os.path.split(file_path)
        output_path = f'../output/test/csv/{file_name}.csv'
        marked_path = f'../output/test/imgs_marked/{file_name}'
        ocr.scan(file_path, output_path, marked_path)
