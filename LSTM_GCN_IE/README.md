<h1>LSTM-GCN 火车票识别</h1>

<h2>项目说明</h2>

本项目使用 <b>长短期记忆网络（LSTM） + 图卷积神经网络（GCN）</b> 实现火车票识别。该项目同样适用于身份证、护照、驾驶证、发票和购物小票等证件和票据的识别。

<h2>项目环境</h2>

Python, PyTorch, PaddlePaddle

相关库安装`pip install -r requirement.txt`。

<h2>项目目录</h2>

```
LSTM_GCN_IE
    ├─ config.py           配置文件
    ├─ figures             训练过程的可视化图像
    ├─ input               输入数据
    ├─ model.py            模型文件
    ├─ output              输出数据
    ├─ predict.py          预测文件
    ├─ process             数据预处理
    │    ├─ graph.py       创建图节点链接
    │    ├─ ocr.py         OCR识别
    │    └─ other.py       生成词表和标签表
    ├─ requirements.txt    需求文件
    ├─ test.py             测试文件
    ├─ train.py            训练文件
    └─ utils.py            工具函数
```

<h2>模型结构</h2>

采用 LSTM 提取节点文字的特征向量，并定义 GCN 模型，将 LSTM 提取的特征作为图节点的特征进行卷积。

<h2>数据集</h2>

通过爬虫在百度图片爬取的无水印火车票照片，分别存放在 input/imgs 目录下的 train、test 和 predict 文件夹中。

<h2>数据预处理</h2>

依次运行 process 目录下的`ocr.py`，`graph.py`和`other.py`。

1. 字符识别。采用 PaddleOCR 模块对火车票进行OCR识别，并对识别出来的所有信息进行手动标注，包括车票号、始发站、终点站、发车时间、座位号、车票等级、姓名等等。
2. 构建图结构。根据火车票票面文字的位置信息构建图结构，计算邻接矩阵；

3. 根据OCR识别出的文字生成词表和标签表；由于火车票识别后节点数量不尽相同，每个节点字数也不同，难以实现batch加载，因此定义一个函数实现单文件加载。

<h2>训练和测试</h2>

运行`train.py`和`test.py`。

学习率为1e-3，训练100个epoch，测试集F1得分为0.96，识别效果非常理想。

<h2>预测</h2>

运行`predict.py`。
