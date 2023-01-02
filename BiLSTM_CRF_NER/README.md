<h1>BiLSTM-CRF 命名实体识别</h1>
<h2>项目说明</h2>

《瑞金医院MMC人工智能辅助构建知识图谱大赛》第一赛季，要求在糖尿病相关的学术论文和临床指南的基础上，做实体的标注。本项目采用 <b>双向长短期记忆网络（BiLSTM） + 条件随机场（CRF）</b> 来实现。

<h2>项目环境</h2>

PyTorch, Python

相关库安装`pip install -r requirement.txt`。

<h2>数据集</h2>

[中文糖尿病科研文献实体关系数据集DiaKG_数据集-阿里云天池 (aliyun.com)](https://tianchi.aliyun.com/dataset/88836)

本数据集来源于41篇中文糖尿病领域专家共识，数据包括基础研究、临床研究、药物使用、临床病例、诊治方法等多个方面，时间跨度达到7年，涵盖了近年来糖尿病领域最广泛的研究内容和热点。数据集的标注者都具有医学背景，共标注了22,050个医学实体和6,890对实体关系。

<h2>项目目录</h2>

```
BiLSTM_CRF_NER
    ├─ config.py           配置文件
    ├─ data_process.py     数据预处理
    ├─ figures             训练过程的可视化图像
    ├─ input               输入数据
    ├─ model.py            模型文件
    ├─ output              输出数据
    ├─ predict.py          预测文件
    ├─ requirements.txt    需求文件
    ├─ test.py             测试文件
    ├─ train.py            训练文件
    └─ utils.py            工具函数
```

<h2>数据预处理</h2>

运行`data_process.py`。

<h2>训练和测试</h2>

运行`train.py`和`test.py`。

学习率为1e-3，batch size为64，训练40个epoch，测试集平均F1得分为0.67，还有优化空间，最理想值为0.76。
