# BERT-CasRel 实体关系抽取

## 项目说明

本项目采用 **BERT + 层叠式指针标注框架（CasRel）** 实现实体关系抽取。模型来自 [weizhepei/CasRel](https://github.com/weizhepei/CasRel) 。

该模型属于 Joint Model，由实体识别和关系分类两部分组成，实体和关系共享一个 encoder，解码用两个 decoder，并只计算一次损失。

## 项目环境

Python, PyTorch, Transformers

相关库安装`pip install -r requirement.txt`。

## 项目目录

```
Bert_CasRel_RE
    ├─ config.py             配置文件
    ├─ data                  数据
    │    ├─ input            数据集存放位置
    │    └─ output           输出数据
    │           └─ models    保存的模型
    ├─ figures               训练过程的可视化图像
    ├─ model.py              模型文件
    ├─ predict.py            预测文件
    ├─ process.py            预处理文件
    ├─ requirements.txt      需求文件
    ├─ test.py               测试文件
    ├─ train.py              训练文件
    └─ utils.py              工具函数
```

## 模型结构

CasRel 模型分为两部分，一部分负责预测 subject 实体位置，另一部分负责预测 relation 和 object 的实体位置矩阵。

![model_structure](model_structure.png)

## 数据集

DuIE2.0 是业界规模最大的中文关系抽取数据集，其 schema 在传统简单关系类型基础上添加了多元复杂关系类型，此外其构建语料来自百度百科、百度信息流及百度贴吧文本，全面覆盖书面化表达及口语化表达语料，能充分考察真实业务场景下的关系抽取能力。

## 数据预处理

运行`process.py`，解析实体关系 json 文件并输出为 csv 文件。

## 自定义损失函数

subject 部分的损失函数分别为首/尾索引的二元交叉熵，relation 和 object 部分的损失函数为 48（relation 的总数）个首/尾索引的二元交叉熵；其中 subject 部分的损失函数需要乘以权重参数来解决目标值不均衡的问题。最后将四部分求和即为模型整体的损失。

## 训练和测试

运行`train.py`和`test.py`。

学习率为 1e-5，batch size 为 8，训练 10 个 epoch，测试时使用自定义的评估函数进行评估，最终F1得分为 0.702，三元组提取结果较好。

## 预测

运行`predict.py`。
