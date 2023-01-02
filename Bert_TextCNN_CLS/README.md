<h1>Bert-TextCNN-CLS 新闻文本分类</h1>
<h2>项目说明</h2>

本项目使用 <b>BERT + TextCNN</b> 实现新闻文本分类。

<h2>项目环境</h2>

PyTorch, Python

相关库安装`pip install -r requirement.txt`。

<h2>项目目录</h2>

```
Bert_TextCNN_CLS
        ├─ config.py          配置文件
        ├─ data               数据集
        ├─ figures            训练过程的可视化图像
        ├─ model.py           模型文件
        ├─ predict.py         预测文件
        ├─ process.py         预处理
        ├─ requirements.txt   需求文件
        ├─ test.py            测试文件
        ├─ train.py           训练文件
        └─ utils.py           工具函数
```

<h2>数据集</h2>

清华大学的 THUCNews 新闻文本分类数据集（子集），训练集18w，验证集1w，测试集1w。

一共10个类别：金融、房产、股票、教育、科学、社会、政治、体育、游戏、娱乐。

<h2>文本最大长度选取</h2>

运行`process.py`统计所有句子长度并根据直方图分布选取一个最大长度，填入`config.py`中的 TEXT_LEN 属性中。

<h2>训练和测试</h2>

运行`train.py`和`test.py`。

<h2>预测</h2>

运行`predict.py`。

<h2>训练自己的数据集</h2>

train.txt、dev.txt、test.txt 的数据格式：文本\t标签（数字表示）

```
华南理工大学2009年硕士研究生招生简章\t3   
传《半条命2：第三章》今年发售无望\t8   
```

class.txt：标签类别（文本）

<h3>修改内容</h3>

在配置文件`config.py`中修改句子长度、类别数和预训练模型名称。

```python
TEXT_LEN = 30
NUM_CLASSES = 10
BERT_MODEL = 'bert-base-chinese'
```
