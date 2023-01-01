<h1>Bert-TextCNN-CLS 新闻文本分类</h1>
<h2>项目说明</h2>
本项目使用 <b>BERT + TextCNN</b> 实现新闻文本分类。
<h2>数据预处理</h2>
运行 process.py 统计所有句子长度并根据直方图分布选取一个最大长度，填在 config.py 中的 TEXT_LEN 里。<br>数据集位于 data/input 内。

<h2>训练和测试</h2>
运行 train.py 和 test.py。
<h2>预测</h2>
运行 predict.py。