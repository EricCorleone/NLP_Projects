<h1>Bert-LCF-ATEPC-ABSA 属性级情感分析</h1>
<h2>项目说明</h2>
本项目采用 <b>BERT + 局部上下文特征（LCF） + 属性提取和属性情感分类（ATEPC）</b> 来实现对电商平台用户评价的属性级情感分析（ABSA），模型有稍作修改。
<h2>数据预处理</h2>
运行 process.py 将样本处理成一句一行的形式，并剔除异常数据。

<h2>训练和测试</h2>
运行 train.py 和 test.py。
<h2>预测</h2>
运行 predict.py。