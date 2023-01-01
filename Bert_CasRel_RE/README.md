<h1>Bert-CasRel-RE 实体关系抽取</h1>
<h2>项目说明</h2>
本项目采用 <b>BERT + 层叠式指针标注框架（CasRel）</b> 实现实体关系抽取。

该模型属于Joint Model，由实体识别和关系分类两部分组成，实体和关系共享一个encoder，解码用两个decoder，并只计算一次损失。

<h2>数据预处理</h2>
运行 process.py，解析实体关系 json 文件并输出为 csv 文件。

<h2>训练和测试</h2>
运行 train.py 和 test.py。
<h2>预测</h2>
运行 predict.py。