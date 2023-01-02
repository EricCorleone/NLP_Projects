<h1>BiLSTM-CRF 命名实体识别</h1>
<h2>项目说明</h2>
《瑞金医院MMC人工智能辅助构建知识图谱大赛》第一赛季，要求在糖尿病相关的学术论文和临床指南的基础上，做实体的标注。本项目采用 <b>双向长短期记忆网络（BiLSTM） + 条件随机场（CRF）</b> 来实现。
<h2>数据预处理</h2>
运行 data_process.py。<br>
数据集位于 input/origin 内。
<h2>训练和测试</h2>
运行 train.py 和 test.py。
