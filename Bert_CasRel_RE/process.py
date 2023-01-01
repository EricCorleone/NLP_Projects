"""
数据预处理
"""
import json
import pandas as pd
from config import *


def generate_rel():
    with open(SCHEMA_PATH, encoding='utf8') as f:
        rel_list = []
        for line in f.readlines():
            info = json.loads(line)
            rel_list.append(info['predicate'])
        rel_dict = {v: k for k, v in enumerate(rel_list)}
        df = pd.DataFrame(rel_dict.items())
        df.to_csv(REL_PATH, header=False, index=False)


if __name__ == '__main__':
    generate_rel()
