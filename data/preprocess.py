import json
import numpy as np
from sklearn.model_selection import train_test_split

all_data_path = "./ccks_task2_train.txt"
eval_data_path = "./ccks_task2_eval_data.txt"
reason_type_path = "./reason_type.json"
result_type_path = "./result_type.json"
reason_result_schema = "reason_result_schema.json"

train_data_path = "./train.json"
dev_data_path = "./dev.json"

def read_by_line(path):
    """read data by line with json"""
    with open(path,encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def write_by_line(obj, path):
    """write data by line with json"""
    with open(path, 'w',encoding="utf-8") as w:
        for line in obj:
            w.write(json.dumps(line, ensure_ascii=False) + '\n')

#从数据中整理出40个原因事件和19个结果事件的组合
#写入文件
def get_event_schem(data_path,schema=reason_result_schema):

    data = read_by_line(data_path)
    events = []
    lens = []
    for line in data:
        text_len = len(line["text"])
        lens.append(text_len)
        for r in line["result"]:
            event = {}

            event["reason_type"] = r["reason_type"]
            event["result_type"] = r["result_type"]
            if event not in events:
                events.append(event)
    print(max(lens),np.mean(lens))
    write_by_line(events,schema)

def split_test_train(data_path,train_path,dev_path):
    "按照1:9划分验证集和训练集"
    data = read_by_line(data_path)
    train,dev = train_test_split(data,test_size=0.1,random_state=0)
    write_by_line(train,train_path)
    write_by_line(dev,dev_path)


get_event_schem(all_data_path)
split_test_train(all_data_path,train_data_path,dev_data_path)

# data = read_by_line(train_data_path)
# reason = {}
# for line in data:
#
#     result = line["result"]
#     for res in result:
#         if res["reason_type"] not in reason:
#             reason[res["reason_type"]]=0
#             #print(1)
# print(2)































