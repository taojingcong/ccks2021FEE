# CCKS2021

## 介绍
CCKS 2021：面向金融领域的篇章级事件抽取和事件因果关系抽取
url:https://www.biendata.xyz/competition/ccks_2021_task6_2
最终成绩第十

## 主要框架
代码有两部分组成,event classification和arguement detection

#### event classification
- 将原始数据中的reason_type和result_type事件提取出来
- 使用BERT+linear
- 输入[cls]+key_word+[sep]+sentence
- 输出为原因事件的组合reason_type#result_type的概率

#### arguement classification
- 通过第一步的结果提取句子的事件角色
- 使用BERT+linear
- 输入[cls]+role+[sep]+event_type+[sep]+key_word+[sep]+sentence
- 输出为角色在句子中的起始位置概率和结束位置概率
- 加入了负例,负例中的原因或结果其中一个和正例相同

## 运行
- 将data.rar解压
- 在ccks2021FEE下放入roberta模型
- 运行event/event_train.py进行第一部分训练
- 运行arguemnt/argue_train.py进行第二部分训练

## 效果
event_train的f1值在0.64左右
argue_train的f1值在0.82左右

## 文件说明
* event: event classification相关代码

  - event_train:  训练代码
  - model:    模型
  - event_dataReader: 数据处理代码
  - statistics:   对多组模型结果投票

* arguement: arguement detection相关代码
  - event_train:  训练代码
  - model:    模型
  - argu_data_reader: 数据处理代码
  - statistics:   对多组模型结果投票

- evaluate: validation验证代码