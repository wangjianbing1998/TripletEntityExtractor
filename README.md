# Entity-Relation-Extraction


### 任务
给定schema约束集合及句子sent，其中schema定义了关系P以及其对应的主体S和客体O的类别，例如（S_TYPE:人物，P:妻子，O_TYPE:人物）、（S_TYPE:公司，P:创始人，O_TYPE:人物）等。 任务要求参评系统自动地对句子进行分析，输出句子中所有满足schema约束的SPO三元组知识Triples=[(S1, P1, O1), (S2, P2, O2)…]。
输入/输出:
(1) 输入:schema约束集合及句子sent
(2) 输出:句子sent中包含的符合给定schema约束的三元组知识Triples

**例子**
输入句子： ```"text": "《古世》是连载于云中书城的网络小说，作者是未弱"```

输出三元组： ```"spo_list": [{"predicate": "作者", "object_type": "人物", "subject_type": "图书作品", "object": "未弱", "subject": "古世"}, {"predicate": "连载网站", "object_type": "网站", "subject_type": "网络小说", "object": "云中书城", "subject": "古世"}]}```

### 数据简介
本次竞赛使用的SKE数据集是业界规模最大的基于schema的中文信息抽取数据集，其包含超过43万三元组数据、21万中文句子及50个已定义好的schema，表1中展示了SKE数据集中包含的50个schema及对应的例子。数据集中的句子来自百度百科和百度信息流文本。数据集划分为17万训练集，2万验证集和2万测试集。其中训练集和验证集用于训练，可供自由下载，测试集分为两个，测试集1供参赛者在平台上自主验证，测试集2在比赛结束前一周发布，不能在平台上自主验证，并将作为最终的评测排名。

## Getting Started
### Environment Requirements
+ python 3.6+
+ Tensorflow 1.12.0+

### Step 1: Environmental preparation
+ Install Tensorflow 
+ Dowload [bert-base, chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip), unzip file and put it in ```pretrained_model``` floader.

### Step 2: Download the training data, dev data and schema files
Please download the training data, development data and schema files from [the competition website](http://lic2019.ccf.org.cn/kg), then unzip files and put them in ```./raw_raw_data/``` folder.
```
cd data
unzip train_data.json.zip 
unzip dev_data.json.zip
...
```

Official Data Download Address [baidu](http://ai.baidu.com/broad/download)

There is no longer a raw data download, if you have any questions, you can contact my mailbox wangzichaochaochao@gmail.com

**关系分类模型和实体序列标注模型可以同时训练，但是只能依次预测！**
## 准备自己的数据集
准备数据进raw_data/文件夹下，格式如样例所示即可
- `train_data.json`: 训练数据

- `test1_data_postag.json`:测试数据

- `dev_data.json`:验证数据

本样例为了数据的方便处理，将测试数据与验证数据都使用与训练数据一样的数据集，当然您也可以使用其他数据集

## 训练阶段

您可以一步运行完所有的代码：
```
sh run.sh
```
也可以使用下面的步骤一步一步的运行
### 准备关系分类数据
```
python bin/predicate_classifiction/predicate_data_manager.py
```

### 关系分类模型训练
```
python run_predicate_classification.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=bin/predicate_classifiction/classification_data \
--vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=6.0 \
--output_dir=./output/predicate_classification_model/epochs6/
```

### 准备序列标注数据
```
python bin/subject_object_labeling/sequence_labeling_data_manager.py
```

### 序列标注模型训练
```
python run_sequnce_labeling.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=bin/subject_object_labeling/sequence_labeling_data \
--vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=9.0 \
--output_dir=./output/sequnce_labeling_model/epochs9/
```

## 预测阶段

### 关系分类模型预测
```
python run_predicate_classification.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/predicate_classifiction/classification_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/predicate_classification_model/epochs6/model.ckpt-212 \
  --max_seq_length=128 \
  --output_dir=./output/predicate_infer_out/epochs6/ckpt212
```

### 把关系分类模型预测结果转换成序列标注模型的预测输入
```
python bin/predicate_classifiction/prepare_data_for_labeling_infer.py
```

### 序列标注模型预测
```
python run_sequnce_labeling.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/subject_object_labeling/sequence_labeling_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/epochs9/model.ckpt-296 \
  --max_seq_length=128 \
  --output_dir=./output/sequnce_infer_out/epochs9/ckpt296
```

### 生成实体-关系结果
```
python produce_submit_json_file.py
```

### 转换数据结果
```
python json2excel.py

```
## Reference
- [Reference](https://github.com/yuanxiaosc/Entity-Relation-Extraction)
