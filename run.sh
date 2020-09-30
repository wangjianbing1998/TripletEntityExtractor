echo '准备关系分类数据';
python bin/predicate_classifiction/predicate_data_manager.py;

echo 关系分类模型训练;
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
        --output_dir=./output/predicate_classification_model/epochs6;


ehco 准备序列标注数据;
python bin/subject_object_labeling/sequence_labeling_data_manager.py;

ehco 列标注模型训练;
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
--output_dir=./output/sequnce_labeling_model/epochs9/;

ehco 关系分类模型预测;
python run_predicate_classification.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/predicate_classifiction/classification_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/predicate_classification_model/epochs6/model.ckpt-212 \
  --max_seq_length=128 \
  --output_dir=./output/predicate_infer_out/epochs6/ckpt212;

ehco 把关系分类模型预测结果转换成序列标注模型的预测输入;
python bin/predicate_classifiction/prepare_data_for_labeling_infer.py;

ehco 序列标注模型预测;
python run_sequnce_labeling.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/subject_object_labeling/sequence_labeling_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/epochs9/model.ckpt-296 \
  --max_seq_length=128 \
  --output_dir=./output/sequnce_infer_out/epochs9/ckpt296;


ehco 生成实体-关系结果;
python produce_submit_json_file.py