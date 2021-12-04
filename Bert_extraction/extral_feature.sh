# /mnt/raid5/data3/xlzhu/learn_DL/BERT_pretrain/model_pt/multi_cased_L-12_H-768_A-12
# /mnt/raid5/data3/xlzhu/learn_DL/BERT_pretrain/model_pt/probert
export BERT_BASE_DIR=/mnt/raid5/data3/xlzhu/learn_DL/BERT_pretrain/model_pt/probert
#export BERT_BASE_DIR=/mnt/raid5/data3/xlzhu/learn_DL/BERT_pretrain/model_pt/multi_cased_L-12_H-768_A-12

python3 /mnt/raid5/data3/xlzhu/learn_DL/BERT_pretrain/bert/extract_features.py \
    --input_file=/mnt/raid5/data4/publice/gly_site_pred/test_data/fold10/valid.txt \
    --output_file=/mnt/raid5/data4/publice/gly_site_pred/probert_fea/source_bert_data/fold10/valid.json \
    --layers=-1 \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/model.ckpt \
    --max_seq_length=64 \
    --batch_size=32
