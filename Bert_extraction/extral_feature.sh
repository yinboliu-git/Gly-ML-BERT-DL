BERT_BASE_DIR=your_bert_file
python3 ./bert_bash/extract_features.py \
    --input_file=../data/dataset/fold10/valid.txt \
    --output_file=../data/dataset/valid.json \
    --layers=-1 \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/model.ckpt \
    --max_seq_length=64 \
    --batch_size=32
