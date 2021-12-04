for i in `seq 2 10`
do
        python3 extral_CLS_fea.py ./source_bert_fea/fold$i/train.json ./CLS_fea/fold$i/train.txt;
        python3 extral_CLS_fea.py ./source_bert_fea/fold$i/valid.json ./CLS_fea/fold$i/valid.txt;
done
