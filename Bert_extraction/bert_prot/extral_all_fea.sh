for i in `seq 1 10`
do
        #python3 extral_all_fea.py ./source_bert_fea/fold$i/train.json ./all_fea/fold$i/train.txt;
        python3 extral_all_fea.py ./source_bert_fea/fold$i/test.json ./all_fea/fold$i/test.np;
        python3 extral_all_fea.py ./source_bert_fea/fold$i/valid.json ./all_fea/fold$i/valid.np;
done
