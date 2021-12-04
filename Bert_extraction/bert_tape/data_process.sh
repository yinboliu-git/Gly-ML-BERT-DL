for i in `seq 2 10`
do
       python3 data_process.py fold$i/train.npz fold$i/CLS_train.txt fold$i/aa_train.np;
       python3 data_process.py fold$i/test.npz fold$i/CLS_test.txt fold$i/aa_test.np;
       python3 data_process.py fold$i/valid.npz fold$i/CLS_valid.txt fold$i/aa_valid.np;
done
