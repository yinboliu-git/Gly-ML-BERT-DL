from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#!/opt/share/bin/anaconda3/bin python
# coding: utf-8

import shap  # 这行代码没有用，但是不能删除，删除后不可运行！！！

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve,auc
from tensorflow_core.python.keras.backend import set_session
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.callbacks import Callback

np.random.seed(1337)
from sklearn.metrics import roc_auc_score, confusion_matrix
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
import sys,os

class UseANN(object):
    def __init__(self, fold, self_test=False, gpu_id=0):
        self.fold = fold
        self.self_test = self_test
        self.need = list(range(0,31))
        self.average = 'no'

        print("\n第{} fold开始..".format(self.fold))

    # 读取1D数据的时候使用，只读取CLS
    def get_CLS_data(self, train_file, label_file):
        print('从硬盘读取数据开始...')
        self.get_class = 'CLS' # 控制 generate_test 的读取
        self.train = []
        with open(train_file, 'r') as f:
            for line in f:
                line = line.split(',')
                self.train.append([float(x) for x in line[:]])

        label = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.split('\t')
                label.append(int(line[0]))

        self.train = np.array(self.train)
        le = preprocessing.LabelEncoder()
        le.fit(label)
        self.y_train = le.transform(label)
        self.train = self.train.astype(np.float32)
        try:
            shape_2 = self.train .shape[2]
        except Exception:
            shape_2 = 1
        self.train  = self.train.reshape(-1, self.train.shape[1])

        self.y_train = self.y_train.astype(np.float)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.y_train = self.y_train.reshape(len(self.y_train), 1)
        # train_label = np.array(train_label).reshape(len(train_label), 1)
        self.y_train = onehot_encoder.fit_transform(self.y_train)

        return self.train, self.y_train

    # 读取2D数据的时候使用,也就是31+3的数据
    def get_all_data(self, train_file, label_file):
        print('从硬盘读取数据开始...')
        self.get_class = 'all'
        import joblib
        self.train = np.array(joblib.load(train_file))
        self.train = self.train.astype(np.float32)
        # need = [1]
        need = self.need
        self.train = self.train[:, need, :]
        if isinstance(need, list):
            self.train = self.train.transpose((0, 2, 1))
            if self.average == 'yes':
                self.train = np.average(self.train, axis=2)

        label = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.split('\t')
                label.append(int(line[0]))
        le = preprocessing.LabelEncoder()
        le.fit(label)
        self.y_train = le.transform(label)

        try:
            shape_2 = self.train .shape[2]
        except Exception:
            shape_2 = 1
        self.train  = self.train.reshape(-1, self.train.shape[1])

        self.y_train = self.y_train.astype(np.float)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.y_train = self.y_train.reshape(len(self.y_train), 1)
        # train_label = np.array(train_label).reshape(len(train_label), 1)
        self.y_train = onehot_encoder.fit_transform(self.y_train)

        return self.train, self.y_train

    # 1d数据生成，以及2d数据使用1d卷积请使用此函数
    def generate_1D_test(self, test_file, test_label):
        self.y_test = []
        self.x_test = []
        def get_CLS_data_test(self):
            with open(test_file, 'r') as f:
                for line in f:
                    line = line.split(',')
                    self.x_test.append([float(x) for x in line[:]])
            self.x_test = np.array(self.x_test)

            self.y_test = []
            with open(test_label, 'r') as f:
                for line in f:
                    line = line.split('\t')
                    self.y_test.append(int(line[0]))

            le = preprocessing.LabelEncoder()
            le.fit(self.y_test)
            self.y_test = le.transform(self.y_test)
            self.x_test = self.x_test.astype(np.float32)

            return self.x_test, self.y_test

        # 读取2D数据的时候使用,也就是31+3的数据
        def get_all_data_test(self):
            import joblib
            self.x_test = joblib.load(test_file)
            self.x_test = np.array(self.x_test)
            # need = [0]
            # need.extend(list(range(1,33)))
            need = self.need
            self.x_test = self.x_test[:, need, :]
            if isinstance(need, list):
                self.x_test = self.x_test.transpose((0, 2, 1))
                if self.average == 'yes':
                    self.x_test = np.average(self.x_test, axis=2)

            self.y_test = []
            with open(test_label, 'r') as f:
                for line in f:
                    line = line.split('\t')
                    self.y_test.append(int(line[0]))
            le = preprocessing.LabelEncoder()
            le.fit(self.y_test)
            self.y_test = le.transform(self.y_test)
            self.x_test = self.x_test.astype(np.float32)

            return self.x_test, self.y_test

        if self.get_class == 'CLS':
            get_CLS_data_test(self)
        elif self.get_class == 'all':
            get_all_data_test(self)

        self.y_test = self.y_test
        self.x_test = np.array(self.x_test)
        try:
            shape_2 = self.x_test.shape[2]
        except Exception:
            shape_2 = 1
        self.x_test  = self.x_test .reshape(-1, self.x_test .shape[1])

        return self.x_test , self.y_test

    def generate_indp_test(self, test_file, test_label):
        self.y_test = self.y_test.astype(np.float)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.y_test = self.y_test.reshape(len(self.y_test), 1)
        # train_label = np.array(train_label).reshape(len(train_label), 1)
        self.y_test = onehot_encoder.fit_transform(self.y_test)

        self.train = np.concatenate((self.train,self.x_test), axis=0)
        self.y_train = np.concatenate((self.y_train, self.y_test), axis=0)

        try:
            shape_2 = self.train .shape[2]
        except Exception:
            shape_2 = 1
        self.train  = self.train.reshape(-1, self.train.shape[1])


        self.y_test = []
        self.x_test = []
        def get_CLS_data_test(self):
            with open(test_file, 'r') as f:
                for line in f:
                    line = line.split(',')
                    self.x_test.append([float(x) for x in line[:]])
            self.x_test = np.array(self.x_test)

            self.y_test = []
            with open(test_label, 'r') as f:
                for line in f:
                    line = line.split('\t')
                    self.y_test.append(int(line[0]))

            le = preprocessing.LabelEncoder()
            le.fit(self.y_test)
            self.y_test = le.transform(self.y_test)
            self.x_test = self.x_test.astype(np.float32)

            return self.x_test, self.y_test

        # 读取2D数据的时候使用,也就是31+3的数据
        def get_all_data_test(self):
            import joblib
            self.x_test = joblib.load(test_file)
            self.x_test = np.array(self.x_test)
            # need = [0]
            # need.extend(list(range(1,33)))
            need = self.need
            self.x_test = self.x_test[:, need, :]
            if isinstance(need, list):
                self.x_test = self.x_test.transpose((0, 2, 1))
                if self.average == 'yes':
                    self.x_test = np.average(self.x_test, axis=2)

            self.y_test = []
            with open(test_label, 'r') as f:
                for line in f:
                    line = line.split('\t')
                    self.y_test.append(int(line[0]))
            le = preprocessing.LabelEncoder()
            le.fit(self.y_test)
            self.y_test = le.transform(self.y_test)
            self.x_test = self.x_test.astype(np.float32)

            return self.x_test, self.y_test

        if self.get_class == 'CLS':
            get_CLS_data_test(self)
        elif self.get_class == 'all':
            get_all_data_test(self)

        self.y_test = self.y_test
        self.x_test = np.array(self.x_test)
        try:
            shape_2 = self.x_test.shape[2]
        except Exception:
            shape_2 = 1
        self.x_test  = self.x_test .reshape(-1, self.x_test .shape[1])

        return self.x_test , self.y_test

    def plt_tsne(self):
        y_ = np.argmax(self.y_train, axis=1)
        x_tsne = TSNE(n_components=2, learning_rate=100, random_state=501).fit_transform(self.train)

        plt.figure(figsize=(6, 6))
        r = 2  # 4
        area = np.pi * r ** 2  # 点面积

        plt.scatter(x_tsne[y_ == 0, 0], x_tsne[y_ == 0, 1], s=area, c='g', alpha=0.4, label='class-0')
        plt.scatter(x_tsne[y_ == 1, 0], x_tsne[y_ == 1, 1], s=area, c='b', alpha=0.4, label='class-1')
        plt.legend()
        plt.show()

        return plt


def plt_tsen(model_type_list=[],number=[1], train_dir='./', label_dir='./', ctl='CLS', bert_type='base',need=0,average='no'):
    if need == 0 and average == 'no':
        ctl_ = ctl
    elif need != 0 and average == 'no':
        ctl_ = 'K'
    elif need == 0 and average == 'yes':
        ctl_ = 'average'
    else:
        raise Exception('need,average')
    bert_type2 = ['bert-base', 'probert-base', 'Tape']
    i = 1
    for j in number:
        Train_ANN = UseANN(i)
        if need != 0:
            Train_ANN.need = need
        Train_ANN.average = average

        # 工作目录创建
        if not os.path.exists(bert_type):
            os.mkdir(bert_type)
        if not os.path.exists('./' + bert_type +'/' +ctl_):
            os.mkdir('./' + bert_type +'/' +ctl_)
        if not os.path.exists('./' + bert_type +'/' +ctl_ +'/'+ bert_type2[j]):
            os.mkdir('./' + bert_type +'/' +ctl_+ '/' + bert_type2[j])
            print('创建工作目录成功')

        # 数据的读取
        # if ctl == 'CLS':
        #     train_file = train_dir + '/fold' + str(i) + '/CLS_train.txt'
        #     label_file = label_dir + 'fold' + str(i) + '/train.tsv'
        #
        #     test_file = train_dir +  '/fold' + str(i) + '/CLS_valid.txt'
        #     test_label_file = label_dir + 'fold' + str(i) + '/valid.tsv'
        #
        #     indp_file = train_dir + '/fold' + str(i) + '/CLS_test.txt'
        #     indp_label_file = label_dir + 'fold' + str(i) + '/test.tsv'
        #
        #     Train_ANN.get_CLS_data(train_file,label_file)  # 这里决定使用什么CLS还是all_31
        #
        # if ctl == 'all':
        #     train_file = train_dir + '/fold' + str(i) + '/aa_train.np'
        #     label_file = label_dir  + 'fold' + str(i) + '/train.tsv'
        #
        #     test_file = train_dir + '/fold' + str(i) + '/aa_valid.np'
        #     test_label_file = label_dir + 'fold' + str(i) + '/dev.tsv'
        #
        #     indp_file = train_dir + '/fold' + str(i) + '/aa_test.np'
        #     indp_label_file = label_dir + 'fold' + str(i) + '/test.tsv'
        #     Train_ANN.get_all_data(train_file, label_file)  # 这里决定使用什么CLS还是all_31

        if ctl == 'CLS':
            train_file = train_dir + 'CLS_fea/' + 'fold' + str(i) + '/train.txt'
            label_file = label_dir + 'fold' + str(i) + '/train.tsv'

            test_file = train_dir + 'CLS_fea/' + 'fold' + str(i) + '/valid.txt'
            test_label_file = label_dir + 'fold' + str(i) + '/valid.tsv'

            indp_file = train_dir + 'CLS_fea/' + 'fold' + str(i) + '/test.txt'
            indp_label_file = label_dir + 'fold' + str(i) + '/test.tsv'

            Train_ANN.get_CLS_data(train_file, label_file)  # 这里决定使用什么CLS还是all_31

        if ctl == 'all':
            train_file = train_dir + 'all_fea/' + 'fold' + str(i) + '/train.np'
            label_file = label_dir + 'fold' + str(i) + '/train.tsv'

            test_file = train_dir + 'all_fea/' + 'fold' + str(i) + '/valid.np'
            test_label_file = label_dir + 'fold' + str(i) + '/dev.tsv'

            indp_file = train_dir + 'all_fea/' + 'fold' + str(i) + '/test.np'
            indp_label_file = label_dir + 'fold' + str(i) + '/test.tsv'
            Train_ANN.get_all_data(train_file, label_file)  # 这里决定使用什么CLS还是all_31


        print('训练集数据读取完成...')
        Train_ANN.generate_1D_test(test_file,test_label_file)
        print('数据读取完成...')

        Train_ANN.generate_indp_test(indp_file,indp_label_file)
        print('独立测试集数据读取完成...')

        plt_temp = Train_ANN.plt_tsne()
        if not os.path.exists('./img'):
            os.mkdir('./img')
        plt_temp.savefig('./img/'+ctl_ + '.png')

        del Train_ANN

if __name__ == '__main__':

    ctl_list = ['CLS']
    nlp_type = 'new_probert' # new_bert, new_probert, new_tape
    number = [0]

    train_dir = '/mnt/raid5/data4/publice/gly_site_pred/probert_fea/'
    label_dir = '/mnt/raid5/data4/publice/gly_site_pred/test_data/'  # 必须以 / 结尾

    # need = 15 # 0 = ctl, 15 = K,
    # average = 'no' # yes, no
    # m_type=CNN1D epoch=31  bs=32  l=2e-05 beta=0.88
    # model_list = [ 'CNN1D_BiLSTM', 'BiLSTM','CNN1D']

    model_list = ['CNN1D','BiLSTM']

    print(model_list)

    # /mnt/raid5/data4/publice/gly_site_pred/tape_fea/fold1
    for ctl in ctl_list:
        if ctl == 'CLS':
            need = 0
            average = 'no'
            plt_tsen(model_type_list=model_list, number=number, train_dir=train_dir,
                         label_dir=label_dir, ctl=ctl, bert_type=nlp_type, need=need, average=average)

        if ctl == 'all':
            # need = 0
            # average = 'no'
            # plt_tsen(model_type_list=model_list,  number=number, train_dir=train_dir,
            #                          label_dir=label_dir, ctl=ctl, bert_type=nlp_type, need=need,average=average)

            need = 15
            average = 'no'
            plt_tsen(model_type_list=model_list, number=number, train_dir=train_dir,
                                     label_dir=label_dir, ctl=ctl, bert_type=nlp_type, need=need,average=average)

            need = 0
            average = 'yes'
            plt_tsen(model_type_list=model_list,  number=number, train_dir=train_dir,
                         label_dir=label_dir, ctl=ctl, bert_type=nlp_type, need=need,average=average)

    # use_fold_LSTM(param=[10,32,2e-4],number=3,train_dir=train_dir,label_dir=label_dir)

