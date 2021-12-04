import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import xgboost as xgb


def xgb_xlf(md=2, lr=0.01, nesti=1800):
    return xgb.XGBClassifier(max_depth=md,
                             learning_rate=lr,
                             n_estimators=nesti,
                             objective='binary:logistic',
                             nthread=-1,
                             gamma=0,
                             min_child_weight=1,
                             max_delta_step=0,
                             subsample=0.85,
                             colsample_bytree=0.7,
                             colsample_bylevel=1,
                             reg_alpha=0,
                             reg_lambda=1,
                             scale_pos_weight=1,
                             seed=1440,
                             missing=None,
                             n_jobs=-1)


def svm_xlf(svm_c=1, svm_g=2 ** -3):
    return svm.SVC(C=svm_c, kernel='rbf', gamma=svm_g, probability=True)


def rf_xlf(md=4, nesti=1400):
    return RandomForestClassifier(max_depth=md, n_estimators=nesti,random_state=1440, n_jobs=-1)


def knn_xlf(n=2, p=1):
    return KNeighborsClassifier(n_neighbors=n, p=p, weights="distance", algorithm="auto", n_jobs=-1)


param_grid = {
    'rf': {
        'max_depth': [2, 4, 8],
        'n_estimators': [1600, 1800, 2000],
    },  # 请改回来 测试用
    'xgb': {
        'max_depth': [2, 4, 8],
        'learning_rate': [0.005, 0.01, 0.02],
        'n_estimators': [1800, 2000, 2200, ],
    },
    'svm': {
        "kernel": ['rbf'],
        "gamma": [2 ** i for i in range(-7, -3)],
        "C": [2 ** i for i in range(-1, 3)],
    },

    'knn': {
        'n_neighbors': [2, 4, 6],
        'p': [1, 2, 3, 4],
    }

}


# param_grid = {
#     'rf': {
#         'max_depth': [2, ],
#         'n_estimators': [1000, ],
#     },  # 请改回来 测试用
#     'xgb': {
#         'max_depth': [1, ],
#         'learning_rate': [0.005, ],
#         'n_estimators': [1400, ],
#     },
#     'svm': {
#         "kernel": ['rbf'],
#         "gamma": [2 ** -3],
#         "C": [2 ** 1],
#     },
#
#     'knn': {
#         'n_neighbors': [3],
#         'p': [1],
#     }
#
# }

xlf_dict = {'rf': rf_xlf(), 'svm': svm_xlf(),'xgb': xgb_xlf(), 'knn': knn_xlf()}

from sklearn.model_selection import ParameterGrid


def get_data(train_file, label_file):
    print('从硬盘读取数据开始...')
    x_train = pd.read_csv(train_file, header=None, index_col=None)
    x_train = np.array(x_train)

    label = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.split('\t')
            label.append(int(line[0]))

    label = np.array(label)

    return x_train, label


def grid_xlf(key, f_key, xlf, param, train, y_train, x_test, y_test):
    for i in param.keys():
        if not (hasattr(xlf, i)):
            raise Exception('xlf_append: {} 属性在{}中不存在..'.format(i, xlf()))
    param_grid_dict = [param]

    param_EI_list = []
    score_list = [[],[],[]]
    for param_sigle in param_grid_dict:
        print(key, f_key, param_sigle)

        for keys in param.keys():
            setattr(xlf, keys, param_sigle[keys])

        xlf.fit(train, y_train)
        yy_pred = xlf.predict_proba(x_test)

        y_pred = np.argmax(yy_pred, axis=1)
        true_values = np.array(y_test)
        y_scores = yy_pred[:, 1]
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        ACC = (TP + TN) / (TP + FP + FN + TN)
        Precision = TP / (TP + FP)
        F1Score = 2 * TP / (2 * TP + FP + FN)
        MCC = ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
        AUC = roc_auc_score(true_values, y_scores)
        pre, rec, thresholds = precision_recall_curve(y_test, y_scores)
        prc_area = auc(rec, pre)
        EI = [key, f_key, param_sigle, TN, FP, FN, TP, Precision, Sensitivity, Specificity, F1Score, MCC, ACC, prc_area,
              AUC]
        print('auc:{}\n'.format(AUC))
        param_EI_list.append(EI)
        score_list[0] = y_test
        score_list[1] = y_pred
        score_list[2] = yy_pred

    return EI, score_list


import os,ast,joblib

def get_param(file):
    data = pd.read_excel(file, index_col=False)
    return  data


def fold_grid_xlf(file_featrue, filename_label, save_file='./save_best_param/', number=0):
    EI_mean = []
    need_aac = list(range(0, 20))
    need_pwaa = list(range(max(need_aac) + 1, max(need_aac) + 22))
    need_dbpb = list(range(max(need_pwaa) + 1, max(need_pwaa) + 61))
    need_ebgw = list(range(max(need_dbpb) + 1, max(need_dbpb) + 16))
    need_kspace = list(range(max(need_ebgw) + 1, max(need_ebgw) + 442))
    need_knn = list(range(max(need_kspace) + 1, max(need_kspace) + 6))

    feature_dict = {'AAC': need_aac, 'PWAA': need_pwaa, 'DBPB': need_dbpb, 'EBGW': need_ebgw, 'kspace': need_kspace,
                    'KNN': need_knn}

    data = pd.read_excel(file_featrue + './ml_param.xlsx', index_col=False)

    for f_key in feature_dict.keys():
        save_file = './single_feature_fold/' + f_key + '/'
        if not os.path.exists('./single_feature_fold/'):
            os.mkdir('./single_feature_fold/')
        if not os.path.exists(save_file):
            os.mkdir(save_file)
        print('格点搜索开始...')
        for i in range(1, number + 1):
            # /mnt/raid5/data3/ybliu/Gly/Gly_ml/feature_fold/fold1
            train_label = filename_label + 'fold' + str(i) + '/train.tsv'
            test_label = filename_label + 'fold' + str(i) + '/dev.tsv'

            train_file_fold = file_featrue + 'fold' + str(i) + '/train.csv'
            test_file_fold = file_featrue + 'fold' + str(i) + '/test.csv'

            x_train_, y_train = get_data(train_file_fold, train_label)
            x_test_, y_test = get_data(test_file_fold, test_label)

            if not os.path.exists(save_file + 'fold' + str(i)):
                os.mkdir(save_file + 'fold' + str(i))

            x_train, x_test = x_train_[:, feature_dict[f_key]], x_test_[:, feature_dict[f_key]]

            score_dict = {}
            EI_list = []
            for key in xlf_dict.keys():
                param = data.iloc[list((data.iloc[:,0] == f_key) & (data.iloc[:,1] == key)),2].item()
                param = ast.literal_eval(param)
                EI, score = grid_xlf(key, f_key, xlf_dict[key], param, x_train, y_train, x_test, y_test)
                score_dict[key] = score
                EI_list.append(EI)

            EI_list = pd.DataFrame(EI_list, columns=None, index=None)
            if i == 1:
                    EI_mean = EI_list
            else:
                    EI_mean.iloc[:, 3:] = EI_mean.iloc[:, 3:] + EI_list.iloc[:, 3:]

            EI_list.to_csv(save_file + 'fold' + str(i) + '/param_EI.csv', header=False, index=False)
            joblib.dump(score_dict,save_file + 'fold' + str(i) + '/scores.np')


        EI_mean.iloc[:, 3:] = EI_mean.iloc[:, 3:] / number

        keys = set(EI_mean.iloc[:, 0])

        EI_best = []
        for key in keys:
                EI_temp = EI_mean[EI_mean.iloc[:, 0] == key]
                EI_best.append(EI_mean.iloc[EI_temp.iloc[:, -1].idxmax(), :])

        EI_best = pd.DataFrame(EI_best)
        EI_best.to_csv(save_file + 'best_param_EI.csv', header=False, index=False)


if __name__ == '__main__':
    data_file = './feature_fold/'
    label_file = "/mnt/raid5/data4/publice/gly_site_pred/test_data/"
    fold_grid_xlf(data_file, label_file, number=10)
