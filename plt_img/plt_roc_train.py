from multiprocessing import Process
from sklearn import metrics
import pylab as plt
import numpy as np


# 画roc曲线
from sklearn.metrics import roc_auc_score


def ks(y_predicted1, y_true1, y_predicted2, y_true2, y_predicted3, y_true3,k,name):
    Font = {'size': 18, 'family': 'Times New Roman'}

    label1 = y_true1
    label2 = y_true2
    label3 = y_true3
    fpr1, tpr1, thres1 = metrics.roc_curve(label1, y_predicted1[:,1])
    fpr2, tpr2, thres2 = metrics.roc_curve(label2, y_predicted2[:,1])
    fpr3, tpr3, thres3 = metrics.roc_curve(label3, y_predicted3[:,1])
    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    roc_auc3 = metrics.auc(fpr3, tpr3)

    plt.figure(figsize=(9,9), dpi=600)
    plt.plot(fpr1, tpr1, 'b', label= k[0] + ' = %0.3f' % roc_auc1, color='Red')
    plt.plot(fpr2, tpr2, 'b', label=k[1]+' = %0.3f' % roc_auc2, color='k')
    plt.plot(fpr3, tpr3, 'b', label=k[2]+ ' = %0.3f' % roc_auc3, color='RoyalBlue')
    plt.legend(loc='lower right', prop=Font)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=15)
    plt.title(name.upper())

    plt.grid()  # linewidth并不是网格宽度而是网格线的粗细
    plt.savefig('./img/' + name)
    plt.show()

    return abs(fpr1 - tpr1).max(), abs(fpr2 - tpr2).max(), abs(fpr3 - tpr3).max()


def ks_2(y_predicted1, y_true1, y_predicted2, y_true2,name):
    Font = {'size': 18, 'family': 'Times New Roman'}

    label1 = y_true1
    label2 = y_true2

    fpr1, tpr1, thres1 = metrics.roc_curve(label1, y_predicted1)
    fpr2, tpr2, thres2 = metrics.roc_curve(label2, y_predicted2)

    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr1, tpr1, 'b', label='Shap = %0.3f' % roc_auc1, color='Red')
    plt.plot(fpr2, tpr2, 'b', label='No-shap = %0.3f' % roc_auc2, color='Blue')

    plt.legend(loc='lower right', prop=Font)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=15)

    plt.title(name.upper()+'-XGBoost')
    plt.show()
    return abs(fpr1 - tpr1).max(), abs(fpr2 - tpr2).max()

# 获取scores
from matplotlib.colors import cnames
def ks_multi(name,auc,*args):
    import pylab as plt
    Font = {'size': 18, 'family': 'Times New Roman'}
    Font_2 = {
            'size': 18,
        'weight': 'bold'
    }
    color_list = list(cnames.values())  ## 颜色限制 ！！！
    plt.figure(figsize=(9, 9), dpi=600)

    print(int(len(args[0])/2))
    for i in range(0,len(args)):
        print(i)
        label = args[i][2]
        y_predicted = args[i][1]
        try:
            y_predicted.shape[1]
            y_predicted = y_predicted[:,1]
        except:
            y_predicted = y_predicted
        fpr1, tpr1, thres1 = metrics.roc_curve(label, y_predicted)
        roc_auc = auc[args[i][0]]
        print(roc_auc)
        label = 'AUROC(' + args[i][0] + ')=' + '%0.3f' % roc_auc
        plt.plot(fpr1, tpr1, 'b', label=label, color=color_list[i*7 + 20],linewidth=2)


    plt.legend(loc='lower right', prop=Font) # 图例
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', Font_2)
    plt.xlabel('False Positive Rate', Font_2)
    plt.tick_params(labelsize=15)
    plt.title(name.upper(), Font_2)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细

    plt.grid()  # linewidth并不是网格宽度而是网格线的粗细
    plt.savefig('./'+MD+'l-img/' + name)
    del plt
    return 0


import joblib
def read_data(root_data_name):
    data = joblib.load(root_data_name)

    y_true = data[0]
    scores = data[2]

    return y_true,scores
def read_file(file,f,number):
    # /mnt/raid5/data3/ybliu/Gly/Gly_ml/single_feature_fold/AAC/fold1
        data_all = 0
        roc_auc = 0
        roc_auc = {}
        for i in number:
            fn = file + f + '/fold' + str(i) + '/scores.np'
            data = joblib.load(fn)

            if i == 1:
                for k in data.keys():
                    roc_auc[k] = roc_auc_score(data[k][0], data[k][2][:,1])
                data_all = data
            else:
                for k in data.keys():
                    roc_auc[k] += roc_auc_score(data[k][0], data[k][2][:,1])
                    data_all[k][0] = np.concatenate((data_all[k][0],data[k][0]),axis=0)
                    data_all[k][1] = np.concatenate((data_all[k][1],data[k][1]),axis=0)
                    data_all[k][2]  = np.concatenate((data_all[k][2],data[k][2]),axis=0)
        for k in data.keys():
            roc_auc[k] = roc_auc[k]/len(number)
        print(roc_auc)
        return data_all,roc_auc


if __name__ == '__main__':

 ## -- ML --##
 # feature_dict = {'AAC': 1, 'PWAA': 2, 'DBPB': 3, 'EBGW': 4, 'kspace': 5,
 #                 'KNN': 6}
 # for f in list(feature_dict.keys()):
 #    MD = 'M'
 #    name = f
 #    file = '/mnt/raid5/data3/ybliu/Gly/Gly_ml/single_feature_fold/'
 ## -- ML --##

 ## -- DL ——##
 for type_ in ['tape', 'bert', 'probert']:
  for f in ['CLS','average', 'K']:
    MD = 'D'
    name = f + '-' + type_
    file = '/mnt/raid5/data3/ybliu/Gly/Gly_SLTM/score_'+ type_ +'/'
 ## -- DL -- ##

    number = [1,2,3,4,5,6,7,8,9,10]
    data,roc_auc = read_file(file,f,number)
    y_test = []
    y_pred = []
    k = []
    for k_ in data.keys():
        y_test.append(data[k_][0])
        y_pred.append(data[k_][2])
        k.append(k_)

    # print(y_test)
    process_self = []
    process_id = 0
    print('.....')
    for i in [1]:
        print('.....')
        if MD == 'M':
            p = Process(target=ks_multi, args=(name,roc_auc, (k[0],y_pred[0][:,1],y_test[0]),(k[1],y_pred[1][:,1],y_test[1]),(k[2],y_pred[2][:,1],y_test[2]),(k[3],y_pred[3][:,1],y_test[3])))
        else:
            p = Process(target=ks_multi, args=(name,roc_auc, (k[0],y_pred[0][:,1],y_test[0]),(k[1],y_pred[1][:,1],y_test[1]),(k[2],y_pred[2][:,1],y_test[2])))

        # p.daemon = True
        process_id = process_id + 1
        process_self.append(p)
        print('pid{}正常构建'.format(process_id))

    for p_temp in process_self:
        p_temp.start()
