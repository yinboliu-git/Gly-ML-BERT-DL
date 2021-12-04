import numpy as np
import pandas as pd


# def get_fold_best_param(filename):
def get_mean(filename,number, list_grid=[0,135,270]):
        mean_list = []
        for i in range(1,number+1):
            f = filename+'fold' + str(i) +'/parm_EI.csv'
            data = pd.read_csv(f, header=None,index_col=None)
            a = data[0:6]
            if i == 1:
                mean_list = data
            else:
                if mean_list.iloc[:,0:6].equals(data.iloc[:,0:6]):
                    mean_list.iloc[:,6:] = mean_list.iloc[:,6:] + data.iloc[:,6:]
                else:
                    ctl = 0
                    long_ = list_grid[1]-list_grid[0]
                    for j in list_grid:
                        for k in list_grid:
                            if mean_list.iloc[j:j+long_,0:6].equals(data.iloc[k:k+long_,0:6]):
                                mean_list.iloc[j:j+long_, 6:] = mean_list.iloc[j:j+long_, 6:] + data.iloc[k:k+long_, 6:]
                                ctl = ctl+1
                                break
                    if ctl !=3:
                        print('错误')
                        return 0


        mean_list.iloc[:,6:] = mean_list.iloc[:,6:]/number

        return mean_list


def get_best(mean_list):
        a1 = mean_list.iloc[list(mean_list.iloc[:,0] == 'CNN1D'),:]
        a2 = mean_list.iloc[list(mean_list.iloc[:,0] == 'BiLSTM'),:]
        a3 = mean_list.iloc[list(mean_list.iloc[:,0] == 'CNN1D_BiLSTM'),:]

        best_param = []
        best_param.append(mean_list.iloc[a1.iloc[:,-1].idxmax(),:])
        best_param.append(mean_list.iloc[a2.iloc[:, -1].idxmax(), :])
        best_param.append(mean_list.iloc[a3.iloc[:, -1].idxmax(), :])

        return pd.DataFrame(best_param)

if __name__ == '__main__':
    filename = './score_bert/CLS/'
    mean_list = get_mean(filename, 10, [0,int(108/3 * 1), int(108/3 *2)])
    best_data = get_best(mean_list)
    h = ['method', 'b', 'l', 'beta', 'e',',', 'TN', 'FP', 'FN', 'TP', 'Precision', 'Sensitivity', 'Specificity',
         'F1Score', 'MCC', 'ACC', 'prc_area', 'AUC']
    best_data.to_csv(filename + 'best_param_EI.csv', header=h)
    print('获取最优参数完成...')
