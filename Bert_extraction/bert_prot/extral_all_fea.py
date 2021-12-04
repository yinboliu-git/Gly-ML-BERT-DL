import json
import sys
import joblib
import numpy as np


input_file = sys.argv[1]
output_file = sys.argv[2]
# label = sys.argv[3]

# fea = open('train_select_fea.txt','w')

tmp = []
for line in open(input_file,'r'):
    tmp.append(json.loads(line))

data_save_list = []
for i in range(len(tmp)):
    # 提取除[UNK]和[SEP]标记的所有特征
    data_save_list.append([])
    for f in range(len(tmp[i]['features'])):
        # print(tmp[i]['features'][f]['token'])
        #if tmp[i]['features'][f]['token'] == '[UNK]' or tmp[i]['features'][f]['token'] == '[SEP]':
        #    pass
        #else:
        data_save_list[i].append(tmp[i]['features'][f]['layers'][0]['values'])

need = [0]
need.extend(list(range(2,33)))
data_save_list = np.array(data_save_list)
data_save_list = data_save_list[:,need,:]

joblib.dump(data_save_list,output_file)


