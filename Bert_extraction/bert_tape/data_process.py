import numpy as np
import joblib
import sys


input_file = sys.argv[1]
arrays = np.load(input_file, allow_pickle=True)

output_CLS_file = sys.argv[2]
output_aa_file = sys.argv[3]
CLS_feaflie = open(output_CLS_file, 'w')
aa_feafile = open(output_aa_file, 'wb')
keys = list(arrays.keys())
all_fea = dict()
for i in range(len(keys)):
# for i in range(1,5):
    #print(keys[i])
    item_dict = arrays['{}'.format(i)].item()
    dict_keys = item_dict.keys()
    seq = item_dict['seq']
    #print(dict_keys)
    all_fea[keys[i]] = seq
#print('----------------')
#print(all_fea['5'])
#print('----------------')
# 排序
all_sort_fea = sorted(all_fea.items(), key=lambda x: x[0])
#print(all_sort_fea[0][1][32])

print(len(keys))
CLS_fea = []
aa_fea = []
for i in range(len(keys)):
# for i in range(0, 4):
    CLS = np.array(all_sort_fea[i][1][0])
    # print(CLS)
    for j in range(len(CLS)):
        if j != len(CLS)-1:
            CLS_feaflie.write(str(CLS[j]) + ',')
        else:
            CLS_feaflie.write(str(CLS[j]))
    CLS_feaflie.write('\n')

    # aa_feafile.write(str([m for m in all_sort_fea[i][1][1:32]]))
    aa_fea.append([])
    for aa in all_sort_fea[i][1][1:32]:
        aa_fea[i].append([m for m in aa])

    # aa_feafile.write('\n')

joblib.dump(np.array(aa_fea), aa_feafile)
