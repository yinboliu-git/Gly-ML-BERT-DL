import math
import numpy as np
import pandas as pd


def AAC(frag):
    lines = frag
    L=len(lines[1])
    n=int(len(lines))
    AAs='ACDEFGHIKLMNPQRSTVWYO'
    m=len(AAs)
    aac=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            frequency=lines[i].count(AAs[j])
            frequency=float('%.2f'%frequency)
            aac[i][j]=frequency/L
    aac=aac[:,0:20]
    return np.array(aac)


def k_space(frag):
    lines = frag
    L = len(lines[1])
    n = int(len(lines))
    AAs = 'ACDEFGHIKLMNPQRSTVWYO'
    m = len(AAs)
    pair = []
    for i in range(m):
        for j in range(m):
            pair.append(AAs[i] + AAs[j])
    new_lines = []
    for i in range(n):
        new_lines.append(lines[i])

    for k in range(5):
        k_space = np.zeros((n, 441))
        for t in range(n):
            for i in range(L - k - 1):
                AApair = new_lines[t][i] + new_lines[t][i + k + 1]
                for j in range(441):
                    if AApair == pair[j]:
                        k_space[t][j] += 1
        if k == 0:
            Kspace = k_space
        else:
            # Kspace = np.concatenate((Kspace, k_space), axis=1)
            Kspace = Kspace + k_space
    return Kspace


def PWAA(frag):
    lines = frag
    L=len(lines[1])
    n=int(len(lines))
    AAs='ACDEFGHIKLMNPQRSTVWYO'
    l=int((L-1)/2)
    data=np.zeros((n,21))
    for i in range(n):
        for k in range(len(AAs)):
            pos=[ii for ii,v in enumerate(lines[i]) if v==AAs[k]]
            pos2=[jj+1 for jj in pos]
            p=[]
            c=[]
            for j in pos2:
                if j>=1 and j<=l:
                    p.append(j-l-1)
                if j>l and j<=L:
                    p.append(j-l-1)
            for m in p:
                if m>=-l and m<=l:
                    S1=float('%.2f'%abs(m))
                    c.append(m+S1/l)
            S2=float('%.2f'%sum(c))
            data[i][k]=S2/(l*(l+1))
    return data


def DBPB(frag, file_save=0, trainall=0, trainall_label=0):
    def generate_csv(trainall_label, trainall,file_save):
        # generate DBPB csvfile
        POS = []
        NEG = []
        for w in range(len(trainall_label)):
            if trainall_label[w] == 1:
                POS.append(trainall[w])
            else:
                NEG.append(trainall[w])

        print(len(POS))
        print(len(NEG))

        AAs = 'ACDEFGHIKLMNPQRSTVWYO'
        sym = []
        for i in AAs:
            for j in AAs:
                sym.append(str(i) + str(j))
        print(len(sym))

        X = []
        X.append(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
             56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
             83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
             108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
             129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
             150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
             171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
             192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
             213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
             234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
             255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
             276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
             297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
             318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
             339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
             360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380,
             381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401,
             402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422,
             423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440])
        for i in range(30):
            M = []
            for j in range(len(sym)):
                Y = 0
                for k in range(len(POS)):
                    if sym[j] == POS[k][i:i + 2]:
                        Y += 1
                Y /= (len(POS) * 1.0)
                M.append(Y)
            X.append(M)
        print(np.array(X).shape)
        np.savetxt(file_save + 'data_positive.csv', X, fmt='%s', delimiter=',')

        X = []
        X.append(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
             56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
             83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
             108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
             129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
             150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
             171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
             192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
             213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
             234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
             255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
             276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
             297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
             318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
             339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
             360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380,
             381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401,
             402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422,
             423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440])
        for i in range(30):
            M = []
            for j in range(len(sym)):
                Y = 0
                for k in range(len(NEG)):
                    if sym[j] == NEG[k][i:i + 2]:
                        Y += 1
                Y /= (len(NEG) * 1.0)
                M.append(Y)
            X.append(M)
        print(np.array(X).shape)
        np.savetxt(file_save + 'data_negative.csv', X, fmt='%s', delimiter=',')
        return 0

    if not isinstance(trainall,int):
        print('计算DBPB.csv开始...')
        generate_csv(trainall_label, trainall, file_save)

    lines = frag
    n=int(len(lines))
    L=len(lines[1])
    new_data=[]
    for i in range(n):
        new_data.append(lines[i])
    AAs='ACDEFGHIKLMNPQRSTVWYO'
    sym=[]
    for i in AAs:
        for j in AAs:
            sym.append(str(i)+str(j))
    pospath=file_save + 'data_positive.csv'
    data_positive=pd.read_csv(pospath)
    data_positive=data_positive.values
    negpath=file_save + 'data_negative.csv'
    data_negative=pd.read_csv(negpath)
    data_negative=data_negative.values

    DBPB=np.zeros((n,(L-1)*2))
    for i in range(n):
        for j in range(L-1):
            pos=sym.index(new_data[i][j:j+2])
            DBPB[i][j]=data_positive[j][pos]
            DBPB[i][j+L-1]=data_negative[j][pos]
    DBPB = DBPB / DBPB.max(axis=0)
    return DBPB


def EBGW(frag):
    lines = frag
    L = len(lines[1])
    l = L
    n = int(len(lines))
    C1 = 'AFGILMPVW'
    C2 = 'CNQSTY'
    C3 = 'HKR'
    C4 = 'DE'
    ucidata = []
    EBGW = []
    ucidata1 = []
    ucidata2 = []
    ucidata3 = []
    for i in range(n):
        ucida = []
        for j in range(l):
            pos = [ii for ii, v in enumerate(C1) if v == lines[i ][j]]
            pos1 = [ii for ii, v in enumerate(C2) if v == lines[i ][j]]
            pos2 = [ii for ii, v in enumerate(C3) if v == lines[i ][j]]
            pos3 = [ii for ii, v in enumerate(C4) if v == lines[i ][j]]
            if len(pos) == 1 or len(pos1) == 1:
                ucida.append(1)
            elif len(pos2) == 1 or len(pos3) == 1:
                ucida.append(0)
            else:
                ucida.append(0)
            pos = []
            pos1 = []
            pos2 = []
            pos3 = []

        ucidata1.append(ucida)

    for i in range(n):
        ucida1 = []
        for j in range(l):
            pos = [ii for ii, v in enumerate(C1) if v == lines[i][j]]
            pos1 = [ii for ii, v in enumerate(C3) if v == lines[ i ][j]]
            pos2 = [ii for ii, v in enumerate(C2) if v == lines[i ][j]]
            pos3 = [ii for ii, v in enumerate(C4) if v == lines[i][j]]
            if len(pos) == 1 or len(pos1) == 1:
                ucida1.append(1)
            elif len(pos2) == 1 or len(pos3) == 1:
                ucida1.append(0)
            else:
                ucida1.append(0)
            pos = []
            pos1 = []
            pos2 = []
            pos3 = []
        ucidata2.append(ucida1)
    for i in range(n):
        ucida2 = []
        for j in range(l):
            pos = [ii for ii, v in enumerate(C1) if v == lines[i][j]]
            pos1 = [ii for ii, v in enumerate(C3) if v == lines[ i ][j]]
            pos2 = [ii for ii, v in enumerate(C2) if v == lines[i ][j]]
            pos3 = [ii for ii, v in enumerate(C4) if v == lines[i][j]]
            if len(pos) == 1 or len(pos1) == 1:
                ucida2.append(1)
            elif len(pos2) == 1 or len(pos3) == 1:
                ucida2.append(0)
            else:
                ucida2.append(0)
            pos = []
            pos1 = []
            pos2 = []
            pos3 = []
        ucidata3.append(ucida2)
    ucidata = np.hstack((ucidata1, ucidata2, ucidata3))

    ur, uc = np.shape(ucidata)
    k1 = 5
    x1 = []
    x2 = []
    x3 = []
    for i in range(ur):
        x11 = []
        x22 = []
        x33 = []
        a = 0
        b = 0
        c = 0
        for j in range(int(k1)):
            a = sum(ucidata1[i][0:int(math.floor(l * (j + 1) / k1))]) / math.floor(l * (j + 1) / k1)
            b = sum(ucidata2[i][0:int(math.floor(l * (j + 1) / k1))]) / math.floor(l * (j + 1) / k1)
            c = sum(ucidata3[i][0:int(math.floor(l * (j + 1) / k1))]) / math.floor(l * (j + 1) / k1)
            x11.append(a)
            x22.append(b)
            x33.append(c)
        x1.append(x11)
        x2.append(x22)
        x3.append(x33)
    EBGW = np.hstack((x1, x2, x3))
    EBGW = np.array(EBGW)
    return EBGW


def CalculateCon(pos_index, myDistance, j):
        pos = pos_index[0:j]
        count = 0.0
        for i in pos:
            if int(myDistance[i][1]) == 1:
                count += 1
        feature_value = count / j
        return feature_value


def get_feature_knn(data_, k_list):
    blosum62 = [
        [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
        [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
        [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
        [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
        [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
        [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
        [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
        [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
        [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
        [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
        [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
        [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
        [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
        [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
        [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
        [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    ]
    AA = 'ARNDCQEGHILKMFPSTWYVO'
    myDict = {}
    for ran_i in range(len(AA)):
        myDict[AA[ran_i]] = ran_i
    maxValue, minValue = 11, -4
    Sim_function = lambda a, b: (blosum62[myDict[a]][myDict[b]] - minValue) / (maxValue - minValue)  # 使用匿名函数提高运算速度

    feature = []
    feature_all = []
    print('数据共长：{}'.format(len(data_)))
    print('计算个数：{}'.format(len(k_list)))

    cont = 0
    for i_data in data_:
        all_distance = []
        for j_data in data_:
            if i_data[0] != j_data[0]:
                sequence1 = i_data[0]
                sequence2 = j_data[0]
                count = 0.0
                d = len(sequence1)
                for i_d in range(d):
                    a = sequence1[i_d]
                    b = sequence2[i_d]

                    count += Sim_function(a, b)

                distance = 1 - count / d
                all_distance.append([j_data[0], j_data[1], distance])
        pos_index = np.argsort(np.array(np.array(all_distance).T)[2])

        feature_single = []
        for k in k_list:
            feature_single.append(CalculateCon(pos_index,all_distance, k))

        cont = cont + 1
        if cont % 100 == 0:
            print('已执行到{}...'.format(cont))
        feature_all.append(feature_single)
        feature.append([i_data[0], i_data[1], feature_single])
    print('特征提取完成...')

    return feature_all


def get_feature_knn_test(data_, train_data, k_list):
    blosum62 = [
        [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
        [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
        [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
        [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
        [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
        [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
        [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
        [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
        [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
        [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
        [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
        [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
        [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
        [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
        [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
        [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    ]
    AA = 'ARNDCQEGHILKMFPSTWYVO'
    myDict = {}
    for ran_i in range(len(AA)):
        myDict[AA[ran_i]] = ran_i
    maxValue, minValue = 11, -4
    Sim_function = lambda a, b: (blosum62[myDict[a]][myDict[b]] - minValue) / (maxValue - minValue)  # 使用匿名函数提高运算速度

    feature = []
    feature_all = []
    print('数据共长：{}'.format(len(data_)))
    print('计算个数：{}'.format(len(k_list)))
    cont = 0
    for i_data in data_:
        all_distance = []
        for j_data in train_data:
            if i_data[0] != j_data[0]:
                sequence1 = i_data[0]
                sequence2 = j_data[0]
                count = 0.0
                d = len(sequence1)
                for i_d in range(d):
                    a = sequence1[i_d]
                    b = sequence2[i_d]
                    count += Sim_function(a, b)
                distance = 1 - count / d
                all_distance.append([j_data[0], j_data[1], distance])
        pos_index = np.argsort(np.array(np.array(all_distance).T)[2])

        feature_single = []
        for k in k_list:
            feature_single.append(CalculateCon(pos_index,all_distance, k))

        cont = cont + 1
        if cont % 100 == 0:
            print('已执行到{}...'.format(cont))
        feature_all.append(feature_single)
        feature.append([i_data[0], i_data[1], feature_single])
    print('特征提取完成...')

    return feature_all


def get_xy():
    pass


def get_file(filename):
    x_test = []
    y_test = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split('\t')
            x_temp = ''
            for i in line[1]:
                if i not in [' ', '\n']:
                    x_temp += i
            x_test.append(x_temp)
            y_test.append(int(line[0]))
    return x_test,y_test


def get_feature(f_train, f_test=0, file_save=0):
    train, y_train = get_file(f_train)

    train_f1 = pd.DataFrame(AAC(train))
    train_f2 = pd.DataFrame(PWAA(train))
    train_f3 = pd.DataFrame( DBPB(train, file_save, train, y_train))
    train_f4 = pd.DataFrame( EBGW(train))
    train_f5 = pd.DataFrame(k_space(train))
    xy_train = list(np.concatenate(((np.array(train)).reshape(-1,1),(np.array(y_train)).reshape(-1,1)),axis=1))
    train_f6 =pd.DataFrame(get_feature_knn(xy_train,[2,4,8,16,32]))
    feature_train = pd.concat([train_f1,train_f2,train_f3,train_f4,train_f5,train_f6],axis=1)
    if f_test != 0:
        test, y_test = get_file(f_test)
        test_f1 = pd.DataFrame(AAC(test))
        test_f2 = pd.DataFrame(PWAA(test))
        test_f3 = pd.DataFrame( DBPB(test, file_save))
        test_f4 = pd.DataFrame( EBGW(test))
        test_f5 = pd.DataFrame(k_space(test))
        xy_test = list(np.concatenate(((np.array(test)).reshape(-1,1),(np.array(y_test)).reshape(-1,1)),axis=1))
        test_f6 =pd.DataFrame(get_feature_knn_test(xy_test,xy_train,[2,4,8,16,32]))
        feature_test = pd.concat([test_f1,test_f2,test_f3,test_f4,test_f5,test_f6],axis=1)
            # feature_train.to_csv('./f_train.csv')
            # feature_test.to_csv(('./f_test.csv'))
        return feature_train, feature_test
    return feature_train

import os

def process_get_feature(filename_read,filename_save, i, id):
    print('pid{}正常执行..'.format(id))
    filename_train_fold = filename_read + 'fold' + str(i) + '/train.tsv'
    # /mnt/raid5/data4/publice/gly_site_pred/test_data/
    filename_test_fold = filename_read + 'fold' + str(i) + '/dev.tsv'

    ftrain, ftest = get_feature(filename_train_fold, filename_test_fold, filename_save + 'fold' + str(i) + '/')

    filename_train_save = filename_save + 'fold' + str(i) + '/train.csv'
    filename_test_save = filename_save + 'fold' + str(i) + '/test.csv'
    if not os.path.exists(filename_save + 'fold' + str(i)):
        os.mkdir(filename_save + 'fold' + str(i))

    ftrain.to_csv(filename_train_save,  header=False, index=False)
    ftest.to_csv(filename_test_save, header=False, index=False)
    print('pid{}结束..'.format(id))

from multiprocessing import Process
import signal
def term(sig_num, addtion):
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def get_fold_feature(filename_read, filename_save, number=0):
    signal.signal(signal.SIGTERM, term)
    process_self = []
    process_id = 0
    print('.....')
    for i in range(1,number+1):
            print('.....')
            p = Process(target=process_get_feature, args=(filename_read,filename_save,i,process_id))
            # p.daemon = True
            process_id = process_id + 1
            process_self.append(p)
            print('pid{}正常构建'.format(process_id))

    for p_temp in process_self:

        p_temp.start()

    for p_temp in process_self:
        p_temp.join()


if __name__ == '__main__':
    f_read = '/mnt/raid5/data4/publice/gly_site_pred/test_data/'
    f_save = './feature_fold/'

    get_fold_feature(f_read, f_save, number=10)




