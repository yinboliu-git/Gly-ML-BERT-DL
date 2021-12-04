import matplotlib.pyplot as plt
import numpy as np

# 设置画布的大小
plt.figure(figsize=(8,6),dpi=600)

# 输入统计数据
name = ('Sen', 'Spe', 'Mcc','ACC','AUC') # 在这里设置名字
GLy_pseaac = [0.156, 0.183,0,0.170,0.170] # 在这里设置高度
BPB_Glysite = [0.510, 0.506,0.156,0.508,0.499]
GlyNN = [0.538, 0.527,0.074,0.535,0.539]
my_model = [0.49,0.73,0.226623552,0.61,0.689175]

bar_width = 0.20  # 条形宽度
index_a = np.arange(len(name))
index_b = index_a + bar_width
index_c = index_b + bar_width
index_d = index_c + bar_width
y = [GLy_pseaac,BPB_Glysite,GlyNN,my_model]
x = [index_a, index_b, index_c,index_d]
plt.grid(zorder=0)

# 使用4次 bar 函数画出两组条形图
plt.bar(index_a, height=GLy_pseaac, width=bar_width, color='b', label='Gly_PseAAC',zorder=10)
plt.bar(index_b, height=BPB_Glysite, width=bar_width, color='y', label='BPB_Glysite',zorder=10)
plt.bar(index_c, height=GlyNN, width=bar_width, color='c', label='GlyNN',zorder=10)
plt.bar(index_d, height=my_model, width=bar_width, color='r', label='My_Model',zorder=10)


plt.legend()  # 显示图例
plt.xticks(index_a+bar_width*1.5, name)  # 将横坐标显示在每个坐标的中心位置
plt.ylim([0, 1.1])

# 给柱状图添加高度
Font_1 = {
    'size': 6,
    'weight': 'bold'
}
for x_ind,y_ind in zip(x,y):
        for i in range(len(x_ind)):
                xx1 = x_ind[i]
                yy1 = y_ind[i]
                plt.text(xx1, yy1+0.01, ('%.2f' % yy1),fontdict=Font_1, ha='center', va='bottom',zorder=11)

Font_2 = {
            'size': 15,
        'weight': 'bold'
    }

plt.xlabel('Evaluating Indicator', Font_2)  # 横坐标轴标题
plt.ylabel('Values', Font_2)  # 纵坐标轴标题
# plt.title('test')  # 图形标题
plt.savefig('./me_other.png')
