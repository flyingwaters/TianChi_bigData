# @ Fenglongyu
# @ digging_for_happiness data preprocess
# @ 2019年12月
from sklearn.decomposition import PCA
import numpy as np
import torch
j = open(r'data/1.csv', "r", encoding="GB2312")
# 训练集,标签集和验证集 split train.txt 按照比例

num = 0
train_attr = j.readline()
# 每个属性的名称
train = []
label = []

# 训练数据标签

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

for i in j:
    temp = []
    ll = i.split(',')
    if int(ll[1]) < 0:
        continue
    p = ll[-1]
    t_str = ll[-1].replace("\n", "")
    ll[-1] = t_str

    time = ll[5]
    s = time.split(' ')
    time = s[0]

    # 切割出年份
    year = time[0:4]

    # 切割月份
    if time[6] != '/':
        month = time[5:7]
    else:
        month = time[5]

    # 切割日期
    if time[6] == '/':
        day = time[7:]
    else:
        day = time[8:]
    # print(year, month, day)
# 分割　year month day
# 三个部分
    for index, ite in enumerate(ll):
        if index == 0:
            continue
        # 除去几行中文数据？和调查时间数据？
        if index == 28 or index == 5:
            continue
        if index == 88 or index == 12:
            continue
        if index == 1:
            # label.append(eval(ite))
            continue
        if ite == '':
            ite = 0
        temp.append(float(ite))

    train.append(temp)
    # 对数据进行均值化和归一化,
    #　结尾加入三个日期的维度

np_train = np.array(train)

# 维度的统一 8000*136
# (8000,) ----> 8000*136
m = np.mean(np_train, axis=0)
m = np.expand_dims(m, axis=0)
std = np.std(np_train, axis=0)
std = np.expand_dims(std, axis=0)
res_train = ((np_train - m) / (std+0.000001)+1)/2

# res_label = np.array(label)
# res_label = np.expand_dims(res_label, axis=1)

#
pca = PCA(n_components=70, whiten=True)
#
pca.fit(res_train)
fin = pca.transform(res_train)
#
fin_validation = torch.from_numpy(fin)
print("valid_shape:", fin_validation.shape)



# 训练数据归一化过程
# train_x, test_x, train_y, test_y = train_test_split(fin_train, res_label, test_size=0.1, random_state=1)









