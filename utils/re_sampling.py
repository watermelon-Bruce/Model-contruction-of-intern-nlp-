from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

'''
oversampling and downsampling
for oversampling, i use the SMOTE 
'''


def over_sampling(matrix1,label1,matrix2,label2):
    print("正在进行过采样")
    print("两样本数量分别为", len(matrix1), " ", len(matrix2))
    length1 = len(matrix1)  # 记录两个长度
    length2 = len(matrix2)
    l1 = label1[0]  # 记录两个label
    l2 = label2[0]
    number = int(input("对于数量较少的一类，想要扩充至多少？"))
    for i in range(len(label2)):
        matrix1.append(matrix2[i])
        label1.append(label2[i])
    matrix = matrix1
    label = label1
    if length1 < length2:
        smo = SMOTE(random_state=42, sampling_strategy={l1: number})
        matrix_smo, label_smo = smo.fit_resample(matrix, label)
    else:
        smo = SMOTE(random_state=42, sampling_strategy={l2: number})
        matrix_smo, label_smo = smo.fit_resample(matrix, label)
    print("过采样后，两者数量分别为", Counter(label_smo))
    return matrix_smo, label_smo


def under_sampling(matrix1,label1,matrix2,label2):
    print("正在进行负采样")
    print("两样本数量分别为", len(matrix1), " ", len(matrix2))
    length1 = len(matrix1)  # 记录两个长度
    length2 = len(matrix2)
    l1 = label1[0]  # 记录两个label
    l2 = label2[0]
    number = int(input("对于数量较多的一类，想要减少至多少？"))
    for i in range(len(label2)):
        matrix1.append(matrix2[i])
        label1.append(label2[i])
    matrix = matrix1
    label = label1
    if length1<length2:
        undersample=RandomUnderSampler(random_state=1,sampling_strategy={l2:number})
        matrix_under,label_under=undersample.fit_resample(matrix,label)
    else:
        undersample = RandomUnderSampler(random_state=1, sampling_strategy={l1: number})
        matrix_under, label_under = undersample.fit_resample(matrix, label)
    print("过采样后，两者数量分别为", Counter(label_under))
    return matrix_under,label_under




def re_sampling(flag,matrix1,label1,matrix2,label2):
    if flag==0:
        for i in range(len(label2)):
            matrix1.append(matrix2[i])
            label1.append(label2[i])
        matrix = matrix1
        label = label1
        return matrix,label

    if flag==1:
        matrix,label=over_sampling(matrix1,label1,matrix2,label2)
        return matrix,label

    if flag==2:
        matrix,label=under_sampling(matrix1,label1,matrix2,label2)
        return matrix,label

    if flag==3:
        matrix, label = over_sampling(matrix1, label1, matrix2, label2)
        fixlabel=label[0]
        matrix1=[]
        label1=[]
        matrix2=[]
        label2=[]
        for i in range(len(label)):
            if label[i]==fixlabel:
                matrix1.append(matrix[i])
                label1.append(label[i])
            else:
                matrix2.append(matrix[i])
                label2.append(label[i])
        matrix,label=under_sampling(matrix1, label1, matrix2, label2)
        return matrix,label


