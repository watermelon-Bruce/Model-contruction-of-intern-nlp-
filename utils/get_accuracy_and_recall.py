import numpy as np

'''
in here,we define different way to calculate the accuracy and recall
based on some real demand
'''


def get_accuracy_and_recall(original_label,predict_label):

    right_number = 0
    for i in range(len(original_label)):
        '''
        如果原本的label是001：国际工程
        那么预测为001：国际工程，002：电力工程，004：航空行业，005：汽车制造，006：石油行业，007：化工行业
        都算正确
        '''
        if original_label[i] == 1:
            if predict_label[i] == 1 or predict_label[i] == 2 or predict_label[i] == 4 or predict_label[i] == 5 or \
                    predict_label[i] == 6 or predict_label[i] == 7 or predict_label[i] == 11:
                right_number += 1
        '''
        如果label为002：电力工程，那么预测为001国际工程也算正确
        '''
        if original_label[i] == 2:
            if predict_label[i] == 1 or predict_label[i] == 2:
                right_number += 1
        '''
        如果label为003：财经行业，那么预测为008法律行业也算正确
        因为法律行业的语料总共只有几十条，而且法律行业的语料主要是关于金融法律。所以这两者不做区分
        '''
        if original_label[i] == 3:
            if predict_label[i] == 8 or predict_label[i] == 3:
                right_number += 1

        if original_label[i] == 4:
            if predict_label[i] == 1 or predict_label[i] == 4:
                right_number += 1

        if original_label[i] == 5:
            if predict_label[i] == 1 or predict_label[i] == 5:
                right_number += 1
        '''
        如果label为006：石油行业
        那么预测成为001：国际工程或者007化工行业也算正确
        因为化工行业和石油行业的语料非常相近，两者构成的分类器分类效果也微乎其微
        '''
        if original_label[i] == 6:
            if predict_label[i] == 1 or predict_label[i] == 6 or predict_label[i] == 7:
                right_number += 1

        if original_label[i] == 7:
            if predict_label[i] == 1 or predict_label[i] == 7 or predict_label[i] == 6:
                right_number += 1

        if original_label[i] == 8:
            if predict_label[i] == 3 or predict_label[i] == 8:
                right_number += 1

        if original_label[i] == 9:
            if predict_label[i] == 9:
                right_number += 1

        if original_label[i] == 11:
            if predict_label[i] == 11 or predict_label[i] == 1:
                right_number += 1
    accuracy=right_number / len(original_label)



    recall_number = {}
    category_number = {}
    for k in range(1, 12):
        recall_number[k] = 0
        category_number[k] = 0
    recall_number.pop(10)
    category_number.pop(10)

    for i in range(len(original_label)):
        '''
                    如果原本的label是001：国际工程
                    那么预测为001：国际工程，002：电力工程，004：航空行业，005：汽车制造，006：石油行业，007：化工行业
                    都算召回
                    '''
        if original_label[i] == 1:
            category_number[1] += 1
            if predict_label[i] == 1 or predict_label[i] == 2 or predict_label[i] == 4 or predict_label[i] == 5 or \
                    predict_label[i] == 6 or predict_label[i] == 7 or predict_label[i] == 11:
                recall_number[1] += 1
        '''
        如果label为002：电力工程，那么预测为001国际工程也算召回
        '''
        if original_label[i] == 2:
            category_number[2] += 1
            if predict_label[i] == 1 or predict_label[i] == 2:
                recall_number[2] += 1
        '''
        如果label为003：财经行业，那么预测为008法律行业也算召回
        因为法律行业的语料总共只有几十条，而且法律行业的语料主要是关于金融法律。所以这两者不做区分
        '''
        if original_label[i] == 3:
            category_number[3] += 1
            if predict_label[i] == 8 or predict_label[i] == 3:
                recall_number[3] += 1

        if original_label[i] == 4:
            category_number[4] += 1
            if predict_label[i] == 1 or predict_label[i] == 4:
                recall_number[4] += 1

        if original_label[i] == 5:
            category_number[5] += 1
            if predict_label[i] == 1 or predict_label[i] == 5:
                recall_number[5] += 1
        '''
        如果label为006：石油行业
        那么预测成为001：国际工程或者007化工行业也算正确
        因为化工行业和石油行业的语料非常相近，两者构成的分类器分类效果也微乎其微
        '''
        if original_label[i] == 6:
            category_number[6] += 1
            if predict_label[i] == 1 or predict_label[i] == 6 or predict_label[i] == 7:
                recall_number[6] += 1

        if original_label[i] == 7:
            category_number[7] += 1
            if predict_label[i] == 1 or predict_label[i] == 7 or predict_label[i] == 6:
                recall_number[7] += 1

        if original_label[i] == 8:
            category_number[8] += 1
            if predict_label[i] == 3 or predict_label[i] == 8:
                recall_number[8] += 1

        if original_label[i] == 9:
            category_number[9] += 1
            if predict_label[i] == 9:
                recall_number[9] += 1

        if original_label[i] == 11:
            category_number[11] += 1
            if predict_label[i] == 11 or predict_label[i] == 1:
                recall_number[11] += 1

    recall_ratio = []
    for key, value in recall_number.items():
        recall_ratio.append(value / category_number[key])

    recall=np.mean(recall_ratio)

    return accuracy,recall

