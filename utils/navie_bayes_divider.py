from sklearn.naive_bayes import GaussianNB
from utils.get_category_list_by_industry import get_category_list_by_insudtry
from utils.get_group_dir import get_group_dir
from utils.pickle_operation import pickle_store,pickle_read
from utils.re_sampling import re_sampling
from utils.consin_similarity_average import consin_similarity_average
from utils.model_read import model_read
from utils.get_accuracy_and_recall import get_accuracy_and_recall
import os




class navie_bayes_divider(object):
    def __init__(self,doc_vec_train,navie_bayes_model_by_industry_dir_road,doc_vec_validation,doc_vec,doc_vec_kernel,predict_result_road):
        self.doc_vec_train=doc_vec_train
        self.navie_bayes_model_by_industry_dir_road=navie_bayes_model_by_industry_dir_road
        self.doc_vec_validation=doc_vec_validation
        self.doc_vec_kernel=doc_vec_kernel
        self.doc_vec=doc_vec
        self.predict_result_road=predict_result_road

    def navie_bayes_model_train(self):
        category_list_by_industry = get_category_list_by_insudtry()

        # get every two categories to build dichotomy divider
        for i in range(len(category_list_by_industry)):
            for j in range(i + 1, len(category_list_by_industry)):
                category_name1 = category_list_by_industry[i]
                category_name2 = category_list_by_industry[j]
                if category_name1+"_"+category_name2 in os.listdir(self.navie_bayes_model_by_industry_dir_road):
                    continue
                category_group_name = get_group_dir(category_name1,category_name2,self.navie_bayes_model_by_industry_dir_road)
                matrix1=pickle_read(self.doc_vec_train+category_name1+".pckl")
                label1=[int(category_name1) for _ in range(len(matrix1))]
                matrix2 = pickle_read(self.doc_vec_train + category_name2 + ".pckl")
                label2 = [int(category_name2) for _ in range(len(matrix2))]
                print(category_name1+"的样本数量是"+str(len(matrix1))+"\n"+
                               category_name2+"的样本数量是"+str(len(matrix2)))
                if len(matrix1)>len(matrix2):
                    print("是否对",category_name1,"进行负采样，或者对",category_name2,"进行过采样？")
                else:
                    print("是否对", category_name2, "进行负采样，或者对", category_name1, "进行过采样？")
                flag=int(input("若两者都不进行，请输入0\n若只进行过采样，请输入1\n若只进行负采样，请输入2\n若两者都进行，请输入3\n"))

                matrix,label=re_sampling(flag,matrix1,label1,matrix2,label2)



                gnb = GaussianNB()

                gnb.fit(
                    matrix,label
                )


                pickle_store(gnb,self.navie_bayes_model_by_industry_dir_road+category_group_name+"/gnb_model.pckl")


                matrix_validation1=pickle_read(self.doc_vec_validation+category_name1+".pckl")
                label_validation1=[int(category_name1) for _ in range(len(matrix_validation1))]
                matrix_validation2 = pickle_read(self.doc_vec_validation + category_name2 + ".pckl")
                label_validation2 = [int(category_name2) for _ in range(len(matrix_validation2))]
                for m in range(len(label_validation2)):
                    matrix_validation1.append(matrix_validation2[m])
                    label_validation1.append(label_validation2[m])
                matrix_validaion = matrix_validation1
                label_validaion = label_validation1
                acc_score = gnb.score(matrix_validaion, label_validaion)
                print("当前二分类器的验证集上的准确率为",acc_score)

    def navie_bayes_predict(self):
        print("正在计算多分类器的预测结果")
        if 'navie_bayes_predict_result.pckl' not in os.listdir(self.predict_result_road):
            category_list_by_industry = get_category_list_by_insudtry()
            original_label=[]
            predict_label=[]
            predict_result={}
            model_dict=model_read(self.navie_bayes_model_by_industry_dir_road)

            for category_by_industry in category_list_by_industry:
                try:
                    doc_veclist=pickle_read(self.doc_vec+category_by_industry+"doc_vec.pckl")
                except:
                    doc_veclist=pickle_read(self.doc_vec+category_by_industry+"_doc_tfidf.pckl")
                print("正在遍历",category_by_industry,"下的所有docvec")
                number=0
                for docvec in doc_veclist:
                    original_label.append(int(category_by_industry))

                    similarity_dict={}

                    for i in category_list_by_industry:
                        similarity_dict[i]=consin_similarity_average(docvec,self.doc_vec_kernel,i)


                    kernel_category_list=[]
                    Large_6th = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
                    Large_6th = Large_6th[:6]

                    for single_set in Large_6th:
                        kernel_category_list.append(single_set[0])

                    #get the prediction result by voting
                    predict_to_be_chosed=[]
                    for m in range(len(kernel_category_list)):
                        for n in range(m+1,len(kernel_category_list)):
                            if int(kernel_category_list[m])<int(kernel_category_list[n]):
                                navie_bayes_model=model_dict[kernel_category_list[m]+"_"+kernel_category_list[n]]
                                predict_to_be_chosed.append(navie_bayes_model.predict([docvec]))
                            else:
                                navie_bayes_model = model_dict[kernel_category_list[n]+"_"+kernel_category_list[m]]
                                predict_to_be_chosed.append(navie_bayes_model.predict([docvec]))


                    predict_label.append(max(predict_to_be_chosed,key=predict_to_be_chosed.count)[0])
                    number+=1
                    if number%200==0:
                        print(category_by_industry,"已遍历",str(number)+"/",len(doc_veclist))

            predict_result['original_label']=original_label
            predict_result['predict_label']=predict_label
            pickle_store(predict_result,self.predict_result_road+"navie_bayes_predict_result.pckl")
        else:
            predict_result=pickle_read(self.predict_result_road+"navie_bayes_predict_result.pckl")

        original_label = predict_result['original_label']
        predict_label = predict_result['predict_label']
        accuracy,recall=get_accuracy_and_recall(original_label,predict_label)
        print("准确率为",accuracy)
        print("召回率为",recall)

