from utils.pickle_operation import pickle_read,pickle_store
from utils.consin_similarity import cosine_similarity
from utils.get_category_list_by_industry import get_category_list_by_insudtry
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

'''
user the kmeans to find doc emebdding with strong features

'''

def to_cluster(doc_vector, category_by_insudtry, thresh_of_similarity, max_number_of_center):
    print("正在筛选类",category_by_insudtry,'的中心文档')
    try:
        try:
            doc_vector=pickle_read(doc_vector + category_by_insudtry + "doc_vec.pckl")
        except:
            doc_vector = pickle_read(doc_vector + category_by_insudtry + "_doc_tfidf.pckl")
    except:
        doc_vector=pickle_read(doc_vector + category_by_insudtry + ".pckl")
    #map of id_vector
    doc_vector_and_id={}
    for i in range(len(doc_vector)):
        doc_vector_and_id[i]=doc_vector[i]


    doc_id = []
    for key, value in doc_vector_and_id.items():
        doc_id.append(key)

    distortions = []
    total_kernel_doc_id = []


    for i in range(1, max_number_of_center):
        number = 0
        print('质心个数',i)
        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, tol=1e-4, random_state=0)
        km.fit(doc_vector)
        center_vector_list = km.cluster_centers_  # the centre of every cluster
        for center_vector in center_vector_list:
            classdict = {}
            for j in range(len(doc_id)):
                classdict[doc_id[j]] = cosine_similarity(np.array(center_vector), np.array(doc_vector[j]))
            a1 = sorted(classdict.items(), key=lambda x: x[1], reverse=True)

            kernel_doc_id = []
            for sets in a1:
                if sets[1] >= thresh_of_similarity:
                    kernel_doc_id.append(sets[0])



            print("当前聚类中心下的中心文档个数为",len(kernel_doc_id),"/",len(doc_id))
            number+=len(kernel_doc_id)
            total_kernel_doc_id.append(kernel_doc_id)
            print('')

        print("质心个数为",i,"thresh_of_similarity为",thresh_of_similarity,"时，训练集个数为",number,"/",len(doc_id))
        distortions.append(km.inertia_)

    plt.plot(range(1, max_number_of_center), distortions)
    plt.show()
    return total_kernel_doc_id,doc_vector_and_id


def get_train_validation(doc_vector, doc_vec_train, doc_vec_validation):
    category_list_by_insudtry=get_category_list_by_insudtry()
    category_id_by_insudtry=0
    while category_id_by_insudtry<len(category_list_by_insudtry):
        print("当前类别为",category_list_by_insudtry[category_id_by_insudtry])
        thresh_of_similarity=float(input("请输入thresh_of_similarity \n"))
        max_number_of_center=int(input("请输入max_number_of_center \n"))+1
        total_kernel_doc_id,doc_vector_and_id=to_cluster(doc_vector, category_list_by_insudtry[category_id_by_insudtry], thresh_of_similarity,max_number_of_center)
        flag=input("是否修改thresh_of_similarity或者max_number_of_center？y/n \n")
        if flag=='y':
            pass
        else:
            doc_vec_train_matrix=[]
            doc_vec_validation_matrix=[]

            kernel_doc_id=set()
            for i in total_kernel_doc_id:
                for j in i:
                    kernel_doc_id.add(j)

            for key,value in doc_vector_and_id.items():
                if key in kernel_doc_id:
                    doc_vec_train_matrix.append(value)
                else:
                    doc_vec_validation_matrix.append(value)

            pickle_store(doc_vec_train_matrix,doc_vec_train+category_list_by_insudtry[category_id_by_insudtry]+".pckl")
            pickle_store(doc_vec_validation_matrix,doc_vec_validation+category_list_by_insudtry[category_id_by_insudtry]+".pckl")
            category_id_by_insudtry+=1




