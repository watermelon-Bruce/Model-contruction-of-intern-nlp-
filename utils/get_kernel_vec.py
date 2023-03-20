from utils.get_train_validation import to_cluster
from utils.pickle_operation import pickle_store
from utils.get_category_list_by_industry import get_category_list_by_insudtry

def get_kernel_vec(doc_vec_train,doc_vec_kernel):
    category_list_by_insudtry = get_category_list_by_insudtry()
    category_id_by_insudtry = 0
    while category_id_by_insudtry < len(category_list_by_insudtry):
        print("当前类别为", category_list_by_insudtry[category_id_by_insudtry])
        thresh_of_similarity = float(input("请输入thresh_of_similarity \n"))
        max_number_of_center = int(input("请输入max_number_of_center \n")) + 1
        total_kernel_doc_id, doc_vector_and_id = to_cluster(doc_vec_train,
                                                            category_list_by_insudtry[category_id_by_insudtry],
                                                            thresh_of_similarity, max_number_of_center)
        flag = input("是否修改thresh_of_similarity或者max_number_of_center？y/n \n")
        if flag == 'y':
            pass
        else:
            doc_vec_kernel_matrix = []


            kernel_doc_id = set()
            for i in total_kernel_doc_id:
                for j in i:
                    kernel_doc_id.add(j)

            for key, value in doc_vector_and_id.items():
                if key in kernel_doc_id:
                    doc_vec_kernel_matrix.append(value)

            pickle_store(doc_vec_kernel_matrix,
                         doc_vec_kernel + category_list_by_insudtry[category_id_by_insudtry] + ".pckl")
            category_id_by_insudtry += 1




