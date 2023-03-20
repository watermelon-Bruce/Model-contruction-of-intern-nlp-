from utils.consin_similarity import cosine_similarity
from utils.pickle_operation import pickle_read
import numpy as np



def consin_similarity_average(docvec,doc_vec_train,category_by_industry):
    doc_vec_train_matrix=pickle_read(doc_vec_train+category_by_industry+".pckl")
    doc_vec_train_matrix=np.array(doc_vec_train_matrix)
    docvec=np.array(docvec)
    length=len(doc_vec_train_matrix)
    doc_vec_sum=0
    for i in doc_vec_train_matrix:
        doc_vec_sum+=cosine_similarity(docvec,i)
    return doc_vec_sum/length#take the average of all the distances
