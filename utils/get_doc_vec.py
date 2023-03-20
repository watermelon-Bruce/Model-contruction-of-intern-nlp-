import numpy as np
from gensim.models import Word2Vec
from utils.get_category_list_by_industry import get_category_list_by_insudtry
from utils.pickle_operation import pickle_store,pickle_read


'''
get doc embedding vector
'''

def get_doc_vec(word_vec_model,total_phrase_set_final,total_seglist_by_industry_dir_road,doc_vec):
    category_list_by_insudtry=get_category_list_by_insudtry()
    #key word set
    total_set=set()
    for category_by_industry in category_list_by_insudtry:
        word_set=pickle_read(total_phrase_set_final+category_by_industry+".pickle")
        total_set=total_set.union(word_set)

    #read the model which already trained
    model = Word2Vec.load(word_vec_model + "word_vec_model.model")
    keys = model.wv.index_to_key
    keys_index = model.wv.key_to_index
    wordvector = model.wv.vectors
    keys_set = set(keys)


    for category_by_industry in category_list_by_insudtry:
        print("正在生成类",category_by_industry,"的doc_vec")


        seglist=pickle_read(total_seglist_by_industry_dir_road+"seglist_"+category_by_industry+".pckl")

        word_vector_sum = []#get the key words embedding sum of every sentence
        word_vector_number = []#get the number of key words in every sentence
        doc_vector = []

        number = 0
        for seg in seglist:
            word_vector_sum.append(np.zeros(500))
            word_vector_number.append(0)
            doc_vector.append(np.zeros(500))
            for word in seg.split():#iterate every word in current doc
                #if word in total_set and word in keys_set:
                if word in keys_set:
                    word_vector_sum[-1]+=wordvector[keys_index[word]]
                    word_vector_number[-1] += 1
                    doc_vector[-1] = word_vector_sum[-1] / word_vector_number[-1]
            number+=1
            if number%500==0:
                print("已处理",number,"/",len(seglist))
        pickle_store(doc_vector,doc_vec+category_by_industry+"doc_vec.pckl")


















