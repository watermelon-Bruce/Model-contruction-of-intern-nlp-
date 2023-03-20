from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from utils.pickle_operation import pickle_read,pickle_store
from utils.get_category_list_by_industry import get_category_list_by_insudtry
import jieba


'''

Calculate the tfidf value of a document to measure the importance of the document
Perform pca dimensionality reduction on the document vector to improve the efficiency of  model construction

Store the results using pickle files
'''

def get_total_word_set(total_phrase_set_final):
    category_list_by_industry=get_category_list_by_insudtry()
    total_word_set=set()
    for category_by_industry in category_list_by_industry:
        phrase_set=pickle_read(total_phrase_set_final+category_by_industry+".pickle")
        total_word_set=total_word_set.union(phrase_set)
        for phrase in phrase_set:
            for word in jieba.cut(phrase):
                total_word_set.add(word)
    return total_word_set

def get_doc_tfidf(total_phrase_set_final,total_seglist_by_industry_dir_road,doc_tfidf):
    print("正在生成tfidf矩阵")
    total_word_set=get_total_word_set(total_phrase_set_final)
    category_list_by_industry=get_category_list_by_insudtry()
    corpous = []
    cutposition=[]
    cutposition.append(0)
    for category_by_industry in category_list_by_industry:
        seglist_file=pickle_read(total_seglist_by_industry_dir_road+"seglist_"+category_by_industry+".pckl")
        position=cutposition[-1]+len(seglist_file)
        cutposition.append(position)
        for sentence in seglist_file:
            text=' '
            for word in sentence.split():
                if len(word)!=1:
                    text+=word
                    text+=" "
            corpous.append(text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpous)
    word_list=vectorizer.get_feature_names_out()


    print("正在加权关键词")
    for word_id in range(len(word_list)):
        if word_list[word_id] in total_word_set:
            new_column=tfidf_matrix[:,word_id]*5
            tfidf_matrix[:,word_id]=new_column
        if word_id%1000==0:
            print("已遍历关键词",word_id,"条")

    print("正在进行pca降维")
    svd = TruncatedSVD(n_components=2000)
    newX = svd.fit_transform(tfidf_matrix)
    matrix=newX
    for id in range(len(category_list_by_industry)):
        matrix_cut=matrix[cutposition[id]:cutposition[id+1]]
        pickle_store(matrix_cut,doc_tfidf+category_list_by_industry[id]+"_doc_tfidf.pckl")



























