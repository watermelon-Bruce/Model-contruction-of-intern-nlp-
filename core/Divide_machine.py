from utils.get_union import get_union
from utils.word_cut import word_cut
from utils.get_doc_vec import get_doc_vec
from utils.get_doc_tfidf import get_doc_tfidf
from utils.get_word_vec import get_word_vec
from utils.get_train_validation import get_train_validation
from utils.get_kernel_vec import get_kernel_vec
from set.config import path_get
from utils.navie_bayes_divider import navie_bayes_divider
from utils.svm_divider import svm_divider



#this class is to do the data preprocess before training our model
class data_processor(object):
    #to get the union of the keywords list
    def To_union(self,chen_final,yang_final,total_final):
        get_union(chen_final,yang_final,total_final)

    #do the word cutting
    def To_word_cut(self,stop_words_road,total_phrase_set_final,total_content_by_industry_dir_road,total_seglist_by_industry_dir_road):
        word_cut(stop_words_road,total_phrase_set_final,total_content_by_industry_dir_road,total_seglist_by_industry_dir_road)

    #get the word embedding vectors
    def To_word_vec(self,total_seglist_by_industry_dir_road,word_vec_model,stop_words_road):
        get_word_vec(total_seglist_by_industry_dir_road,word_vec_model,stop_words_road)

    #get the doc embedding vectors
    def To_doc_vec(self,word_vec_model,total_phrase_set_final,total_seglist_by_industry_dir_road,doc_vector):
        get_doc_vec(word_vec_model,total_phrase_set_final,total_seglist_by_industry_dir_road,doc_vector)

    #get the doc tfidf vectors
    def To_doc_tfidf(self,total_phrase_set_final,total_seglist_by_industry_dir_road,doc_tfidf):
        get_doc_tfidf(total_phrase_set_final,total_seglist_by_industry_dir_road,doc_tfidf)

    #get train and validation set
    def To_train_validation(self,doc_vector, doc_vec_train, doc_vec_validation):
        get_train_validation(doc_vector, doc_vec_train, doc_vec_validation)

    #get training data with trong features
    def To_kernel_vec(self,doc_vec_train,doc_vec_kernel):
        get_kernel_vec(doc_vec_train,doc_vec_kernel)


if __name__=='__main__':
    my_processer=data_processor()
    path_handler=path_get('../set/config.yaml')#get all the path
    total_content_by_industry_dir_road=path_handler.get_total_content_by_industry_dir_road()
    total_seglist_by_industry_dir_road=path_handler.get_total_seglist_by_industry_dir_road()

    #read the path
    word_vec_model=path_handler.get_word_vec_model()
    stop_words_road=path_handler.get_stop_words_road()
    doc_vector=path_handler.get_doc_vec()
    doc_tfidf=path_handler.get_doc_tfidf()
    doc_vec_train=path_handler.get_doc_vec_train()
    doc_tfidf_train=path_handler.get_doc_tfidf_train()
    doc_vec_validation=path_handler.get_doc_vec_validation()
    doc_tfidf_validation=path_handler.get_doc_tfidf_validation()
    doc_vec_kernel=path_handler.get_doc_vec_kernel()
    doc_tfidf_kernel=path_handler.get_doc_tfidf_kernel()
    navie_bayes_model_by_industry_dir_road=path_handler.get_navie_bayes_model_by_industry_dir_road()
    navie_bayes_model_tfidf_by_industry_dir_road=path_handler.get_navie_bayes_model_tfidf_by_industry_dir_road()
    svm_model_by_industry_dir_road=path_handler.get_svm_model_by_industry_dir_road()
    svm_model_tfidf_by_industry_dir_road=path_handler.get_svm_model_tfidf_by_industry_dir_road()
    predict_result_road_vec=path_handler.get_predict_result_road_vec()
    predict_result_road_tfidf=path_handler.get_predict_result_road_tfidf()


    # to get the union of the keywords list
    chen_phrase_set_final,yang_phrase_set_final,total_phrase_set_final=path_handler.get_ununion_dir_road()
    my_processer.To_union(chen_phrase_set_final, yang_phrase_set_final, total_phrase_set_final)

    # do the word cutting
    my_processer.To_word_cut(
        stop_words_road=stop_words_road,
        total_phrase_set_final=total_phrase_set_final,
        total_content_by_industry_dir_road=total_content_by_industry_dir_road,
        total_seglist_by_industry_dir_road=total_seglist_by_industry_dir_road
    )


    #get the word embedding vectors
    my_processer.To_word_vec(
        word_vec_model=word_vec_model,
        total_seglist_by_industry_dir_road=total_seglist_by_industry_dir_road,
        stop_words_road=stop_words_road
    )


    #get the doc embedding vectors
    my_processer.To_doc_vec(
        word_vec_model=word_vec_model,
        total_phrase_set_final=total_phrase_set_final,
        total_seglist_by_industry_dir_road=total_seglist_by_industry_dir_road,
        doc_vector=doc_vector
    )


    #get the doc tfidf vectors
    my_processer.To_doc_tfidf(
        total_phrase_set_final=total_phrase_set_final,
        total_seglist_by_industry_dir_road=total_seglist_by_industry_dir_road,
        doc_tfidf=doc_tfidf
    )


    # get train and validation set of doc embedding vectors
    my_processer.To_train_validation(
        doc_vector=doc_vector,
        doc_vec_train=doc_vec_train,
        doc_vec_validation=doc_vec_validation
    )

    # get train and validation set of tfidf vectors
    my_processer.To_train_validation(
        doc_vector=doc_tfidf,
        doc_vec_train=doc_tfidf_train,
        doc_vec_validation=doc_tfidf_validation
    )

    # get training doc embedding data with trong features
    my_processer.To_kernel_vec(
        doc_vec_train=doc_vec_train,
        doc_vec_kernel=doc_vec_kernel
    )

    # get training doc tfidf data with trong features
    my_processer.To_kernel_vec(
        doc_vec_train=doc_tfidf_train,
        doc_vec_kernel=doc_tfidf_kernel
    )

    #create naive base divider for doc embedding
    my_navie_bayes_divider=navie_bayes_divider(
        doc_vec_train=doc_vec_train,
        navie_bayes_model_by_industry_dir_road=navie_bayes_model_by_industry_dir_road,
        doc_vec_validation=doc_vec_validation,
        doc_vec=doc_vector,
        doc_vec_kernel=doc_vec_kernel,
        predict_result_road=predict_result_road_vec
    )
    my_navie_bayes_divider.navie_bayes_model_train()
    my_navie_bayes_divider.navie_bayes_predict()

    # create naive base divider for doc tfidf
    my_navie_bayes_divider_tfidf=navie_bayes_divider(
        doc_vec_train=doc_tfidf_train,
        navie_bayes_model_by_industry_dir_road=navie_bayes_model_tfidf_by_industry_dir_road,
        doc_vec_validation=doc_tfidf_validation,
        doc_vec=doc_tfidf,
        doc_vec_kernel=doc_tfidf_kernel,
        predict_result_road=predict_result_road_tfidf
    )
    my_navie_bayes_divider_tfidf.navie_bayes_model_train()
    my_navie_bayes_divider_tfidf.navie_bayes_predict()


    # create svm divider for doc embedding
    my_svm_divider=svm_divider(
        doc_vec_train=doc_vec_train,
        svm_model_by_industry_dir_road=svm_model_by_industry_dir_road,
        doc_vec_validation=doc_vec_validation,
        doc_vec=doc_vector,
        doc_vec_kernel=doc_vec_kernel,
        predict_result_road=predict_result_road_vec
    )
    my_svm_divider.svm_model_train()
    my_svm_divider.svm_predict()

    # create svm divider for doc tfidf
    my_svm_divider_tfidf=svm_divider(
        doc_vec_train=doc_tfidf_train,
        svm_model_by_industry_dir_road=svm_model_tfidf_by_industry_dir_road,
        doc_vec_validation=doc_tfidf_validation,
        doc_vec=doc_tfidf,
        doc_vec_kernel=doc_tfidf_kernel,
        predict_result_road=predict_result_road_tfidf
    )
    my_svm_divider_tfidf.svm_model_train()
    my_svm_divider_tfidf.svm_predict()
























