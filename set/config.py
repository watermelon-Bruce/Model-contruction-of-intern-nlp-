import yaml
class path_get(object):
    def __init__(self,yaml_path):
        self.yaml_obj = open(yaml_path, encoding="utf-8")  # 传入配置文件的地址
        self.config_cache = yaml.load(self.yaml_obj, Loader=yaml.FullLoader)  # 加载配置文件信息

    def get_ununion_dir_road(self):
        return self.config_cache.get('chen_phrase_set_final'),self.config_cache.get('yang_phrase_set_final'),self.config_cache.get('total_phrase_set_final')

    def get_total_content_by_kind_dir_road(self):
        return self.config_cache.get('total_content_by_kind_dir_road')

    def get_total_content_by_industry_dir_road(self):
        return self.config_cache.get('total_content_by_industry_dir_road')

    def get_total_seglist_by_kind_dir_road(self):
        return self.config_cache.get('total_seglist_by_kind_dir_road')

    def get_total_seglist_by_industry_dir_road(self):
        return self.config_cache.get('total_seglist_by_industry_dir_road')

    def get_word_vec_model(self):
        return self.config_cache.get('word_vec_model')

    def get_doc_vec(self):
        return self.config_cache.get('doc_vec')

    def get_doc_tfidf(self):
        return self.config_cache.get('doc_tfidf')

    def get_stop_words_road(self):
        return self.config_cache.get('stop_words_road')

    def get_doc_vec_train(self):
        return self.config_cache.get('doc_vec_train')

    def get_doc_tfidf_train(self):
        return self.config_cache.get('doc_tfidf_train')

    def get_doc_vec_validation(self):
        return self.config_cache.get('doc_vec_validation')

    def get_doc_tfidf_validation(self):
        return self.config_cache.get('doc_tfidf_validation')

    def get_doc_vec_kernel(self):
        return self.config_cache.get('doc_vec_kernel')

    def get_doc_tfidf_kernel(self):
        return self.config_cache.get('doc_tfidf_kernel')

    def get_navie_bayes_model_by_industry_dir_road(self):
        return self.config_cache.get('navie_bayes_model_by_industry_dir_road')

    def get_navie_bayes_model_by_kind_dir_road(self):
        return self.config_cache.get('navie_bayes_model_by_kind_dir_road')

    def get_navie_bayes_model_tfidf_by_industry_dir_road(self):
        return self.config_cache.get('navie_bayes_model_tfidf_by_industry_dir_road')

    def get_navie_bayes_model_tfidf_by_kind_dir_road(self):
        return self.config_cache.get('navie_bayes_model_tfidf_by_kind_dir_road')

    def get_svm_model_by_industry_dir_road(self):
        return self.config_cache.get('svm_model_by_industry_dir_road')

    def get_svm_model_by_kind_dir_road(self):
        return self.config_cache.get('svm_model_by_kind_dir_road')

    def get_svm_model_tfidf_by_industry_dir_road(self):
        return self.config_cache.get('svm_model_tfidf_by_industry_dir_road')

    def get_svm_model_tfidf_by_kind_dir_road(self):
        return self.config_cache.get('svm_model_tfidf_by_kind_dir_road')

    def get_predict_result_road_vec(self):
        return self.config_cache.get('predict_result_road_vec')

    def get_predict_result_road_tfidf(self):
        return self.config_cache.get('predict_result_road_tfidf')
















