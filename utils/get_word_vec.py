from gensim.models import Word2Vec
from utils.pickle_operation import pickle_read
from utils.get_stoplist import create_stoplist
import os

'''
get the word_to_vec model from the library gensim
convert the word to vectors

store the well trained model
'''

def get_text(total_seglist_by_industry_dir_road,stop_words_road):
	print("正在获取所有语料")
	text=[]
	stoplist=create_stoplist(stop_words_road)
	seglist_file_list = os.listdir(total_seglist_by_industry_dir_road)
	for seglist_file in seglist_file_list:
		seglist = pickle_read(total_seglist_by_industry_dir_road + str(seglist_file))
		for sentence in seglist:
			sentence =sentence.split()
			text.append([word for word in sentence if word not in stoplist])
	return text



def get_word_vec(total_seglist_by_industry_dir_road,word_vec_model,stop_words_road):
	text=get_text(total_seglist_by_industry_dir_road,stop_words_road)
	print("正在生成word_vector")
	model=Word2Vec(sentences=text,window=8,epochs=50,vector_size=500,min_count=10,sg=0,cbow_mean=1,negative=5)
	model.save(word_vec_model+"word_vec_model.model")
