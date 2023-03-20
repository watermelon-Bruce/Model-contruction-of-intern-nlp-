import os
from utils.open_file import open_txt_file,open_csv_file
from utils.get_category import get_category
from utils.pickle_operation import pickle_store

def get_union(chen_phrase_set_final, yang_phrase_set_final, total_phrase_set_final):

    chen_final_file_list=os.listdir(chen_phrase_set_final)#所有文件名

    for chen_final_file in chen_final_file_list:
        try:
            df=open_csv_file(chen_phrase_set_final + chen_final_file)
            chen_set=set(df['word'])
            category=get_category(chen_final_file)#识别出当前类别
            yang_set=set(open_txt_file(yang_phrase_set_final + str(category) + '.docx.txt').split())
            final_set=chen_set.union(yang_set)#取并集
            pickle_store(final_set, total_phrase_set_final + str(category) + '.pickle')
        except:
            pass










