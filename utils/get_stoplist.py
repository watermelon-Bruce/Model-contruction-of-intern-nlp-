import os
from data_process.utils.get_encode import get_encoding
def create_stoplist(stop_words_road):
    stoplists= set()
    for dirpath, dirname, filenames in os.walk(stop_words_road):  # 遍历当前文件夹下所有的停用词表
        for filename in filenames:
            encode = get_encoding(stop_words_road + str(filename))
            f = open(stop_words_road + str(filename), "r", encoding=encode, errors='ignore')
            for word in f.readlines():
                word = word.replace('\n', '')
                stoplists.add(word)  # 取并集
    return stoplists