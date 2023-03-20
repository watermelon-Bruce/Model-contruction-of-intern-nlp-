import jieba
from utils.get_category_list_by_industry import get_category_list_by_insudtry
from utils.pickle_operation import pickle_store,pickle_read
from utils.get_stoplist import create_stoplist


def word_cut(stop_words_road,total_phrase_set_final,total_content_by_industry_dir_road,total_seglist_by_industry_dir_road):
    print("正在进行分词")

    stoplist=create_stoplist(stop_words_road)


    category_list_by_industry=get_category_list_by_insudtry()
    for category_by_industry in category_list_by_industry:
        #获取当前的关键短语，并加入到词典
        print("正在处理类",category_by_industry)
        total_phrase_set=pickle_read(total_phrase_set_final+category_by_industry+".pickle")
        for phrase in total_phrase_set:
            jieba.add_word(phrase)
        #读入当前类别的content，并生成对应的分词结果seglist
        total_content_file=pickle_read(total_content_by_industry_dir_road+"content_"+category_by_industry+".pckl")
        seglist_file=[]
        #遍历当前类别的每一个content
        number=0#计数器
        for content in total_content_file:
            seg=jieba.cut(content)
            sentence=''
            for i in seg:
                if i not in stoplist:
                    sentence += i
                    sentence += " "
            seglist_file.append(sentence)
            number+=1
            if number%500==0:
                print('已处理',number,'篇')
        pickle_store(seglist_file,total_seglist_by_industry_dir_road+"seglist_"+category_by_industry+".pckl")

