from utils.pickle_operation import pickle_read
import os

def model_read(model_road):
    category_group_list=os.listdir(model_road)
    model_dict={}
    for category_group in category_group_list:
        model_name=os.listdir(model_road+category_group)[0]
        model_dict[category_group]=pickle_read(model_road+category_group+"/"+model_name)
    return model_dict

