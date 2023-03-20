import os
from utils.get_category import get_category
from set.config import path_get

path_handler=path_get('../set/config.yaml')

def get_category_list_by_insudtry(total_content_by_industry_dir_road=path_handler.get_total_content_by_industry_dir_road()):
    content_name_list_by_industry=os.listdir(total_content_by_industry_dir_road)
    category_list_by_industry = []  # hold all the categories
    for name in content_name_list_by_industry:
        category_list_by_industry.append(get_category(name))

    return category_list_by_industry


