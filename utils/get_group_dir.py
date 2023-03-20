import os


def get_group_dir(category_name1,category_name2,*position_list):
    category_group_name = category_name1 + '_' + category_name2

    for single_position in position_list:
        if category_group_name not in os.listdir(single_position):
            os.mkdir(single_position + category_group_name)

    return category_group_name