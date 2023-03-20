import re

def get_category(file_name):
    return re.findall(r"[0-9|X][0-9]+",file_name)[0]


