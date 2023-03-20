from utils.get_encode import get_encoding
import pandas as pd

def open_txt_file(file_road):
    encode = get_encoding(file_road)
    f = open(file_road, "r", encoding=encode,errors='ignore')
    content = f.read()
    return content

def open_csv_file(file_road):
    encode = get_encoding(file_road)
    df=pd.read_csv(file_road,encoding=encode)
    return df