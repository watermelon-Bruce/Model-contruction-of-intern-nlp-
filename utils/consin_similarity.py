import numpy as np

def cosine_similarity(x, y):  # calculate cosin similarity
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom==0:
        return 0
    else:
        return num / denom