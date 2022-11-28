# -*- coding: utf-8 -*- 
# @Time : 2022/11/24 16:05 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com
import sys
import ray
import time
import numpy as np
import random
import string
from tqdm import tqdm

def search(array, n):
    res = -1
    for i in range(len(array)):
        if array[i] == n:
            res = i
            break
    return res

if __name__ == '__main__':
    array = np.random.randint(0,1000,size=100)
    array = list(array)
    start = time.time()
    results = [search(array, random.randint(0,1000)) for x in range(int(sys.argv[1]))]
    print("duration = "+ str(str(time.time() - start)))