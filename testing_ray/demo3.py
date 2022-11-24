# -*- coding: utf-8 -*- 
# @Time : 2022/11/24 16:06 
# @Author : YeMeng 
# @File : demo3.py 
# @contact: 876720687@qq.com
import ray
import time
import numpy as np
import random
import string
from tqdm import tqdm
import sys

@ray.remote
def search(array, n):
    res = -1
    for i in range(len(array)):
        if array[i] == n:
            res = i
            break
    return res


if __name__ == '__main__':
    ray.init(num_cpus=10, ignore_reinit_error=True)
    array = np.random.randint(0,1000,size=100)
    array = list(array)
    start = time.time()
    result_ids = [search.remote(array, random.randint(0,1000)) for x in range(int(sys.argv[1]))]
    results = ray.get(result_ids)
    print("duration = "+ str(time.time()-start))