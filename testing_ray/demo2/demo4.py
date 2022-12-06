# -*- coding: utf-8 -*- 
# @Time : 2022/11/25 14:46 
# @Author : YeMeng 
# @File : demo5.py 
# @contact: 876720687@qq.com
import ray
import time
import numpy as np
import random
import string
from tqdm import tqdm

ray.init(address='auto')
# ray.init(num_cpus = 4)
# # When connecting to an existing cluster, num_cpus and num_gpus must not be provided.

def tiny_work(array, n):
    res = -1
    for i in range(len(array)):
        if array[i] == n:
            res = i
            break
    return res

@ray.remote
def mega_work(array, n, start, end):
    return [tiny_work(array, n) for x in range(start, end)]

array = np.random.randint(0,1000,size=100)
array = list(array)
start = time.time()
result_ids = []
[result_ids.append(mega_work.remote(array,random.randint(0,1000), x*1000, (x+1)*1000)) for x in range(100)]

results = ray.get(result_ids)

print("duration = " + str(time.time() - start))