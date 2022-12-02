# -*- coding: utf-8 -*- 
# @Time : 2022/11/30 13:58 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com
#main.py
import os
import numpy as np
import torch
device = torch.device("cuda")
data_dir = '/tcdata'
a = np.load(os.path(data_dir,a.npy))
b = np.load(os.path(data_dir,b.npy))
a = torch.from_numpy(a).to(device)
b = torch.from_numpy(b).to(device)
c = torch.matmul(a,b).cpu()
print(c)
np.save("result.npy", c)