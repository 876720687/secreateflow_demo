# -*- coding: utf-8 -*- 
# @Time : 2022/11/2 19:39 
# @Author : YeMeng 
# @File : demo4.py 
# @contact: 876720687@qq.com
from sklearn.metrics import roc_auc_score

from demo_spu_neral.demo3_train import predict


def validate_model(params, X_test, y_test):
    y_pred = predict(params, X_test)
    return roc_auc_score(y_test, y_pred)