# -*- coding: utf-8 -*- 
# @Time : 2022/12/5 16:59 
# @Author : YeMeng 
# @File : demo1_single.py 
# @contact: 876720687@qq.com

# TODO：AttributeError: 'NoneType' object has no attribute 'XGDMatrixFree'
# success run in debug.

import secretflow as sf
# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], num_cpus=16, log_to_driver=True)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

import xgboost as xgb
import pandas as pd
from secretflow.utils.simulation.datasets import dataset

df = pd.read_csv(dataset('dermatology'))
df.fillna(value=0)
print(df.dtypes)
y = df['class']
y = y - 1
x = df.drop(columns="class")
dtrain = xgb.DMatrix(x,y)
dtest = dtrain
params = {
            'max_depth': 4,
            'objective': 'multi:softmax',
            'min_child_weight': 1,
            'max_bin': 10,
            'num_class': 6,
            'eval_metric': 'merror',
        }
num_round = 4
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=2)

from secretflow.data.horizontal import read_csv
from secretflow.security.aggregation import SecureAggregator
from secretflow.security.compare import SPUComparator
from secretflow.utils.simulation.datasets import load_dermatology

aggr = SecureAggregator(charlie, [alice, bob])
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))
comp = SPUComparator(spu)
data = load_dermatology(parts=[alice, bob], aggregator=aggr,
                        comparator=comp)
data.fillna(value=0, inplace=True)

params = {# XGBoost parameter tutorial
         # https://xgboost.readthedocs.io/en/latest/parameter.html
         'max_depth': 4, # max depth
         'eta': 0.3, # learning rate
         'objective': 'multi:softmax', # objection function，support "binary:logistic","reg:logistic","multi:softmax","multi:softprob","reg:squarederror"
         'min_child_weight': 1, # The minimum value of weight
         'lambda': 0.1, # L2 regularization term on weights (xgb's lambda)
         'alpha': 0, # L1 regularization term on weights (xgb's alpha)
         'max_bin': 10, # Max num of binning
         'num_class':6, # Only required in multi-class classification
         'gamma': 0, # Same to min_impurity_split,The minimux gain for a split
         'subsample': 1.0, # Subsample rate by rows
         'colsample_bytree': 1.0, # Feature selection rate by tree
         'colsample_bylevel': 1.0, # Feature selection rate by level
         'eval_metric': 'merror',  # supported eval metric：
                                    # 1. rmse
                                    # 2. rmsle
                                    # 3. mape
                                    # 4. logloss
                                    # 5. error
                                    # 6. error@t
                                    # 7. merror
                                    # 8. mlogloss
                                    # 9. auc
                                    # 10. aucpr
         # Special params in SFXgboost
         # Required
         'hess_key': 'hess', # Required, Mark hess columns, optionally choosing a column name that is not in the data set
         'grad_key': 'grad', # Required，Mark grad columns, optionally choosing a column name that is not in the data set
         'label_key': 'class', # Required，ark label columns, optionally choosing a column name that is not in the data set
}




from secretflow.ml.boost.homo_boost import SFXgboost

bst = SFXgboost(server=charlie, clients=[alice, bob])


bst.train(data, data, params=params, num_boost_round = 6)

