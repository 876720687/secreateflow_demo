# -*- coding: utf-8 -*- 
# @Time : 2022/12/6 14:24 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com
import secretflow as sf

sf.shutdown()
sf.init(['alice', 'bob'], num_cpus=8, log_to_driver=True)
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

import pandas as pd

# 当pd.dataframe过宽时，Sphinx目前的主题无法正确显示html格式输出，
# 如果自行运行notebook时可以忽略。
pd.set_option('display.notebook_repr_htm', False)

# secretflow.utils.simulation.datasets contains mirrors of some popular open dataset.
from secretflow.utils.simulation.datasets import dataset

df = pd.read_csv(dataset('bank_marketing_full'), sep=';')
df['uid'] = df.index + 1

import numpy as np

df_alice = df.iloc[:, np.r_[0:8, -1]].sample(frac=0.9)

df_bob = df.iloc[:, 8:].sample(frac=0.9)

import tempfile

_, alice_path = tempfile.mkstemp()
_, bob_path = tempfile.mkstemp()
df_alice.reset_index(drop=True).to_csv(alice_path, index=False)
df_bob.reset_index(drop=True).to_csv(bob_path, index=False)


_, alice_psi_path = tempfile.mkstemp()
_, bob_psi_path = tempfile.mkstemp()

spu.psi_csv(
    key="uid",
    input_path={alice: alice_path, bob: bob_path},
    output_path={alice: alice_psi_path, bob: bob_psi_path},
    receiver="alice",
    protocol="ECDH_PSI_2PC",
    sort=True,
)

# 方法2
from secretflow.data.vertical import read_csv as v_read_csv

vdf = v_read_csv(
    {alice: alice_path, bob: bob_path},
    spu=spu,
    keys="uid",
    drop_keys="uid",
    psi_protocl="ECDH_PSI_2PC",
)




from secretflow.stats.table_statistics import table_statistics

pd.set_option('display.max_rows', None)
data_stats = table_statistics(vdf)

pd.reset_option('display.max_rows')


vdf['education'] = vdf['education'].replace(
    {'tertiary': 3, 'secondary': 2, 'primary': 1, 'unknown': np.NaN}
)

vdf['default'] = vdf['default'].replace({'no': 0, 'yes': 1, 'unknown': np.NaN})

vdf['housing'] = vdf['housing'].replace({'no': 0, 'yes': 1, 'unknown': np.NaN})

vdf['loan'] = vdf['loan'].replace({'no': 0, 'yes': 1, 'unknown': np.NaN})

vdf['month'] = vdf['month'].replace(
    {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12,
    }
)

vdf['y'] = vdf['y'].replace(
    {
        'no': 0,
        'yes': 1,
    }
)

# print(sf.reveal(vdf.partitions[alice].data))
# print(sf.reveal(vdf.partitions[bob].data))


vdf["education"] = vdf["education"].fillna(vdf["education"].mode())
vdf["default"] = vdf["default"].fillna(vdf["default"].mode())
vdf["housing"] = vdf["housing"].fillna(vdf["housing"].mode())
vdf["loan"] = vdf["loan"].fillna(vdf["loan"].mode())

# print(sf.reveal(vdf.partitions[alice].data))
# print(sf.reveal(vdf.partitions[bob].data))


from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.preprocessing.binning.vert_woe_substitution import VertWOESubstitution

binning = VertWoeBinning(spu)
woe_rules = binning.binning(
    vdf,
    binning_method="chimerge",
    bin_num=4,
    bin_names={alice: [], bob: ["duration"]},
    label_name="y",
)

woe_sub = VertWOESubstitution()
vdf = woe_sub.substitution(vdf, woe_rules)

# print(sf.reveal(vdf.partitions[alice].data))
# print(sf.reveal(vdf.partitions[bob].data))


from secretflow.preprocessing.encoder import OneHotEncoder

encoder = OneHotEncoder()
# for vif and correlation only
vdf_hat = vdf.drop(columns=["job", "marital", "contact", "month", "day", "poutcome"])

tranformed_df = encoder.fit_transform(vdf['job'])
vdf[tranformed_df.dtypes.index] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['marital'])
vdf[tranformed_df.dtypes.index] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['contact'])
vdf[tranformed_df.dtypes.index] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['month'])
vdf[tranformed_df.dtypes.index] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['day'])
vdf[tranformed_df.dtypes.index] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['poutcome'])
vdf[tranformed_df.dtypes.index] = tranformed_df

vdf = vdf.drop(columns=["job", "marital", "contact", "month", "day", "poutcome"])

# print(sf.reveal(vdf.partitions[alice].data))
# print(sf.reveal(vdf.partitions[bob].data))




from secretflow.preprocessing import StandardScaler

X = vdf.drop(columns=['y'])
y = vdf['y']
scaler = StandardScaler()
X = scaler.fit_transform(X)
vdf[X.columns] = X
# print(sf.reveal(vdf.partitions[alice].data))
# print(sf.reveal(vdf.partitions[bob].data))



# ----------------- 数据分析 ---------------
from secretflow.stats.table_statistics import table_statistics

pd.set_option('display.max_rows', None)
data_stats = table_statistics(vdf)
pd.reset_option('display.max_rows')

# 相关系数矩阵
from secretflow.stats.ss_pearsonr_v import PearsonR

pearson_r_calculator = PearsonR(spu)
corr_matrix = pearson_r_calculator.pearsonr(vdf_hat)

import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# corr_matrix


from secretflow.stats.ss_vif_v import VIF

vif_calculator = VIF(spu)
vif_results = vif_calculator.vif(vdf_hat)
# print(vdf_hat.columns)
# print(vif_results)




from secretflow.data.split import train_test_split

random_state = 1234

train_vdf, test_vdf = train_test_split(vdf, train_size=0.8, random_state=random_state)

train_x = train_vdf.drop(columns=['y'])
train_y = train_vdf['y']

test_x = test_vdf.drop(columns=['y'])
test_y = test_vdf['y']



stats_df = table_statistics(train_x['balance'])
min_val, max_val = stats_df['min'], stats_df['max']
from secretflow.stats import psi_eval
from secretflow.stats.core.utils import equal_range
import jax.numpy as jnp

split_points = equal_range(jnp.array([min_val, max_val]), 3)
balance_psi_score = psi_eval(train_x['balance'], test_x['balance'], split_points)

sf.reveal(balance_psi_score)



from secretflow.ml.linear.ss_sgd import SSRegression

lr_model = SSRegression(spu)
lr_model.fit(
    x=train_x,
    y=train_y,
    epochs=3,
    learning_rate=0.1,
    batch_size=1024,
    sig_type='t1',
    reg_type='logistic',
    penalty='l2',
    l2_norm=0.5,
)




from secretflow.ml.boost.ss_xgb_v import Xgb

xgb = Xgb(spu)
params = {
    'num_boost_round': 3,
    'max_depth': 5,
    'sketch_eps': 0.25,
    'objective': 'logistic',
    'reg_lambda': 0.2,
    'subsample': 1,
    'colsample_bytree': 1,
    'base_score': 0.5,
}
xgb_model = xgb.train(params=params, dtrain=train_x, label=train_y)



lr_y_hat = lr_model.predict(x=test_x, batch_size=1024, to_pyu=bob)

xgb_y_hat = xgb_model.predict(dtrain=test_x, to_pyu=bob)




# -------------- 模型评估 -----------------
from secretflow.stats.biclassification_eval import BiClassificationEval

biclassification_evaluator = BiClassificationEval(
    y_true=test_y, y_score=lr_y_hat, bucket_size=20
)
lr_report = sf.reveal(biclassification_evaluator.get_all_reports())


print(f'positive_samples: {lr_report.summary_report.positive_samples}')
print(f'negative_samples: {lr_report.summary_report.negative_samples}')
print(f'total_samples: {lr_report.summary_report.total_samples}')
print(f'auc: {lr_report.summary_report.auc}')
print(f'ks: {lr_report.summary_report.ks}')
print(f'f1_score: {lr_report.summary_report.f1_score}')


biclassification_evaluator = BiClassificationEval(
    y_true=test_y, y_score=xgb_y_hat, bucket_size=20
)
xgb_report = sf.reveal(biclassification_evaluator.get_all_reports())


print(f'positive_samples: {xgb_report.summary_report.positive_samples}')
print(f'negative_samples: {xgb_report.summary_report.negative_samples}')
print(f'total_samples: {xgb_report.summary_report.total_samples}')
print(f'auc: {xgb_report.summary_report.auc}')
print(f'ks: {xgb_report.summary_report.ks}')
print(f'f1_score: {xgb_report.summary_report.f1_score}')



from secretflow.stats import pva_eval

lr_pva_score = pva_eval(test_y, lr_y_hat, 1)

sf.reveal(lr_pva_score)



xgb_pva_score = pva_eval(test_y, xgb_y_hat, 1)

sf.reveal(xgb_pva_score)



from secretflow.stats import SSPValue

model = lr_model.save_model()
sspv = SSPValue(spu)
pvalues = sspv.pvalues(test_x, test_y, model)

# pvalues


from secretflow.stats import BiClassificationEval, ScoreCard

sc = ScoreCard(20, 600, 20)
score = sc.transform(xgb_y_hat)

sf.reveal(score.partitions[bob])


#  -------------- 清理临时文件 ---------------
# import os
#
# try:
#     os.remove(alice_path)
#     os.remove(alice_psi_path)
#     os.remove(bob_path)
#     os.remove(bob_psi_path)
# except OSError:
#     pass
#
# sf.shutdown()