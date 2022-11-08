import pandas as pd
import tempfile
import os
from sklearn.datasets import load_iris
import secretflow as sf
from secretflow.data.vertical import read_csv as v_read_csv

# sf.shutdown()

sf.init(['alice', 'bob', 'carol'])
alice, bob, carol = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('carol')


iris = load_iris(as_frame=True)
data = pd.concat([iris.data, iris.target], axis=1)

# 切换到你需要存放的数据路径当中
# temp_dir = tempfile.mkdtemp()
temp_dir = '/home/root-demo1/code/secretflow_demo/input'

# ------按照水平（特征相同）和垂直（样本相同）两种方式对数据进行切分 -------------

# Horizontal partitioning.
h_alice, h_bob, h_carol = data.iloc[:40, :], data.iloc[40:100, :], data.iloc[100:, :]
# Vertical partitioning.
v_alice, v_bob, v_carol = data.iloc[:, :2], data.iloc[:, 2:4], data.iloc[:, 4:]

# 保存数据
h_alice_path = os.path.join(temp_dir, 'h_alice.csv')
h_bob_path = os.path.join(temp_dir, 'h_bob.csv')
h_carol_path = os.path.join(temp_dir, 'h_carol.csv')
h_alice.to_csv(h_alice_path, index=False)
h_bob.to_csv(h_bob_path, index=False)
h_carol.to_csv(h_carol_path, index=False)

v_alice_path = os.path.join(temp_dir, 'v_alice.csv')
v_bob_path = os.path.join(temp_dir, 'v_bob.csv')
v_carol_path = os.path.join(temp_dir, 'v_carol.csv')
v_alice.to_csv(v_alice_path, index=False)
v_bob.to_csv(v_bob_path, index=False)
v_carol.to_csv(v_carol_path, index=False)

# ------------------------- 聚合 -------------------------
from secretflow.data.horizontal import read_csv as h_read_csv
from secretflow.security.aggregation import SecureAggregator
from secretflow.security.compare import SPUComparator

# The aggregator and comparator are respectively used to aggregate
# or compare data in subsequent data analysis operations.
aggr = SecureAggregator(device=alice, participants=[alice, bob, carol])

spu = sf.SPU(sf.utils.testing.cluster_def(parties=['alice', 'bob', 'carol']))
comp = SPUComparator(spu)
hdf = h_read_csv({alice: h_alice_path, bob: h_bob_path, carol: h_carol_path},
                 aggregator=aggr,
                 comparator=comp)


vdf = v_read_csv({alice: v_alice_path, bob: v_bob_path, carol: v_carol_path})


# ----------------- Clean up temporary files ------------

# import shutil
#
# shutil.rmtree(temp_dir, ignore_errors=True)