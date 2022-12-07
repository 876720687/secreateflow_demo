import os
import spu
import secretflow as sf
from pathlib import Path
import sys

from utils.config import *

"""
实现完全分布式。
注意代码运行之前需要先启动ray集群
"""

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


alice, bob, carol = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('carol')



alice_input_path = str(ROOT) + '/data/alice.csv'
carol_input_path = str(ROOT) + '/data/carol.csv'
alice_output_path = str(ROOT) + '/data/alice_psi.csv'
carol_output_path = str(ROOT) + '/data/carol_psi.csv'
input_path = {alice: alice_input_path, carol: carol_input_path}
output_path = {alice: alice_output_path, carol: carol_output_path}


# 单多键隐私求交
spu = sf.SPU(cluster_def=Distributed_doubelCluster_)
# 合并方式为uid
spu.psi_csv(key='uid', input_path=input_path, output_path=output_path, receiver='alice')
# 合并方式为uid, month
spu.psi_csv(key=['uid', 'month'], input_path=input_path, output_path=output_path, receiver='alice')


