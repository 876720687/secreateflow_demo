import os

import spu
import secretflow as sf
from pathlib import Path
import sys

alice, bob, carol = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('carol')

cluster_def={
    'nodes': [
        {
            'party': 'alice',
            'id': '0',
            # Use the address and port of alice instead.
            # Please choose a unused port.
            'address': '192.168.200.203:9395',
        },
        # {
        #     'party': 'bob',
        #     'id': '1',
        #     # Use the address and port of alice instead.
        #     # Please choose a unused port.
        #     'address': '192.168.200.203:9396',
        # },
        {
            'party': 'carol',
            'id': '2',
            # Use the ip and port of bob instead.
            # Please choose a unused port.
            'address': '192.168.200.205:9397',
        },
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    }
}


spu = sf.SPU(cluster_def=cluster_def)


# ----------------- 单多键隐私求交 -------------------
input_path = {alice: './data/alice.csv', carol: './data/carol.csv'}
output_path = {alice: './data/alice_psi.csv', carol: './data/carol_psi.csv'}

# 单机这样部署
#spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob'])


#集群初始化实例
#spu = sf.SPU(parties=['alice', 'carol'])

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 合并方式为uid
spu.psi_csv(key='uid', input_path=input_path, output_path=output_path, receiver='alice')
# 合并方式为uid, month
spu.psi_csv(key=['uid', 'month'], input_path=input_path, output_path=output_path, receiver='alice')


