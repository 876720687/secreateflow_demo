# -*- coding: utf-8 -*- 
# @Time : 2022/12/5 11:15 
# @Author : YeMeng 
# @File : config.py 
# @contact: 876720687@qq.com
import spu
import secretflow as sf

Pseudo_distributed_DoubleCluster = sf.utils.testing.cluster_def(parties=['alice', 'bob'])

Pseudo_distributed_TripleCluster = sf.utils.testing.cluster_def(parties=['alice', 'bob', 'carol'])

Distributed_DoubelCluster = {
    'nodes': [
        {
            'party': 'alice',
            'id': '0',
            # Use the address and port of alice instead.
            # Please choose a unused port.
            'address': '192.168.200.203:9395',
        },
        {
            'party': 'bob',
            'id': '1',
            # Use the address and port of alice instead.
            # Please choose a unused port.
            'address': '192.168.200.203:9396',
        },
        # {
        #     'party': 'carol',
        #     'id': '2',
        #     # Use the ip and port of bob instead.
        #     # Please choose a unused port.
        #     'address': '192.168.200.205:9397',
        # },
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    }
}
