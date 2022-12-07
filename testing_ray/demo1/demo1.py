# -*- coding: utf-8 -*- 
# @Time : 2022/11/19 11:38 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com
import sys
import time
import logging
from absl import app
import spu
import secretflow as sf

# init log
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# SPU settings
cluster_def = {
    'nodes': [
        # <<< !!! >>> replace <192.168.0.1:12945> to alice node's local ip & free port
        {'party': 'alice', 'id': 'local:0', 'address': '192.168.200.203:9394', 'listen_address': '0.0.0.0:12945'},
        # <<< !!! >>> replace <192.168.0.2:12946> to bob node's local ip & free port
        {'party': 'bob', 'id': 'local:1', 'address': '192.168.200.204:9394', 'listen_address': '0.0.0.0:12946'},
        # <<< !!! >>> if you need 3pc test, please add node here, for example, add carol as rank 2
        # {'party': 'carol', 'id': 'local:2', 'address': '192.168.200.205:9394', 'listen_address': '0.0.0.0:12947'},
        # {'party': 'carol', 'id': 'local:2', 'address': '127.0.0.1:12347'},
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
    },
}


def main(_):

    # sf init
    # <<< !!! >>> replace <192.168.0.1:9394> to your ray head
    sf.init(address='192.168.200.203:9394')
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    # carol = sf.PYU('carol')

    # <<< !!! >>> replace path to real parties local file path.
    input_path = {
        alice: '/home/almalinux/sf-benchmark/data/psi_1.csv',
        bob: '/home/almalinux/sf-benchmark/data/psi_2.csv',
        # if run with `ECDH_PSI_3PC`, add carol
        # carol: '/home/almalinux/sf-benchmark/psi_3.csv',
    }
    output_path = {
        alice: '/home/almalinux/sf-benchmark/data/psi_output.csv',
        bob: '/home/almalinux/sf-benchmark/data/psi_output.csv',
        # if run with `ECDH_PSI_3PC`, add carol
        # carol: '/home/almalinux/sf-benchmark/psi_output.csv',
    }
    select_keys = {
        alice: ['id'],
        bob: ['id'],
        # if run with `ECDH_PSI_3PC`, add carol
        # carol: ['id'],
    }
    spu = sf.SPU(cluster_def)

    # prepare data
    start = time.time()

    reports = spu.psi_csv(
        key=select_keys,
        input_path=input_path,
        output_path=output_path,
        receiver='alice',  # if `broadcast_result=False`, only receiver can get output file.
        protocol='KKRT_PSI_2PC',	# psi protocol
        precheck_input=False,  # will cost ext time if set True
        sort=False,  # will cost ext time if set True
        broadcast_result=False,  # will cost ext time if set True
    )
    print(f"psi reports: {reports}")
    logging.info(f"cost time: {time.time() - start}")

    sf.shutdown()


if __name__ == '__main__':
    app.run(main)