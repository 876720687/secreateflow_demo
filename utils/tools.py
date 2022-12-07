# -*- coding: utf-8 -*- 
# @Time : 2022/12/7 10:42 
# @Author : YeMeng 
# @File : tools.py 
# @contact: 876720687@qq.com

from functools import wraps
import secretflow as sf
from pathlib import Path
import sys


def aop(func):
    """aop func"""
    @wraps(func)
    def wrap(*args, **kwargs):
        print('before')
        func(*args, **kwargs)
        print('after')

    return wrap


def aop_with_param(aop_test_str):
    def aop(func):
        """aop func"""
        @wraps(func)
        def wrap(*args, **kwargs):
            print('before ' + str(aop_test_str))
            func(*args, **kwargs)
            print('after ' + str(aop_test_str))
        return wrap
    return aop


def init_env():
    pass


def init_test_distributed_sys_2(parties_number: int = 2):

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    sf.shutdown()  # 防止集群是启动状态的
    if parties_number == 2:
        sf.init(parties=['alice', 'bob'],
                num_cpus=8,
                log_to_driver=True)
        return sf.PYU('alice'), sf.PYU('bob'), sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    # sf.init(parties=['alice', 'bob'],
    #         num_cpus=8,
    #         log_to_driver=True)
    # alice, bob = sf.PYU('alice'), sf.PYU('bob')
    # spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))


