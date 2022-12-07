# -*- coding: utf-8 -*- 
# @Time : 2022/11/3 09:00 
# @Author : YeMeng 
# @File : demo6_spu_secure.py 
# @contact: 876720687@qq.com

import secretflow as sf
from demo3_spu_neral.demo3_train import *

if __name__ == "__main__":

    sf.shutdown()
    sf.init(['alice', 'bob'], num_cpus=8, log_to_driver=True)
    alice, bob = sf.PYU('alice'), sf.PYU('bob')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    x1, _ = alice(breast_cancer)(party_id=1, train=True)
    x2, y = bob(breast_cancer)(party_id=2, train=True)
    init_params = model_init(n_batch)

    device = spu
    x1_, x2_, y_ = x1.to(device), x2.to(device), y.to(device)
    init_params_ = sf.to(device, init_params)

    params_spu = spu(train_auto_grad, static_argnames=['n_batch', 'n_epochs', 'step_size'])(
        x1_, x2_, y_, init_params_, n_batch=n_batch, n_epochs=n_epochs, step_size=step_size
    )

    # ------------------------ validate the model --------------
    X_test, y_test = breast_cancer(train=False)
    auc = validate_model(params, X_test, y_test)
    print(f'auc={auc}')