#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   factor_script.py
@Time    :   2020/10/30 22:57:09
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   multiprocess on factor.py
'''
from factor import factor
from multiprocessing import Process, Pool, freeze_support
from scipy.sparse import load_npz
import numpy as np
import json
import os


def experiment(k, lamb):
    print(f"Process {k} {lamb} start")
    folder = f"./record/k_{k}_lambda_{lamb}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, "log.txt"), 'w+') as f:
        Xtrain = np.array(load_npz("train.npz").todense())
        Xtest = np.array(load_npz("test.npz").todense())
        U, V, record = factor(Xtrain, Xtest, k, 1e-2, lamb, 500, rmse_thd=1e-5, f=f)
        np.savetxt(os.path.join(folder, "U.txt"), U)
        np.savetxt(os.path.join(folder, "V.txt"), V)
    with open(os.path.join(folder, "record.json"), "w") as f:
        record = [(float(x[0]), float(x[1])) for x in record]
        f.write(json.dumps(record))
    print(f"process k: {k}, lambda {lamb} finish")


if __name__ == "__main__":
    # freeze_support()
    k_list = [10, 20, 50, 100, 200]
    lambda_list = [0.001, 0.01, 0.1, 1]
    print(k_list)
    print(lambda_list)


    pool = Pool(2)
    for k in k_list:
        for l in lambda_list:
            pool.apply_async(experiment, args=(k, l, ))

    print("Waiting for all process done")
    pool.close()
    pool.join()
    print("All processes done")

