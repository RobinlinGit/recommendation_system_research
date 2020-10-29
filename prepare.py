#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   prepare.py
@Time    :   2020/10/28 17:00:45
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   load data and store in npz file
'''
import json
from tqdm import tqdm
from scipy.sparse import dok_matrix, save_npz


movie_num = 10000
with open("./data/users.txt", "r") as f:
    users = [int(x) for x in f.readlines()]
    users_dict = {x: i for i, x in enumerate(users)}

train_mat = dok_matrix((len(users), movie_num))
with open("./data/netflix_train.txt", 'r') as f:
    train_rank = []
    for line in tqdm(f.readlines()):
        uid, mid, rank, t = line.split(" ")
        uid = int(uid)
        mid = int(mid)
        rank = int(rank)
        train_mat[users_dict[uid], mid-1] = rank

test_mat = dok_matrix((len(users), movie_num))
with open("./data/netflix_test.txt", 'r') as f:
    test_rank = []
    test_users = set()
    for line in tqdm(f.readlines()):
        uid, mid, rank, t = line.split(" ")
        uid = int(uid)
        mid = int(mid)
        rank = int(rank)
        test_mat[users_dict[uid], mid-1] = rank

save_npz("./train.npz", train_mat.tocsr())
save_npz("./test.npz", test_mat.tocsr())
with open("row2user.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(users_dict))

