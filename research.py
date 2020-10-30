# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import json
from tqdm import tqdm
from time import time
from scipy.sparse import dok_matrix, load_npz
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random


# %%
def RMSE(X0, X1):
    """calc RMSE between Xtest and Xpred.

    Args:
        X0, X1 (sparse array or np.array).
    Returns:
        rmse (float)
    """
    non_zeros = np.sum(X0 != 0)
    dif = X0 - X1
    rmse = np.sum(dif.multiply(dif)) / non_zeros
    return np.sqrt(rmse)


# %%
# load the data
# csc for faster col index
Xtrain = load_npz("train.npz").tocsc()
Xtest = load_npz("test.npz")


# %%
start = time()
sim_mat = cosine_similarity(Xtrain, Xtrain)
np.fill_diagonal(sim_mat, 0)
print(f"constuct sim mat cost {time() - start} s")


# %%
# iter Xtest where Xtest != 0
rmse_dict = {}
for k in [10, 50, 100, 200, 500, 1000, 2000]:
    start = time()
    Xpred = dok_matrix(Xtest.shape)
    Xtest = Xtest.tocoo()
    mask_mat = (Xtrain != 0).todense()
    for user, movie, _ in tqdm(zip(Xtest.row, Xtest.col, Xtest.data)):
        # mask similarity for user who did not rank movie
        movie_score = Xtrain[:, movie]  # shape [10000, 1]
        # mask = np.array((movie_score != 0).todense()).reshape(-1)
        mask = np.array(mask_mat[:, movie]).reshape(-1)
        v = sim_mat[user].copy()
        v[np.logical_not(mask)] = 0
        # find top K
        ind = np.argpartition(v, -k)[-k:]
        v[v < np.min(v[ind])] = 0   # v [10000, ]
        # calc rank
        Xpred[user, movie] = (v.reshape(1, -1) * movie_score / np.sum(v))[0, 0]
    rmse_dict[k] = [time() - start, RMSE(Xpred, Xtest)]

for k, v in rmse_dict.items():
    print(f"K: {k}, time: {v[0]}, rmse: {v[1]}")


# %%
# random rank
random_list = []
for i in range(10):
    Xpred = dok_matrix(Xtest.shape)
    for user, movie, _ in tqdm(zip(Xtest.row, Xtest.col, Xtest.data)):
        Xpred[user, movie] = random.randint(1, 5)
    random_list.append(RMSE(Xpred, Xtest))
print(f"rmse {np.mean(random_list)}")

