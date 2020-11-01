# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from time import time
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse import load_npz


def factor(Xtrain, Xtest, k, lr, lamb, max_iter, rmse_thd=1e-5, f=None):
    """matrix factor X = UV^T
    Args:
        Xtrain (np.ndarray): size [user, movie].
        Xtest (np.ndarray): [user, movie].
        k (int): U [user, k], V [movie, k].
        lamb (float): regular param.
        mat_iter (int): max loop iterations.
        rmse_thd (float): if update rmse < rmse_thd, break loop.
        f (file): redirect output to file.
    Returns:
        U (np.ndarray).
        V (np.ndarray).
        record (list [(loss, rmse on test data)]).
    """
    # A = np.array((Xtrain != 0).todense())
    A = Xtrain != 0
    B = np.logical_not(A)
    non_zeros = np.sum(A)

    N, M = Xtrain.shape
    U = (np.random.rand(N, k) - 0.5) * 0.1 / np.sqrt(k)
    V = (np.random.rand(M, k) - 0.5) * 0.1 / np.sqrt(k)
    best_loss = float('inf')
    record = []

    for i in range(max_iter):
        a = U.dot(V.T)
        a[B] = 0
        start = time()
        dif = Xtrain - a
        U2 = (1 - 2 * lamb * lr) * U + lr * dif.dot(V)
        V2 = (1 - 2 * lamb * lr) * V + lr * dif.dot(U)
        loss = loss_(Xtrain, U2, V2, B, lamb)
        print_str = [f"loss {loss}"]
        if loss < best_loss:
            best_loss = loss
            U = U2
            V = V2
            rmse = RMSE(Xtest, U, V)
            record.append((loss, rmse))
            print_str.append(f" rmse: {rmse}")
        else:
            print_str.append(f" lr {lr} --> {lr / 10}")
            lr /= 10
        print_str.insert(0, f"iter {i} time: {time() - start}s ")
        if f is None:
            print("".join(print_str))
        else:
            print("".join(print_str), file=f)

        # stop if rmse update < thd, stop iter
        if len(record) >= 2:
            update = record[-2][1] - record[-1][1]
            if update < rmse_thd and update > 0:
                break
    return U, V, record


def RMSE(X, U, V):
    # mask = np.array((X == 0).todense())
    mask = X == 0
    Xpred = U.dot(V.T)
    Xpred[mask] = 0
    return np.sum((Xpred - X) ** 2) / np.sum(X != 0)


def loss_(X, U, V, mask, lamb):
    a = U.dot(V.T)
    a[mask] = 0
    dif = X - a
    # dif.data **= 2
    return 0.5 * np.sum(dif ** 2) + lamb * (np.sum(U ** 2) + np.sum(V ** 2))

# %%
# Xtrain = load_npz("./train.npz")
# Xtest = load_npz("./test.npz")
# Xtrain = np.array(load_npz('./train.npz').todense())
# Xtest = np.array(load_npz("./test.npz").todense())


# # %%
# k = 3
# lr = 0.1
# lamb = 1e-4
# max_iter = 200


# # %%
# U, V, record = factor(Xtrain, Xtest, k, lr, lamb, max_iter)

