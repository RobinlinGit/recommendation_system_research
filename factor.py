# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy.sparse import dok_matrix, csc_matrix
from scipy.sparse import load_npz


# %%
Xtrain = load_npz("./train.npz")
Xtest = load_npz("./test.npz")
# Xtrain = np.array(Xtrain)
# Xtest = np.array(Xtest)


# %%
k = 10
alpha = 1e-3
l = 0.1


# %%
A = np.array((Xtrain != 0).todense())
# A = Xtrain != 0
B = np.logical_not(A)


# %%
max_iter = 100
rmse_thres = 1


# %%
non_zeros = np.sum(A)


# %%
print(non_zeros)


# %%
N, M = Xtrain.shape
U = np.random.random((N, k))
V = np.random.random((M, k))


# %%
best_loss = float('inf')
U_best = U
V_best = V
for i in range(max_iter):
    a = U.dot(V.T)
    a[B] = 0
    a = csc_matrix(a)
    dif = Xtrain - a
    print(a[0].sum())
    c = dif.dot(V)

    U2 = (1 - 2 * l * alpha) * U + alpha * dif.dot(V)
    V2 = (1 - 2 * l * alpha) * V + alpha * dif.dot(U)
    U = U2
    V = V2
    dif.data **= 2
    loss = [0.5 * dif.sum(), l * np.sum(U ** 2), l * np.sum(V ** 2)]
    print(f"{i}: loss: {loss}, J= {np.sum(loss)}")
    break


# %%
a = np.array([[1, 2]])
b = np.array([[2], [1]])
print(a.dot(b))


# %%



