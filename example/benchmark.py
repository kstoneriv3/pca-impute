"""
Benchamrk script comparing imputers from scikit-learn and SVD imputer from fancy impute.
On my laptop with Intel i7-8750H and 32 GB memory, the benchmark result looked like this:

$ python benchmark.py
Mean imputation: 0.027649641036987305 sec
MSE:    4.660934114515747
pca_impute.impute: 0.4510965347290039 sec
MSE:    0.4987599067216591
fancyimpute.IterativeSVD: 1.6951684951782227 sec
MSE:    0.7397661208001316
sklearn.impute.IterativeImputer: 87.45235776901245 sec
MSE:    0.557537819762688
sklearn.impute.KNNImputer: 51.810137033462524 sec
MSE:    1.280416815792845
"""
from time import time

import fancyimpute
import numpy as np
import sklearn.impute

import pca_impute


class Timer:
    def __init__(self, description="Execution time"):
        self.description = description

    def __enter__(self):
        self.start = time()

    def __exit__(self, type, value, traceback):
        self.end = time()
        print(f"{self.description}: {self.end - self.start} sec")


def sample_toy_data(n, d, n_components):
    W = np.random.randn(d * n_components).reshape(d, n_components)
    z = np.random.randn(n * n_components).reshape(n, n_components)
    eps = np.random.randn(n * d).reshape(n, d)
    bias = np.random.randn(d)
    return z @ W.T + eps + bias[None, :]


def add_nans(X, p):
    isnan = np.random.rand(*X.shape) < p
    X = np.where(isnan, np.nan, X)
    return X


def main():
    np.random.seed(0)

    X0 = sample_toy_data(10000, 100, 10)
    X = add_nans(X0, 0.4)

    def print_mse(x):
        return print(f"MSE:\t{np.mean((x - X0) ** 2)}")

    with Timer("Mean imputation"):
        X_mean = np.where(np.isnan(X), np.nanmean(X), X)
    print_mse(X_mean)

    with Timer("pca_impute.impute"):
        X_pca_impute = pca_impute.impute(X, 10, n_iter=10)
    print_mse(X_pca_impute)

    with Timer("fancyimpute.IterativeSVD"):
        svd_imputer = fancyimpute.IterativeSVD(rank=10, max_iters=10, verbose=False)
        X_fancy_svd = svd_imputer.solve(X_mean, np.isnan(X))
    print_mse(X_fancy_svd)

    with Timer("sklearn.impute.IterativeImputer"):
        iter_imputer = sklearn.impute.IterativeImputer()
        X_sk_iter = iter_imputer.fit_transform(X)
    print_mse(X_sk_iter)

    with Timer("sklearn.impute.KNNImputer"):
        knn_imputer = sklearn.impute.KNNImputer()
        X_sk_knn = knn_imputer.fit_transform(X)
    print_mse(X_sk_knn)


if __name__ == "__main__":
    main()
