import numpy as np
from sklearn.datasets import load_breast_cancer

import pca_impute

np.random.seed(0)


def add_nans(X, p):
    isnan = np.random.rand(*X.shape) < p
    X = np.where(isnan, np.nan, X)
    return X


def main():
    # randomly drop 40% of entries
    X0 = load_breast_cancer()["data"]
    X = add_nans(X0, p=0.4)

    # Impute by mean
    X_mean = np.where(np.isnan(X), np.nanmean(X, axis=0, keepdims=True), X)
    mse_mean = np.mean((X0 - X_mean) ** 2)
    print("MSE of mean imputation:\t" + str(mse_mean))

    # Impute by iterative PCA
    X_pca = pca_impute.impute(X, n_components=10)
    mse_pca = np.mean((X0 - X_pca) ** 2)
    print("MSE of PCA imputation:\t" + str(mse_pca))


if __name__ == "__main__":
    main()
