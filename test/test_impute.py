import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

import pca_impute

np.random.seed(0)


Data = tuple[np.ndarray, np.ndarray]


def add_nans(X: np.ndarray, p: float = 0.2) -> np.ndarray:
    isnan = np.random.rand(*X.shape) < p
    X = np.where(isnan, np.nan, X)
    return X


@pytest.fixture(scope="session")
def data() -> Data:
    X0 = load_breast_cancer()["data"]
    X = add_nans(X0)
    return X0, X


def test_impute(data: Data) -> None:
    X0, X = data
    X_mean = pca_impute._impute._impute_by_mean(X)
    X_pca = pca_impute.impute(X, n_components=3)

    assert np.any(np.isnan(X))
    assert not np.any(np.isnan(X_pca))

    SST = np.sum((X0 - X_mean) ** 2)
    SSR = np.sum((X0 - X_pca) ** 2)
    R2 = 1 - SSR / SST

    # We expect pca should be much better than mean imputation.
    assert R2 > 0.8


def test_impute_auto_n_components(data: Data) -> None:
    X0, X = data
    X_pca_1 = pca_impute.impute(X, n_components=1)
    X_pca_auto = pca_impute.impute(X, n_components="auto")
    SSR_pca_1 = np.sum((X0 - X_pca_1) ** 2)
    SSR_pca_auto = np.sum((X0 - X_pca_auto) ** 2)
    assert SSR_pca_auto < SSR_pca_1
