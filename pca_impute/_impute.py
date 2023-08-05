from typing import List, Union

import numpy as np
import scipy


def impute(X: np.ndarray, n_components: Union[int, str] = "auto", n_iter: int = 5) -> np.ndarray:
    """Impute the missing values by iteratively refining the guess by PCA reconstruction.

    Args:
        X: 2-d array with missing values represented by np.nan.
        n_components: The number of components of PCA model. When "auto",
            it is automatically chosen `pca_impute._impute._select_n_components`.
        n_iter: Number of refinement of guessed missing values.

    Returns:
        2-d array which is X with its missing value imputed.
    """
    if n_components == "auto":
        n_components = _select_n_components(X, n_iter)
    assert isinstance(n_components, (int, np.integer))

    isnan = np.isnan(X)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    X = (X - mean[None, :]) / std
    X[isnan] = 0
    for i in range(n_iter):
        _, V = scipy.linalg.eigh(
            X.T @ X,
            subset_by_index=[X.shape[1] - n_components, X.shape[1] - 1],
        )
        X_approx = X @ V @ V.T
        X = np.where(isnan, X_approx, X)
    return X * std + mean


def _impute_by_mean(X: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), np.nanmean(X, axis=0, keepdims=True), X)


def _get_candidate_n_components(n_components: int) -> List[int]:
    max_n_conponents = min(100, int(n_components / 2))
    return list(set(np.geomspace(1, max_n_conponents, 10, dtype=int)))


def _select_n_components(X: np.ndarray, n_iter: int) -> int:
    """Evaluate the quality of imputation by imputing artificially dropped values."""
    is_nan = np.isnan(X)
    p_missing = np.mean(is_nan, axis=0)
    should_drop = np.logical_and(np.random.rand(*X.shape) < p_missing[None, :], ~is_nan)
    X_dropped = X.copy()
    X_dropped[should_drop] = np.nan

    X_baseline = _impute_by_mean(X_dropped)
    candidate_n_components = _get_candidate_n_components(X.shape[1])
    scores = []
    for c in candidate_n_components:
        X_filled = impute(X_dropped, c, n_iter)
        SST = np.nansum((X_baseline - X) ** 2)
        SSR = np.nansum((X_filled - X) ** 2)
        R2 = 1 - SSR / SST
        scores.append(R2)
    return int(candidate_n_components[np.argmax(scores)])
