# pca-impute
A reasonably fast and accurate missing value imputation by iterative PCA.


## Motivation
The PCA-based missing value imputer is known to be reasonably fast and accurate, but scikit-learn has only slower and/or less advanced imputers IMPO. So here is an implementation of the PCA-based imputer!

Note: I didn't know fancyimputer before implementing this, but this implementation seems faster than theirs for some reason...


## Installation

```bash
pip install pca-impute
```

## Usage
```python
>>> import pca_impute
>>> X = np.array([[1, 2, 3], [2, 3, None], [3, None, 6], [None, 6, None]], dtype=float)
>>> pca_impute.impute(X, n_components=1)
array([[1.        , 2.        , 3.        ],
       [2.        , 3.        , 4.25710236],
       [3.        , 5.77464251, 6.        ],
       [2.9853071 , 6.        , 5.99766182]])
>>>
```


## Algorithm
1) Initialize the estimate of the missing values by the mean.
Then iterate 2-5 to refine the estimate.
2) Fill the missing values of original data with the estimate.
3) Fit a PCA to the filled data. 
4) Reconstruct the filled data by the PCA. (i.e. project the data to the subspace found by the PCA.)
5) Update the estimate of the missing values by taking the corresponding entries from the reconstruction.


## A bit of thoery

If I'm not mistaken, this algorithm solves 
$$\hat X = \arg\min_{\tilde X\in \mathbb{R}^{n\times d}: \mathrm{Rank}(\tilde X) \leq d} \sum_{(i, j) \in O} \|X_{i,j} - \tilde X_{i, j}\|^2,$$
where $O = \\{(i, j):X_{i, j} \text{ is observed} \\}$, to estimate the missing values.

This is because the above algorithm corresponds to the (block) coordinate descent of 
$$(\hat X, \hat M) = \arg\min_{\tilde X, M\in \mathbb{R}^{n\times d}: \mathrm{Rank}(\tilde X) \leq d} \left[ \sum_{(i, j) \in O} \|X_{i,j} - \tilde X_{i, j}\|^2 + \sum_{(i, j) \not\in O} \|M_{i,j} - \tilde X_{i, j}\|^2 \right],$$
where $M$ corresponds to the guess of missing values and $\tilde X$ corresponds to the PCA reconstruction.


## Comparison to other algorithms / implementations
According to my benchmark with synthetic data from a probabilistic PCA model (with missing at random assumption), pca-impute had better mean squared error (MSE) for predicting artificially dropped values, compared to other imputation algorithms available in scikit-learn.

Additionally, the execution time of the algorithm was about 3 times faster than a similar implementation of the iterative PCA (iterative SVD) algorithm in fancyimpute and our implementation had slightly better MSE.

The benchmark script is available at `example/benchmark.py`, and the following is the result I got on my laptop:

```bash
$ python example/benchmark.py
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
```
