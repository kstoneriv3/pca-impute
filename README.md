# pca-impute
Missing value imputation by iterative PCA.

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
1) Initialize the estimate of the missing values by the mean (or anything reasonable).
Then iterate 2-4 to refine the estimate.
2) Fit a PCA to the data filled by the estimate. 
3) Reconstruct the data with the PCA (i.e. project the data to the subspace found by PCA.).
4) Update the estimate of the missing values by taking the corresponding entries from the reconstruction.


## Comparison to other imputers / implementations
According to my benchmarking with synthetic data from a probabilistic PCA model (wht missing at random assumption), `pca-impute` had better mean squared error (MSE) compared to other imputation algorithms available in sklearn. Additionally, runtime of the algorithm was about 3 times faster than a similar implementation of iterative PCA (iterative SVD) algorithm, with our implementation having slightly better MSE. The benchmark script is available at `example/benchmark.py`, and I got the following result on my laptop:

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
"""

