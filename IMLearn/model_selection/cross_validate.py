from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    #print(np.shape(X))
    m = X.shape[0]
    trains_score = []
    validations_score = []
    for i in range(cv):
        d_index = i * m // cv
        u_index = (i + 1) * m // cv
        c_test_x = X[d_index:u_index]
        c_test_y = y[d_index:u_index]
        c_x = np.concatenate((X[:d_index], X[u_index:]))
        c_y = np.concatenate((y[:d_index], y[u_index:]))
        c_model = estimator.fit(c_x, c_y)
        trains_score.append(scoring(c_y, c_model.predict(c_x)))
        validations_score.append(scoring(c_test_y, c_model.predict(c_test_x)))
    return np.average(np.array(trains_score)), np.average(np.array(validations_score))
