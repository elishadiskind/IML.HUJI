from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base.base_estimator import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import misclassification_error
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        error = np.inf
        feature_ind = None
        threshold = None
        sign = None
        for i, feature in enumerate(X.T):
            temp_threshold_n, temp_error_n = self._find_threshold(feature, y, -1)
            if temp_error_n < error:
                error = temp_error_n
                self.j_ = i
                self.threshold_ = temp_threshold_n
                self.sign_ = -1
            temp_threshold_p, temp_error_p = self._find_threshold(feature, y, 1)
            if temp_error_p < error:
                error = temp_error_p
                self.j_ = i
                self.threshold_ = temp_threshold_p
                self.sign_ = 1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        min_all_losses = np.inf
        index = 0
        for i, sample in enumerate(values):
            temp_pred = np.where(values >= sample, sign, -sign)
            # all_losses.append(misclassification_error(labels, temp_pred))
            loss = np.sum(np.abs(labels)[np.where(np.sign(labels) != temp_pred)])
            if min_all_losses > loss:
                min_all_losses = loss
                index = i
        return values[index], min_all_losses

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(np.sign(y), np.sign(self._predict(X)))
