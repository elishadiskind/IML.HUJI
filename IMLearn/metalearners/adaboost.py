import numpy as np
from IMLearn.base.base_estimator import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics.loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        D = (1 / y.shape[0]) * np.ones(y.shape[0])
        W = []
        models = []
        list_of_d = [D]
        for i in range(self.iterations_):
            print('\r' + str(round((i / self.iterations_) * 100, 2)) + ' %', end='')
            d_model = self.wl_().fit(X, y * D)
            models.append(d_model)
            pred = d_model.predict(X)
            epsilon_t = np.sum(D * (pred != y))
            w_t = 0.5 * np.log((1 / epsilon_t) - 1)
            W.append(w_t)
            not_normalized = D * np.exp(w_t * pred * -y)
            D = not_normalized / np.sum(not_normalized)

            list_of_d.append(D)
        self.weights_ = W
        self.models_ = models
        self.D_ = D

        # self.weights_ = []
        # self.models_ = []
        # self.D_ = np.full((len(y)), 1 / len(y))
        # for t in range(self.iterations_):
        #     print('\r' + str(round((t / self.iterations_) * 100, 2)) + ' %', end='')
        #     h = self.wl_().fit(X, self.D_ * y)
        #     y_pred = h.predict(X)
        #     self.models_.append(h)
        #
        #     eps_t = np.sum(self.D_ * (y_pred != y))
        #     w_t = 0.5 * np.log((1 / eps_t) - 1)
        #     self.weights_.append(w_t)
        #
        #     self.D_ = self.D_ * np.exp(-y * w_t * y_pred)
        #     d_sum = np.sum(self.D_)
        #     self.D_ = self.D_ / d_sum

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

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
        y_hat = self.predict(y)
        return misclassification_error(y_hat, y)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        all_predictions = []
        for i in range(T):
            all_predictions.append(self.models_[i].predict(X) * self.weights_[i])
        all_predictions = np.array(all_predictions)

        return np.sign(np.sum(all_predictions, axis=0))

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_hat = self.partial_predict(X, T)
        return misclassification_error(y_hat, y)
