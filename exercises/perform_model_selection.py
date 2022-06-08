from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.random.uniform(low=-1.2, high=2, size=100)
    f_x = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    epsilon = np.random.normal(loc=0, scale=5, size=100)
    y = f_x + epsilon
    train_x, train_y, test_x, test_y = split_train_test(x, y, 2 / 3)
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Obs plot m = 100, noise = 5'])
    fig.add_trace(go.Scatter(x=x, y=f_x, mode="markers", name="'real X'"), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_x, y=test_y, mode="markers", name='test set'), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_x, y=train_y, mode="markers", name='train set'), row=1, col=1)
    fig.write_html('Q1.html', auto_open=True)
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_error = []
    vali_error = []
    for k in range(11):
        train_er, vali_err = cross_validate(PolynomialFitting(k=k), train_x, train_y, mean_square_error, cv=5)
        train_error.append(train_er)
        vali_error.append(vali_err)
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Obs plot m = 100, noise = 5'])
    fig.add_trace(go.Scatter(x=list(range(11)), y=train_error, mode="markers", name='Train error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(11)), y=vali_error, mode="markers", name='Validation error'), row=1, col=1)
    fig.write_html('Q2.html', auto_open=True)
    print(vali_error.index(min(vali_error)))
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # k* = 4 is the best one
    model_3 = PolynomialFitting(k=4).fit(train_x, train_y)
    test_error = model_3.loss(test_x, test_y)
    print(np.round(test_error, 2))
    # Question 4 - repeat the questions above
    y = f_x  # no noise
    train_x, train_y, test_x, test_y = split_train_test(x, y, 2 / 3)
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Obs plot m = 100, noise = 0'])
    fig.add_trace(go.Scatter(x=test_x, y=test_y, mode="markers", name='test set'), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_x, y=train_y, mode="markers", name='train set'), row=1, col=1)
    fig.write_html('Q1.1.html', auto_open=True)
    # Question 2.1 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_error = []
    vali_error = []
    for k in range(11):
        train_er, vali_err = cross_validate(PolynomialFitting(k=k), train_x, train_y, mean_square_error, cv=5)
        train_error.append(train_er)
        vali_error.append(vali_err)
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Obs plot m = 100, noise = 0'])
    fig.add_trace(go.Scatter(x=list(range(11)), y=train_error, mode="markers", name='Train error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(11)), y=vali_error, mode="markers", name='Validation error'), row=1, col=1)
    fig.write_html('Q2.1.html', auto_open=True)
    print(vali_error.index(min(vali_error)))
    # Question 3.1 - Using best value of k, fit a k-degree polynomial model and report test error
    model_3 = PolynomialFitting(k=10).fit(train_x, train_y)
    test_error = model_3.loss(test_x, test_y)
    print(np.round(test_error, 2))
    # Question 5
    x = np.random.uniform(low=-1.2, high=2, size=1500)
    f_x = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    epsilon = np.random.normal(loc=0, scale=10, size=1500)
    y = f_x + epsilon
    train_x, train_y, test_x, test_y = split_train_test(x, y, 2 / 3)
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Obs plot m = 1500, noise = 10'])
    fig.add_trace(go.Scatter(x=x, y=f_x, mode="markers", name="'real X'"), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_x, y=test_y, mode="markers", name='test set'), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_x, y=train_y, mode="markers", name='train set'), row=1, col=1)
    fig.write_html('Q1.2.html', auto_open=True)
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_error = []
    vali_error = []
    for k in range(11):
        train_er, vali_err = cross_validate(PolynomialFitting(k=k), train_x, train_y, mean_square_error, cv=5)
        train_error.append(train_er)
        vali_error.append(vali_err)
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Obs plot m = 1500, noise = 10'])
    fig.add_trace(go.Scatter(x=list(range(11)), y=train_error, mode="markers", name='Train error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(11)), y=vali_error, mode="markers", name='Validation error'), row=1, col=1)
    fig.write_html('Q2.2.html', auto_open=True)
    print(vali_error.index(min(vali_error)))
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # k* = 4 is the best one
    model_3 = PolynomialFitting(k=4).fit(train_x, train_y)
    test_error = model_3.loss(test_x, test_y)
    print(np.round(test_error, 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y, test_x, test_y = split_train_test(X, y, n_samples / len(y))

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    n_lambdas = np.linspace(0.0002, 1, n_evaluations)
    R_train_error = []
    R_vali_error = []
    L_train_error = []
    L_vali_error = []
    for lam in n_lambdas:
        ridg_m = RidgeRegression(lam=lam)
        laso_m = Lasso(alpha=lam)
        r_train_er, r_vali_er = cross_validate(ridg_m, train_x, train_y, mean_square_error, cv=5)
        l_train_er, l_vali_er = cross_validate(laso_m, train_x, train_y, mean_square_error, cv=5)
        R_train_error.append(r_train_er)
        R_vali_error.append(r_vali_er)
        L_train_error.append(l_train_er)
        L_vali_error.append(l_vali_er)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=n_lambdas, y=L_train_error, mode="lines", name='Lasso Train error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=n_lambdas, y=L_vali_error, mode="lines", name='Lasso Validation error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=n_lambdas, y=R_train_error, mode="lines", name='Ridge Train error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=n_lambdas, y=R_vali_error, mode="lines", name='Ridge Validation error'), row=1, col=1)
    fig.write_html('Q7.html', auto_open=True)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_lasso = n_lambdas[np.argmin(L_vali_error)]
    best_lam_ridge = n_lambdas[np.argmin(R_vali_error)]
    lasso_model = Lasso(alpha=best_lam_lasso).fit(train_x, train_y)
    lasso_error = mean_square_error(test_y, lasso_model.predict(test_x))
    ridge_model = RidgeRegression(lam=best_lam_ridge).fit(train_x, train_y)
    ridge_error = mean_square_error(test_y, ridge_model.predict(test_x))
    ls_model = LinearRegression().fit(train_x, train_y)
    ls_error = mean_square_error(test_y, ls_model.predict(test_x))
    print(lasso_error, ridge_error, ls_error)
    print(best_lam_ridge,best_lam_lasso)

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_regularization_parameter()
