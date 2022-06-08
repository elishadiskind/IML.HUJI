import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def decision_surface_by_iters(T, predict, xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], T)
    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                          marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False), hoverinfo="skip",
                          showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False,
                      opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    # (train_X, train_y), (test_X, test_y) = generate_data(5000, 0), generate_data(500,0)
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    model = AdaBoost(DecisionStump, iterations=n_learners)
    model.fit(train_X, train_y)

    train_loss = []
    test_loss = []
    for i in range(1, n_learners + 1):
        train_loss.append(model.partial_loss(train_X, train_y, T=i))
        test_loss.append(model.partial_loss(test_X, test_y, T=i))
    fig = make_subplots(rows=1, cols=1, subplot_titles=['misclassification error as function of number of models used'])
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=train_loss, name="loss ot train set", mode='lines'),
                  row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=test_loss, name="loss ot test set", mode='lines'),
                  row=1,
                  col=1)
    fig.write_html('Q1_noise=' + str(noise) + '.html', auto_open=True)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=['num of iterations = ' + str(i) for i in T],
                        horizontal_spacing=0.04, vertical_spacing=0.07)
    for i, t in enumerate(T):
        # fig.add_trace([decision_surface_by_iters(t, model.partial_predict, lims[0], lims[1], showscale=False),
        #                go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
        #                           marker=dict(color=test_y))], row=(i // 2) + 1, col=(i % 2) + 1)
        fig.add_trace(decision_surface_by_iters(t, model.partial_predict, lims[0], lims[1], showscale=False),
                      row=(i // 2) + 1, col=(i % 2) + 1)
        fig.add_trace(
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", marker=dict(color=test_y)),
            row=(i // 2) + 1, col=(i % 2) + 1)
    fig.write_html('Q2_noise=' + str(noise) + '.html', auto_open=True)

    # Question 3: Decision surface of best performing ensemble
    min_loss_ind = test_loss.index(min(test_loss))
    fig = make_subplots(rows=1, cols=1, subplot_titles=[
        'the ensemble with the lowest error size is ' + str(min_loss_ind + 1) + ' and accuracy of ' + str(
            1 - test_loss[min_loss_ind])])
    fig.add_trace(decision_surface_by_iters(min_loss_ind + 1, model.partial_predict, lims[0], lims[1], showscale=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", marker=dict(color=test_y)),
                  row=1, col=1)
    fig.write_html('Q3_noise=' + str(noise) + '.html', auto_open=True)
    # Question 4: Decision surface with weighted samples
    D = model.D_ / np.max(model.D_) * 5
    fig = make_subplots(rows=1, cols=1, subplot_titles=[
        'This plot shows the weight of each point in the last iterations of the adaboost'])
    fig.add_trace(decision_surface_by_iters(n_learners, model.partial_predict, lims[0], lims[1], showscale=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", marker=dict(color=train_y, size=D),
                             line=dict(color="black", width=1)),
                  row=1, col=1)
    fig.write_html('Q4_noise=' + str(noise) + '.html', auto_open=True)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
