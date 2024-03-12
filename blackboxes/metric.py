"""
  Copyright (c) 2016- by Dietmar W Weiss

  This is free software; you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 3.0 of
  the License, or (at your option) any later version.

  This software is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this software; if not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02110-1301 USA, or see the FSF site: http://www.fsf.org.

  Version:
      2024-03-08 DWW
"""

from __future__ import annotations

__all__ = ['best_metric_index', 'init_metric', 'label_str', 'plot_error_bars',
           'plot_hist_loss', 'title_label', 'update_pred_error']

from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Dict, Sequence


def best_metric_index(metrices: Sequence[Dict[str, Any]],
                      key: str = 'loss') -> int | None:
    """
    :param metrices:
        list of metrices
    :param key:
        indicator
    :return:
        index of best metric
        OR
        None if metrices is empty
    """
    i_best = None
    best = None
    for i in range(len(metrices)):
        if (best is None
                or best.get(key, np.inf) > metrices[i_best].get(key, np.inf)):
            best = metrices[i]
            i_best = i

    return i_best


def init_metric(other: Dict[str, Any] | None = None,
                ) -> Dict[str, Any]:
    """
    Sets default values to metric describing model performance

    Args:
        other:
            other dictionary with initial metric to be added/updated
            to this metrics

    Returns:
        dictionary with default settings for metric of best training
    """

    metric: Dict[str, Any] = dict(
        abs=np.inf,          # maximum absolute error
        activation=None,     # activation function of best trial
        backend=None,        # identifier of best identifier
        batch_size=None,     # size of batches of training data
        epochs=-1,           # number of epochs of best trial
        evaluations=-1,      # number of evaluations of best trial
        i_abs=-1,            # 1D index of maximum absolute error
        i_history=-1,        # index of best history
        iterations=-1,       # number of iterations of best trial
        L2=np.inf,           # L2-norm
        loss=np.inf,         # measure for finding best trial, e.g. MSE
        hist_loss=None,      # history of training error vs epochs
        hist_val_loss=None,  # history of validation error vs epochs
        net=None,            # best network
        neurons=None,        # structure of the best neural network
        ready=False,         # True if ready for prediction after train
        time=None,           # training time
        trainer=None,        # trainer of best trial
        trial=-1,            # index of best trial
        val_loss=np.inf,     # measure for validation of best trial
    )

    if other is not None:
        for key, value in other.items():
            metric[key] = value

    return metric


def label_str(metrices: Sequence[Dict[str, Any]] | None,
              i_metric: int) -> str:
    """
    :param metrices:
        list of metrices
    :param i_metric:
        index of metric in metrices
    :return:
        label containing data of metric with given index
    """

    if metrices is None or i_metric >= len(metrices):
        return ''

    n_act = len(Counter(m['activation'] for m in metrices))
    n_bck = len(Counter(m['backend'] for m in metrices))
    n_bsz = len(Counter(m['batch_size'] for m in metrices))
    n_nrn = len(Counter(str(m['neurons']) for m in metrices))
    n_trl = len(Counter(m['trial'] for m in metrices))
    n_trn = len(Counter(m['trainer'] for m in metrices))

    metric = metrices[i_metric]
    backend = metric.get('backend', '?')
    s = ''
    # s += f"{1e3 * metric.get('loss', '?'):5.2f}e-3 "
    s += f"{metric.get('trainer', '?')[:]}:" if n_trn > 1 else ''
    s += f"{metric.get('backend', '?')}:" if n_bck > 1 else ''
    s += f"{metric.get('activation', '?')[:5]} " if n_act > 1 else ''
    s += f"b{metric.get('batch_size') or '-'} " if n_bsz > 1 else ''
    s += f"[{str(metric['neurons']).replace(', ', ':')[1:-1]}] " \
        if n_nrn > 1 else ''
    s += f"#{metric.get('trial', '?')} " if n_trl > 1 else ''

    return s


def plot_error_bars(metrices: Sequence[Dict[str, Any]],
                    n_best: int | None = None,
                    key: str = 'loss',
                    expected: float | None = None,
                    tolerated: float | None = None) -> None:
    """
    Args:
        metrices:
            sequence of metrices

        n_best:
            number of best metrices to be plotted

        key:
            key of loss in metric

        expected:
            expected loss as vertical line

        tolerated:
            tolerated loss as vertical line
    """
    plt.rcParams.update({'font.size': 10})  # axis fonts
    plt.rcParams['legend.fontsize'] = 6  # fonts in legend

    metrices = sorted(metrices, reverse=False, key=itemgetter(key))
    best = metrices[0]
    if n_best is not None:
        metrices = metrices[:n_best]

    labels, losses = [], []
    for i_metric in range(len(metrices)):
        labels.append(label_str(metrices, i_metric))
        losses.append(metrices[i_metric].get(key))

    fig_size_x = 6.
    fig_size_y = np.clip(len(losses) * 0.25, 3., 20.)
    figsize = (fig_size_x, fig_size_y)
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(losses))
    ax.barh(y_pos, losses, align='center')
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()

    if expected is not None:
        ax.vlines(x=expected, ymin=-0.5, ymax=len(losses)-0.5,
                  color='green', linestyles='-', label='expected')
    if tolerated is not None:
        ax.vlines(x=tolerated, ymin=-0.5, ymax=len(losses)-0.5,
                  color='red', linestyles='-', label='tolerated')

    plt.title('best loss:' + title_label(best))
    plt.subplots_adjust(left=0.35,right=0.95)
    plt.xscale('log')
    plt.xlabel('loss')
    plt.xlim(1e-5, 1e-1)
    plt.ylabel('train config')
    plt.grid()
    plt.legend()
    plt.show()


def plot_hist_loss(
        metrices: Sequence[Dict[str, ArrayLike | Any]] | None = None,
        hist_key: str = 'hist_loss',
        sort_list: bool = True,
        n_best: int | None = None,
        expected: float | None = None,
        tolerated: float | None = None) -> bool:

    plt.rcParams.update({'font.size': 8})  # axis fonts
    plt.rcParams['legend.fontsize'] = 8  # fonts in legend

    if n_best is None:
        n_best = -1

    if not hist_key or not metrices:
        return False
    if sort_list:
        metrices = sorted(metrices, reverse=False, key=itemgetter('loss'))
        best = metrices[0]
    else:
        best = best_metric_index(metrices)
    title = title_label(best)

    max_epochs = 0
    for metric in metrices:
        y = metric.get(hist_key)
        if max_epochs < len(y):
            max_epochs = len(y)

    plt.title(f'Best: {title}')

    has_labels = False
    for i, metric in enumerate(metrices[:n_best]):
        y = metric.get(hist_key)
        if y is None or not len(y):
            continue

        has_labels = True
        label = title_label(metric)
        if len(y) > 1:
            ls = '-' if not sort_list or i == 0 else ':'
            plt.plot(y, ls=ls, label=label)
        else:
            plt.scatter([0], y, label=label)

    y_best = best.get(hist_key)
    plt.scatter(len(y_best)-1, y_best[-1], label='best')

    if expected is not None:
        plt.hlines(y=expected, xmin=0., xmax=max_epochs, ls='--',
                   color='g', label='expected')
    if tolerated is not None:
        plt.hlines(y=tolerated, xmin=0., xmax=max_epochs, ls='--',
                   color='r', label='tolerated')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.ylim(None, 1)
    if has_labels:
        if n_best != -1:
            plt.legend()
        else:
            plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
            plt.subplots_adjust(left=0.1, right=0.6)
    plt.grid()
    plt.show()

    return True


def title_label(metric: Dict[str, Any] | None) -> str:
    if metric is None:
        return ''

    return (f"{1e3 * metric.get('loss'):5.2f}e-3 "
            f"#{metric.get('trial')} "
            f"{metric.get('trainer')[:]}:"
            f"{metric.get('activation')[:5]} "
            f"B{metric.get('batch_size') or '-'} "
            f"[{str(metric['neurons']).replace(', ', ':')[1:-1]}] "
            f"({metric.get('backend', '?')})")


def update_pred_error(metric: Dict[str, Any],
                      X: ArrayLike | None,
                      Y: ArrayLike | None,
                      y: ArrayLike | None,
                      **kwargs: Any) -> Dict[str, Any]:

    do_plot = kwargs.get('plot', 0) > 0
    silent = kwargs.get('silent', True)

    if X is None or Y is None or y is None:
        if not silent:
            if X is None:
                print('??? metric: X is None')
            if Y is None:
                print('??? metric: Y is None')
            if y is None:
                print('??? metric: y is None')
        return metric

    X, Y, y = np.asfarray(X), np.asfarray(Y), np.asfarray(y)
    if len(Y) != len(y):
        if not silent:
            print('??? len(Y) != len(y)')
            print(f'??? {X.shape=}, {Y.shape=}, {y.shape=}')
        return metric

    try:
        dy = y.ravel() - Y.ravel()
    except (ValueError, AttributeError):
        assert 0, f'{X.shape=}, {Y.shape=}, {y.shape=}'

    i_abs = np.abs(dy).argmax()
    abs_ = dy.ravel()[i_abs]
    L2 = np.sqrt(np.mean(np.square(dy)))

    if not isinstance(metric, dict):
        metric = init_metric()
    metric['abs'] = abs_
    metric['i_abs'] = i_abs
    metric['L2'] = L2

    if not silent:
        print(f'    L2={1e3*L2:.2f}e-3, max(abs(y-Y))={1e3*abs_:6.2f}e-3 at '
              f'X[{i_abs}]={X.ravel()[i_abs]:.3f} '
              f'Y[{i_abs}]={Y.ravel()[i_abs]:.3f} '
              f'y[{i_abs}]={y.ravel()[i_abs]:.3f}')

    if not do_plot:
        plt.title(f"Prediction versus reference [{metric.get('backend')}]")
        plt.xlabel('$X$')
        plt.ylabel('$y, Y$')
        plt.plot(X.reshape(-1), y.reshape(-1), label='prd $y$')
        plt.plot(X.reshape(-1), Y.reshape(-1), label='ref $Y$')
        plt.plot(X.reshape(-1)[i_abs], Y.reshape(-1)[i_abs], 'o',
                 label='max abs')
        plt.grid()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        plt.title(f"Maximum absolute error [{metric.get('backend')}]")
        plt.xlabel('$X$')
        plt.ylabel('$y - Y$')
        plt.plot(X.reshape(-1), (y - Y).reshape(-1), label='$y-Y$')
        plt.plot(X.reshape(-1)[i_abs], (y - Y).reshape(-1)[i_abs], 'o',
                 label='max abs')
        plt.grid()
        plt.legend()
        plt.show()

    return metric
