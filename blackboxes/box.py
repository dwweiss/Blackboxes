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
import numpy as np
from numpy.typing import ArrayLike, NDArray
import sys
from typing import Any, Callable, Dict, List

from blackboxes.bruteforce import BruteForce
from blackboxes.metric import init_metric, update_pred_error
from blackboxes.neuralnlb import Neural as NeuralNlb
from blackboxes.neuraltfl import Neural as NeuralTfl
from blackboxes.neuraltch import Neural as NeuralTch
try:
    from blackboxes.neuralfga import Neural as NeuralFga
except ImportError:
    pass


class Black:
    """
    Implementation of class Black without dependency on grayboxes
    library

    Black base model y = phi(x, w) with w = arg min (phi(X, w) - Y)^2
    where Y(X) is training data, x is arbitrary input and w is weights

    Example:
        from blackboxes.box import Black

        def f(x):
            return x**2

        X = np.linspace(0., 1., 200).reshape(-1, 1)     # training input
        x = np.linspace(-1., 2., 100).reshape(-1, 1)
        x = X * 1.9                                         # test input
        Y = f(X)                                                # target
        y = f(x)                                                  # true

        phi = Black()
        opt = dict(neurons=[8, 6], trainer='adam', trials=5)
        metric = phi.train(X=X, Y=Y, **opt)                   # training
        y_trn = phi.predict(x=X)        # prediction with training input
        y_tst = phi.predict(x=x)            # prediction with test input

    Compact form:
        phi = Black()
        y_trn = phi(X=X, Y=Y, x=X, neurons=[8, 6], trainer='adam')
    """

    def __init__(self, f: Callable[..., ArrayLike] | str | None = None,
                 identifier: str = 'Black') -> None:
        """
        Args:
            f:
                theoretical sub-model

            identifier:
                unique object identifier
        """
        self.identifier: str = identifier
        self.f: Callable[..., ArrayLike] | str | None = f

        self._model: BruteForce | None = None
        self.X: NDArray | None = None
        self.Y: NDArray | None = None
        self.x: NDArray | None = None
        self.y: NDArray | None = None

    @property
    def metric(self) -> Dict[str, Any]:
        return self.model.metric

    @property
    def metrices(self) -> List[Dict[str, Any]]:
        return self.model.metrices

    @property
    def model(self) -> BruteForce | None:
        return self._model

    @property
    def ready(self) -> bool:
        return self.model.ready

    def train(self, X: ArrayLike | None, Y: ArrayLike | None,
              **kwargs: Any) -> Dict[str, Any]:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores
        performance of best training trial as self.metric

        Args:
            X:
                training input, shape: (n_point, n_inp)

            Y:
                training target, shape: (n_point, n_out)

        Kwargs:
            backend (str):
                identifier of backend:
                    'ga',
                    'keras'
                    'neurolab'
                    'tensorflow'
                    'torch'
                default: 'keras'
                    
            neurons (int, sequence of int, or None):
                number of neurons in hidden layer(s) of neural network

            trainer (str, sequence of str, or None):
                optimizer of network, see BruteForce.train()

        Returns:
            metric of best training trial
            OR
            initial metric if X is None or Y is None
        """
        silent = kwargs.get('silent', False)

        if X is None or Y is None:
            if not silent:
                print(f'??? invalid X or Y: {X is None}, {Y is None}')
            return init_metric()

        backend = kwargs.get('backend', 'keras').lower()
        if backend in ('neurolab', 'nl', 'nlb'):
            self._model = NeuralNlb(self.f)
        elif backend in ('tensorflow', 'tf', 'tfl', 'keras', 'ks'):
            self._model = NeuralTfl(self.f)
        elif backend in ('torch', 'to', 'tor', 'tch'):
            self._model = NeuralTch(self.f)
        elif 'ga' in backend and 'neuralfga' in sys.modules:
            self._model = NeuralFga(self.f)
        else:
            self._model = NeuralNlb(self.f)

        if not silent:
            print(f'+++ train (backend={self.model.backend})')
        self.X = np.atleast_2d(X)
        self.Y = np.atleast_2d(Y)
        metric = self.model.train(self.X, self.Y, **kwargs)
        if not silent:
            print(f'+++ best loss={1e3*metric.get("loss"):.2f}e-3\n')

        return metric

    def predict(self, x: ArrayLike | None,
                **kwargs: Any) -> NDArray | None:
        """
        Executes base model, stores input as self.x and output as self.y

        Args:
            x:
                prediction input, shape: (n_point, n_inp)
                OR
                None

        Kwargs:
            Keyword arguments of black base model

        Returns:
            prediction output, shape: (n_point, n_out)
            or
            None if self.model is None, or not self.ready, or x is None
        """

        if self.model is None or not self.model.ready or x is None:
            self.y = None
        else:
            self.x = np.atleast_2d(x)
            self.y = self.model.predict(self.x, **kwargs)

        return self.y

    def evaluate(self, X: ArrayLike | None, Y: ArrayLike | None,
                 **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluates difference between prediction y(X) and reference Y(X)

        Args:
            X:
                reference input, shape: (n_point, n_inp)

            Y:
                reference output, shape: (n_point, n_out)

        Kwargs:
            None

        Returns:
            metric of evaluation, update of
                'abs' (float): max{|net(x) - Y|} of best training
                'i_abs' (int): index of Y where absolute err is max
                'L2' (float): sqrt{sum{(net(x)-Y)^2}/N} best train
        Note:
            maximum abs index is 1D index,
            e.g. y_abs_max=Y.ravel()[i_abs_max]
        """
        evaluation = init_metric()

        if X is None or Y is None or not self.ready:
            return evaluation

        opt = kwargs.copy()
        if 'x' in opt:
            del opt['x']
        y = self.predict(x=X, **opt)

        if y is not None:
            update_pred_error(evaluation, X, Y, y,
                              silent=kwargs.get('silent', True))

        return evaluation
