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
      2024-03-12 DWW
"""

__all__ = ['BruteForce']

import inspect
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
import sys
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from blackboxes.metric import (init_metric, plot_error_bars, plot_hist_loss,
                               update_pred_error)
from blackboxes.tools import to_list, to_list_of_list


class BruteForce(object):
    """
    - Encapsulation of implementation of empirical models
    - Brute force search of best model configuration
    - Graphic presentation of training history

    Example of training and prediction of neural network:
    
        one-liner:
            y = Neural()(X=X, Y=Y, x=x, neurons=[6,4], trainer='adam')

        compact:
            phi = Neural()
            y = phi(X=X, Y=Y, x=x, neurons=[6,4], trainer='adam')

        expanded:
            phi = Neural()
            metric = phi.train(X=X, Y=Y, neurons=[6,4], trainer='adam')
            loss = metric['loss']
            y = phi(x=x)

        print(phi.ready)
        print(phi.metric)
        print(phi.metric['loss'])
        phi.plot()

    Literature:
        Activation:
            https://ml-cheatsheet.readthedocs.io/en/latest/
                activation_functions.html
        Regularization:             
            https://www.analyticsvidhya.com/blog/2018/04/
                fundamentals-deep-learning-regularization-techniques/   
            https://machinelearningmastery.com/how-to-stop-training-deep-
                neural-networks-at-the-right-time-using-early-stopping/ 
    """

    def __init__(self, f: Callable[..., ArrayLike] | None = None) -> None:
        """
        Args:
            f:
                theoretical sub-model as method f(self, x) or function f(x)
                OR
                None if f is not relevant
        """
        self._activation_dict: Dict[str, Any] | None = None
        self._backend: str = ''
        self._best_net_file: str = ''
        self._expected: float = 1.0e-3  # stop trials if loss < expected
        self.f = f                                   # theoretical model
        self._metric: Dict[str, Any] = init_metric()     # actual metric
        self._metrices: List[Dict[str, Any]] = []         # all metrices
        self._min_max_scale: bool = True   # is data normal distributed?
        self._net: Any | None = None                    # neural network
        self._ready: bool = False                   # is net is trained?
        self._scale_margin: float = 0.1      # scaled margin for min-max
        self._silent: bool = False   # if True, then no print to console
        self._tolerated: float = 5e-3  # if exceeded, self.ready = False
        self._trainer_dict: Dict[str, Any] | None = None  # trainer pool

        self._X: NDArray | None = None    # training inp (n_point,n_inp)
        self._Y: NDArray | None = None         # target (n_point, n_out)
        self._x: NDArray | None = None     # pred. input (n_point,n_inp)
        self._y: NDArray | None = None     # prediction (n_point, n_out)
        self._x_keys: List[str] = []         # x labels, shape: (n_inp,)
        self._y_keys: List[str] = []         # y labels, shape: (n_out,)
        self._X_stats: List[Dict[str, float]] = []     # shape: (n_inp,)
        #          dictionary with mean, std, min and max of all columns
        self._Y_stats: List[Dict[str, float]] = []     # shape: (n_out,) 
        #          dictionary with mean, std, min and max of all columns

        plt.rcParams.update({'font.size': 10})              # axis fonts
        plt.rcParams['legend.fontsize'] = 8            # fonts in legend

    @property
    def backend(self) -> str:
        return self._backend

    def _create_callbacks(self,
                          epochs: int,
                          silent: bool,
                          patience: int,
                          best_net_file: str,
                          **kwargs: Any) -> List[Any]:
        """
        Args:
            epochs:
                maximum number of epochs

            silent:
                If True, then no print to console

            patience:
                number of epochs before employing early stopping

            best_net_file:
                name of file for storing best network during training

        Kwargs:
            spare arguments

        Returns:
            List of a relevant callbacks for network training

        Note:
            This method should be overwritten in child classes
        """
        return []

    def _create_net(self,
                    n_inp: int, 
                    hiddens: Sequence[int], 
                    n_out: int, 
                    activation: str,
                    output_activation: str,
                    X_stats: Sequence[Dict[str, float]]) -> bool:
        """
        Args:
            n_inp:
                number of input neurons

            hiddens:
                list of number of neurons of hidden layers 

            n_out:
                number of output neurons
                
            activation:
                identifier of activation function of hidden layers

            output_activation:
                identifier of activation function of hidden layers
                
            X_stats:
                list of dictionary of 'min' & 'max' value of every input
                
        Returns:
            multi-layer perceptron with defined activation functions
            OR
            None

        Note:
            This method should be overwritten in child classes
        """
        raise NotImplementedError

    @property
    def f(self) -> Callable[..., ArrayLike] | None:
        return self._f

    @f.setter
    def f(self, value: Callable[..., ArrayLike] | None) -> None:
        if value is not None:
            first_arg = list(inspect.signature(value).parameters.keys())[0]
            if first_arg == 'self':
                value = value.__get__(self, self.__class__)
        self._f = value

    def _get_activation(self, activation: str | None) -> Any | List[str]:
        """
        Args:
            activation:
                key of activation function

        Returns:
            activation function if activation is not None
            OR
            list of all available activation functions if activation is
            None

        Note:
            This method should be overwritten in child classes
        """

        dic = self._activation_dict
        if activation is None:
            return list(dic.keys())
        else:
            return dic.get(activation, list(dic.values())[0])

    def _get_trainer(self, trainer: str | None) -> Any | List[str]:
        """
        Args:
            trainer:
                key of trainer

        Returns:
            trainer function if trainer is not None
            OR
            list of all available trainers keys if trainer is None

        Note:
            This method should be overwritten in child classes
        """
        dic = self._trainer_dict
        if trainer is None:
            return list(dic.keys())
        else:
            return dic.get(trainer, list(dic.values())[0])

    @property
    def has_backend(self) -> bool:
        return self._backend in sys.modules

    def _predict_scaled(self, x_scaled: NDArray,
                        **kwargs: Any) -> NDArray | None:
        """
        Args:
            x_scaled:
                scaled prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

        Kwargs:
            additional keyword arguments of actual backend

        Returns:
            scaled prediction y = net(x_scaled), shape: (n_point, n_out)
            OR
            None

        Note:
            This method should be overwritten in child classes
        """
        raise NotImplementedError

    def _set_trainer(self, trainer: str | None, **kwargs: Any) -> str:
        """
        Sets the actual trainer out of valid trainer_pool

        Args:
            trainer:
                key of trainer

        Returns:
            actual trainer

        Note:
            This method should be overwritten in child classes
        """
        raise NotImplementedError

    def _train_scaled(self, X: NDArray, Y: NDArray,
                      **kwargs: Any) -> Dict[str, Any]:
        """
        Args:
            X:
                scaled training input, shape: (n_point, n_inp)
            Y:
                scaled target, shape: (n_point, n_out)

        Kwargs:
            implementation-specific trainer options

        Returns:
            dictionary of history with keys 'hist_loss' and
            'hist_val_loss' containing lists of loss values
            the dictionary entries 'loss' and 'val_loss'
            contain the last elements of 'hist_loss and
            'hist_val_loss'
        """
        raise NotImplementedError

    @property
    def metric(self) -> Dict[str, Any]:
        return self._metric

    @property
    def metrices(self) -> List[Dict[str, Any]]:
        return self._metrices

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def silent(self) -> bool:
        return self._silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._silent = value

    @property
    def X(self) -> NDArray | None:
        return self._X

    @property
    def x(self) -> NDArray | None:
        return self._x

    @property
    def Y(self) -> NDArray | None:
        return self._Y

    @property
    def y(self) -> NDArray | None:
        return self._y

    def __call__(self,
                 X: ArrayLike | None = None,
                 Y: ArrayLike | None = None,
                 x: ArrayLike | None = None,
                 **kwargs: Any) -> Dict[str, Any] | NDArray | None | Any:
        """
        - Trains neural network if X is not None and Y is not None
        - Sets self.ready to True if training is successful
        - Predicts y for input x if x is not None and self.ready is True
        - Plots history of training and comparison of train and test data
        - Stores best result
        - After training, class keeps the best training result

        Args:
            X:
                training input, shape: (n_point, n_inp)
                shape: (n_point,) is tolerated
                default: self.X

            Y:
                training target, shape: (n_point, n_out)
                shape: (n_point,) is tolerated
                default: self.Y

            x:
                prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated
                default: self.x

        Kwargs:
            keyword arguments, see: train() and predict()

        Returns:
            metric of best training trial if X and Y are not None,
                see self.train()
            OR
            prediction of net(x) if x is not None and self.ready,
                2D array with shape: (n_point, n_out)
            OR
            empty metric
                if x is None and ((X, Y are None) or (not self.ready))

        Note:
            - X.shape[1] must equal x.shape[1]
            - Shape of X, Y and x is corrected to (n_point, n_inp/n_out)
            - References to X, Y, x and y are stored as self.X, self.Y,
              self.x, self.y, see self.train() and self.predict()
        """
        if X is not None and Y is not None:
            self._metric = self.train(X=X, Y=Y, **kwargs)

        if x is not None and self.ready:
            return self.predict(x=x, **kwargs)

        return self.metric

    def _create_default_hiddens(self, n_inp: int,
                                n_out: int) -> List[List[int]] | None:
        """
        initializes randomly the weights of the neural network

        Args:
            n_inp:
                number of inputs

            n_out:
                number of outputs

        Returns:
            List of list of number of hidden neurons per layer
                hiddens[j][k], j=0..n_variation-1, k=1..n_hidden_layer
            OR
            None if n_inp or n_out is None
        """
        if n_inp is None or n_out is None:
            return None

        min_hid_layers = 1
        max_hid_layers = 5
        min_layer_size = max(3, n_inp, n_out)
        max_layer_size = min(8, min_layer_size * 2)

        return [[i] * j
                for j in range(min_hid_layers, max_hid_layers + 1)
                for i in range(min_layer_size, max_layer_size + 1)]

    def _descale(self, X: ArrayLike | None,
                 X_stats: Sequence[Dict[str, float]],
                 min_max_scale: bool) -> NDArray | None:
        """
        Descales array:
          1. X_real_world = min(X) + X_scaled * (max(X) - min(X))
                                                        if min_max_scale
          2. X_real_world = X_scaled * std(X) + mean(X)        otherwise

        Args:
            X:
                scaled 2D array of shape (n_point, n_inp/n_out)

            X_stats:
                list of dictionary of statistics of every column of X
                (keys: 'min', 'max', 'mean', 'std')

            min_max_scale:
                if True, X has been scaled in [0, 1] range, else
                X has been normalized with mean and standard deviation

        Returns:
            real-world 2D array, shape: (n_point, n_inp/n_out)
            or
            None if X is None
        """
        if X is None:
            return None

        X = np.asfarray(X)
        X_real_world = np.zeros(X.shape)

        if min_max_scale:
            for j in range(X_real_world.shape[1]):
                X_real_world[:,j] = (X_stats[j]['min']
                    + X[:,j] * (X_stats[j]['max'] - X_stats[j]['min']))
        else:
            for j in range(X_real_world.shape[1]):
                X_real_world[:,j] = (X[:,j] * (X_stats[j]['std'])
                    + X_stats[j]['mean'])

        return X_real_world

    def evaluate(self, X_ref: ArrayLike | None, Y_ref: ArrayLike | None,
                 **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluates difference between prediction y(X_ref) and
        given reference Y_ref(X_ref)

        Args:
            X_ref:
                real-world reference input, shape: (n_point, n_inp)

            Y_ref:
                real-world reference output, shape: (n_point, n_out)

        Kwargs:
            silent (bool):
                if True then print of norm is suppressed
                default: self.silent

            plot (bool):
                if true, plots comparison of train data with prediction
                default: False

            additional keyword arguments, see: self._predict_scaled()

        Returns:
            metric of evaluation
            or
            default_metric, see metric.init_metric()

        Note:
            max. abs index is 1D index, y_abs_max=Y.ravel()[i_abs_max]
        """
        metric = init_metric()

        if X_ref is None or Y_ref is None or not self.ready:
            return metric

        y = self.predict(x=X_ref, **kwargs)

        metric = update_pred_error(metric, X_ref, Y_ref, y,
                                   silent=False)
        # TODO                            kwargs.get('silent', self.silent))
        return metric

    def _scale(self,
               X: ArrayLike | None,
               X_stats: Sequence[Dict[str, float]],
               min_max_scale: bool) -> NDArray | None:
        """
        Scales array X:
          1. X_scaled = (X - min(X)) / (max(X) - min(X)) if min_max_scale
          2. X_scaled = (X - mean(X)) / std(X)                  otherwise

        Args:
            X:
                real-world 2D array of shape (n_point, n_inp/n_out)

            X_stats:
                list of dictionary of statistics of every column of X
                (keys: 'min', 'max', 'mean', 'std')

            min_max_scale:
                if True, X has been scaled in [0, 1] range, else
                X has been normalized with mean and standard deviation

        Returns:
            scaled 2D array, shape: (n_point, n_inp/n_out)
            or
            None if X is None

        Note:
            see super().set_XY for the lower und upper bound of X_stat
        """
        if X is None:
            return None

        X_scaled = np.zeros(np.shape(X))

        if min_max_scale:
            for j in range(X_scaled.shape[1]):
                X_scaled[:, j] = (X[:, j] - X_stats[j]['min']) / \
                                 (X_stats[j]['max'] - X_stats[j]['min'])
        else:
            for j in range(X_scaled.shape[1]):
                X_scaled[:, j] = (X[:, j] - X_stats[j]['mean']) / \
                                 (X_stats[j]['std'])

        return X_scaled

    def set_XY(self,
               X: ArrayLike | None,
               Y: ArrayLike | None,
               x_keys: Sequence[str] | None = None,
               y_keys: Sequence[str] | None = None) -> None:
        """
        - Stores training input X and training target Y as self.X and
          self.Y
        - converts self.X and self.Y to 2D arrays
        - transposes self.X and self.Y if n_point < n_inp/n_out

        Args:
            X:
                training input, shape: (n_point, n_inp)
                shape: (n_point,) is tolerated

            Y:
                training target, shape: (n_point, n_out)
                shape: (n_point,) is tolerated

            x_keys:
                list of column keys for data selection
                use self._x_keys keys if x_keys is None
                default: ['x0', 'x1', ... ]

            y_keys:
                list of column keys for data selection
                use self._y_keys keys if y_keys is None
                default: ['y0', 'y1', ... ]
        """
        if X is None or Y is None:
            self._X = None
            self._Y = None

        self._X = np.atleast_2d(X)
        self._Y = np.atleast_2d(Y)

        if self._X.shape[0] < self._X.shape[1]:
            self._X = self._X.transpose()
        if self._Y.shape[0] < self._Y.shape[1]:
            self._Y = self._Y.transpose()

        assert self._X.shape[0] == self._Y.shape[0], \
            f'input arrays incompatible [{self._X.shape[0]}] vs. ' \
            f'[{self._Y.shape[0]}]\n{self._X=}\n{self._Y=}'

        # min, max, mean and standard deviation of all columns of X and Y
        self._X_stats = [{'mean': c.mean(), 'std': c.std(),
                          'min': c.min(), 'max': c.max()} for c in self._X.T]
        self._Y_stats = [{'mean': c.mean(), 'std': c.std(),
                          'min': c.min(), 'max': c.max()} for c in self._Y.T]

        # 10% safety margin in distance between lower and upper bound
        for array in (self._X_stats, self._Y_stats):
            for column in array:
                margin = self._scale_margin * (column['max'] - column['min'])
                column['min'] -= margin
                column['max'] += margin

                # avoid zero division in normalization of X and Y
        for stats in (self._X_stats, self._Y_stats):
            for col in stats:
                if np.isclose(col['std'], 0.0):
                    col['std'] = 1e-10

        # set default keys 'xi' and 'yj' if not x_keys or y_keys
        if not x_keys:
            self._x_keys = [f'x{i}' for i in range(self._X.shape[1])]
        else:
            self._x_keys = x_keys
        if not y_keys:
            self._y_keys = [f'y{i}' for i in range(self._Y.shape[1])]
        else:
            self._y_keys = y_keys

    def _shuffle(self, X: ArrayLike| None, Y: ArrayLike | None
                 ) -> Tuple[NDArray, NDArray] | None:
        """
        shuffles unison two 2D arrays
        
        Args:
            X:
                scaled 2D array of shape (n_point, n_inp)

            Y:
                scaled 2D array of shape (n_point, n_out)
                                
        Returns: 
            shuffled 2D arrays, shapes: (n_point, n_inp), 
                                        (n_point, n_out)
            OR
            None if X or Y is None
        """
        if X is None or Y is None:
            return None

        X = np.asfarray(X)
        Y = np.asfarray(Y)
        p = np.random.permutation(np.shape(X)[0])

        return X[p, :], Y[p, :]

    def _kwargs_del(self, kwargs: Dict[str, Any],
                    exclude: str | Sequence[str]) -> Dict[str, Any]:
        """
        Creates a copy of kwargs exclusive keys given

        Returns:
            copy of kwargs excluding values with keys given by 'exclude'
        """
        if isinstance(exclude, str):
            exclude = [exclude]

        return {k: kwargs[k] for k in kwargs.keys() if k not in exclude}

    def _kwargs_get(self, kwargs: Dict[str, Any],
                    include: str | Sequence[str]) -> Dict[str, Any]:
        """
        Creates a copy of members of kwargs which keys given

        Returns:
            copy of kwargs with values of kwargs which have keys
            contained in 'include'
        """
        if isinstance(include, str):
            include = [include]

        return {k: kwargs[k] for k in kwargs.keys() if k in include}

    def n_inp(self) -> int | None:
        """
        Returns:
            number of input of training input (columns)
            OR
            None if net is not trained
        """
        if self._X is None:
            return None

        return self._X.shape[1]

    def n_out(self) -> int | None:
        """
        Returns:
            number of output of target (columns)
            OR
            None if net is not trained
        """
        if self._Y is None:
            return None

        return self._Y.shape[1]

    def n_point(self, arr) -> int | None:
        """
        Returns:
            number of data points (rows)
            OR
            None if arr is None
        """
        if arr is None:
            return None

        return np.shape(arr)[0]

    def plot(self, **kwargs: Any) -> None:
        if kwargs.get('plot', 1):
            self._plot_network()
            self._plot_train_vs_pred()

    def _plot_network(self, file: str = '') -> None:
        pass

    def _plot_train_vs_pred(self, **kwargs: Any) -> None:
        """
        Kwargs:
            see self._predict_scaled(x, **kwargs)
        
        Note:
            This method is called by self.predict(). Thus, this method 
            calls self._predict_scaled() instead of self.predict()
        """

        X_scaled = self._scale(self.X, self._X_stats, self._min_max_scale)
        Y_prd_scaled = self._predict_scaled(X_scaled, **kwargs)
        Y_prd = self._descale(Y_prd_scaled, self._Y_stats, self._min_max_scale)

        if Y_prd is None:
            if not self.silent:
                print('??? plot train vs pred: predict() returned None')
            return 

        X_, Y_, Y_prd_ = self.X[:,0], self.Y[:,0], Y_prd[:,0]
        dY_ = (Y_prd - self.Y)[:,0]

        plt.title(f'Training data versus prediction [{self.backend}]')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(X_, Y_, '.', c='r', label='train')
        plt.plot(X_, Y_prd_, '.', c='b', label='pred')
        plt.legend()
        DY = self.Y.max() - self.Y.min()
        plt.ylim([self.Y.min() - 0.5 * DY, 2 * self.Y.max() + 0.5 * DY])
        plt.grid()
        plt.show()
            
        plt.title(f'Prediction minus target [{self.backend}]')
        plt.xlabel('$x$')
        plt.ylabel(r'$\Delta y = \varphi(X) - Y$')
        plt.plot(X_, dY_, '.')
        plt.grid()
        plt.show()

        plt.title(f'Target versus prediction [{self.backend}]')
        plt.xlabel('target $Y$')
        plt.ylabel('prediction $y$')
        plt.plot(Y_, Y_, '-', label='$Y(X)$')
        plt.plot(Y_, Y_prd_, '.', label=r'$\varphi(X)$')
        plt.legend()
        plt.grid()
        plt.show()

    def predict(self, x: ArrayLike | None,
                **kwargs: Any) -> NDArray | None:
        """
        Executes the network if it is ready

        - Reshapes and scales real-world input x,
        - Executes network,
        - Rescales the scaled prediction y,
        - Stores x as self.x

        Args:
            x:
                real-world prediction input, shape: (n_point, n_inp)
                shape: (n_inp,) is tolerated

        Kwargs:
            plot (bool):
                if true, plots comparison of train data with prediction
                default: False

            additional keyword arguments, see: self._predict_scaled()

        Returns:
            real-world prediction y = net(x)
            or
            None if x is None or not self.ready or
                x shape is incompatible

        Note:
            - Shape of x is corrected to: (n_point, n_inp) if x is 1D
            - Input x and output net(x) are stored as self.x and self.y
        """
        if x is None or not self.ready or self.n_inp() is None:
            self._x, self._y = None, None
            return None

        x = np.asfarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] != self.n_inp():
            x = np.transpose(x)
        if x.shape[1] != self.n_inp():
            print('??? incompatible input {x.shape=}')
            self._x, self._y = None, None
            return None

        self._x = x
        x_scaled = self._scale(self.x, self._X_stats,
                               min_max_scale=self._min_max_scale)

        y_scaled = self._predict_scaled(x_scaled, **kwargs)

        if y_scaled is None:
            if not self.silent:
                print('??? predict: y_scaled is None')
            return None

        self._y = self._descale(y_scaled, self._Y_stats,
                                min_max_scale=self._min_max_scale)

        if kwargs.get('plot', True):
            self._plot_train_vs_pred()

        return self._y

    def train(self, X: ArrayLike | None, Y: ArrayLike | None,
              **kwargs: Any) -> Dict[str, Any] | None:
        """
        Trains model, stores X and Y as self.X and self.Y, and stores
        result of best training trial as self.metric

        [Keras only] see 'patience' for deactivation of early stopping

        Args:
            X:
                training input (real-world), shape: (n_point, n_inp)
                shape: (n_point,) is tolerated
                default: self.X

            Y:
                training target (real-world), shape: (n_point, n_out)
                shape: (n_point,) is tolerated
                default: self.Y

        Kwargs:
            activation (str or list of str):
                activation function of hidden layers
                  'elu': alpha*(exp(x)-1) if x<=0 else x  [not Neurolab]
                  'leakyrelu': max(0,x) - alpha*max(0,-x) [not Neurolab]
                  'linear': x (unmodified input)
                  'relu': max(0, x)                       [not Neurolab]
                  'sigmoid': (LogSig): 1 / (1 + exp(-z))
                  'tanh': (TanSig): tanh(x) or
                default: 'sigmoid'

            backend (str):
                backend identifier ('neurolab, 'keras', 'torch' etc.)

            batch_size (int or None) [not Neurolab] :
                see keras.model.predict()
                default: None

            epochs (int):
                max number of epochs of single trial
                default: 250

            expected (float):
                limit of error for stop of training (0. < goal < 1.)
                default: 5e-4
                [identical with 'goal', 'expected' super-seeds 'goal']

            goal (float) [Neurolab only]:
                limit of error for stop of training (0. < goal < 1.)
                default: 5e-4
                [note: MSE of 1e-4 corresponds to L2-norm of 1e-2]
                [identical with 'expected', 'expected' super-seeds 'goal']

            learning_rate (float or list of float) [Keras only]:
                learning rate of optimizer
                default: None

            momentum (float) [Keras only]:
                momentum of optimizer
                default: None

            neurons (list of int, or list of list of int):
                array of number of neurons in hidden layers

            output (str or None):
                activation function of output layer, see 'activation'
                default: 'linear'

            patience (int) [Keras only]:
                controls early stopping of training
                if patience is <= 0, then early stopping is deactivated
                default: 30

            plot (int):
                controls frequency of plotting progress of training
                    100: plot after every trial
                    1: (plot of all trails)
                    0: no plots

            regularization (float) [Neurolab only]:
                regularization rate (sum of all weights is added to
                cost function of training, 0. <= regularization <= 1.)
                default: 0. (no effect of sum of all weights)
                [same as 'rr']
                [note: neurolab trainer 'bfgs' ignores 'rr' argument]

            rr (float) [Neurolab only]:
                [same as 'regularization']

            show (int) [Neurolab only]:
                control of information about training, if show>0: print
                default: epochs // 10
                [argument 'show' super-seeds 'silent' if show > 0]

            silent (bool):
                if True then no information is sent to console
                default: self.silent
                [Neurolab] argument 'show' super-seeds 'silent' if show > 0

            tolerated (float):
                limit of error for declaring network as ready
                default: 5e-3
                [note: MSE of 1e-4 corresponds to L2-norm of 1e-2]

            trainer (str or list of str):
                if 'all' or None, then all training methods are assigned
                default: 'auto' ==> ['adam', 'rprop']

            trials (int):
                maximum number of training trials
                default: 3

            validation_split (float) [Keras only] :
                share of training data excluded from training
                default: 0.25

            verbose (int):
                controls print of progress of training and prediction
                default: 0 (no prints)
                ['silent' supersedes level of 'verbose']

        Returns:
            metric of best training trial
            OR
            None if X or Y is None

        Note:
            - self.f is optional theoretical sub-model
            - self._X and self._Y is training data (real-world)
            - self._x is prediction input (real-world)
            - The best network is stored as self._net
            - If training fails, then self.ready is False
        """
        if X is None or Y is None:
            return None

        self.set_XY(X, Y)

        activations = kwargs.get('activation')
        if activations is None:
            activations = list(self._activation_dict.keys())[0]
        if activations == 'all':
            activations = list(self._activation_dict.keys())[:-2]
        activations = to_list(activations)

        activation_output = kwargs.get('output', 'linear')

        all_hiddens = kwargs.get('neurons')
        if all_hiddens is None or not all_hiddens:
            all_hiddens = self._create_default_hiddens(self.n_inp(),
                                                       self.n_out())
        all_hiddens = to_list_of_list(all_hiddens)

        if self._backend != 'neurolab':
            batch_sizes = to_list(kwargs.get('batch_size'))
        else:
            batch_sizes = [None]
        epochs = kwargs.get('epochs', 300)
        self._expected = kwargs.get('expected', 0.5e-3)
        # learning_rate = kwargs.get('learning_rate', 0.1)
        patience = kwargs.get('patience', 25)
        regularization = kwargs.get('regularization', 1.)
        self._silent = kwargs.get('silent', self.silent)
        self._tolerated = kwargs.get('tolerated', 5e-3)

        trainers = kwargs.get('trainer')
        if trainers is None:
            trainers = list(self._trainer_dict)[0]
        if trainers == 'all':
            trainers = list(self._trainer_dict)
        trainers = to_list(trainers)

        trials = kwargs.get('trials', 5)
        validation_split = kwargs.get('validation_split', 0.20)
        verbose = 0 if self.silent else kwargs.get('verbose', 0)

        if not self.silent:
            print('--- overview of hiddens')
            for layer in all_hiddens:
                print(f'    {len(layer)} layer(s) ', end='')
                for n in layer:
                    print(f"  {n * '='}", end='')
                print()

        best_trial: Dict[str, Any] | None = None
        callbacks = self._create_callbacks(epochs, self.silent, patience,
                                           self._best_net_file)
        self._metrices = []
        stop_early: bool = False

        # TODO self._min_max_scale = not (activation_output == 'tanh' \
        # and all(a == 'tanh' for a in all_act_hid))
        self._min_max_scale = True
        X_scaled = self._scale(self.X, self._X_stats,
                               min_max_scale=self._min_max_scale)
        Y_scaled = self._scale(self.Y, self._Y_stats,
                               min_max_scale=self._min_max_scale)

        # keras.fit() would shuffle after splitting. Therefore,
        # shuffling is done here AND the shuffle argument is False
        # when calling keras.fit()
        if kwargs.get('shuffle', True):
            X_scaled, Y_scaled = self._shuffle(X_scaled, Y_scaled)
        if 0. < validation_split <= 1.:
            n_trn = int(X_scaled.shape[0] * (1. - validation_split))
            X_trn, Y_trn = X_scaled[:n_trn], Y_scaled[:n_trn]
            X_val, Y_val = X_scaled[n_trn:], Y_scaled[n_trn:]
        else:
            X_trn, Y_trn = X_scaled, Y_scaled
            X_val, Y_val = X_trn, Y_trn
        XY_val_scaled = (X_val, Y_val)

        for hiddens in all_hiddens:
            if not self.silent:
                print(f'--- hidden layers={hiddens}')

            for activation in activations:
                if not self.silent:
                    print(f'    activation: {activation} & {activation_output}')

                for trainer in trainers:
                    if not self.silent:
                        print(f'        trainer: {trainer}')

                    for batch_size in batch_sizes:
                        if not self.silent:
                            print(f'        batch_size: {batch_size}')

                        for trial in range(trials):
                            if not self.silent:
                                print(f'        trial{trial:>3}', end='')

                            self._create_net(self.n_inp(), hiddens,
                                             self.n_out(), activation,
                                             activation_output, self._X_stats)
                            self._set_trainer(trainer,
                                              **self._kwargs_del(kwargs, 'trainer'))

                            # training with early stopping, see callbacks
                            start_time = time.time()
                            hist = self._train_scaled(
                                X=X_trn, Y=Y_trn,
                                batch_size=batch_size,
                                callbacks=callbacks,
                                epochs=epochs,
                                expected=self._expected,
                                regularization=regularization,
                                shuffle=False,  # shuffling was done above
                                validation_data=XY_val_scaled,
                                verbose=verbose,
                            )
                            trial_time = time.time() - start_time
                            # metric of actual trial without values from 'hist'
                            actual_trial = {
                                'activation': activation,
                                'backend': self.backend,
                                'batch_size': batch_size,
                                'epochs': len(hist['hist_loss']),
                                'i_hist': len(self._metrices),
                                'net': self._net,
                                'neurons': hiddens,
                                'time': trial_time,
                                'trainer': trainer,
                                'trial': trial,
                            }

                            # add values of loss, val_loss, hist_loss, and
                            # hist_val_loss from 'hist' to actual_trial
                            actual_trial.update(hist)

                            # adds actual trial to list of all trials
                            self._metrices.append(actual_trial)

                            # updates best history
                            if (best_trial is None or
                                    best_trial['loss'] > actual_trial['loss']):
                                best_trial = actual_trial

                            if kwargs.get('plot', 0) >= 100:
                                plot_hist_loss(self._metrices)

                            # stop training when goal is achieved
                            if (kwargs.get('early_stop', True) and
                                    best_trial is not None and
                                    best_trial['loss'] < self._expected):
                                stop_early = True
                                print(' ==> early stop of multiple trials,\n',
                                      f' ==> {1e3 * best_trial["loss"]:5.2f}e-3 / '
                                      f'{1e3 * best_trial["val_loss"]:6.2f}e-3 '
                                      f'[{best_trial["epochs"]:>3}]')

                            if not self.silent:
                                print(f'{" " * 15}: '
                                      f'{1e3 * actual_trial["loss"]:5.2f}e-3 / '
                                      f'{1e3 * actual_trial["val_loss"]:6.2f}e-3 '
                                      f'[{actual_trial["epochs"]:>3}]')

                            if stop_early:
                                break
                        if stop_early:
                            break
                    if stop_early:
                        break
            if stop_early:
                break

        self._metric = best_trial
        self._net = best_trial['net']
        self._ready = (best_trial['loss'] <= self._tolerated)

        if kwargs.get('plot', 0):
            self._plot_network()
            plot_hist_loss(self._metrices)
            plot_error_bars(self.metrices)

        return self.metric
