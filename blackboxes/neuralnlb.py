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
      2024-02-06 DWW

  Acknowledgement:
      Neurolab is a contribution by E. Zuev (pypi.python.org/pypi/neurolab)
"""

__all__ = ['Neural']

from collections import OrderedDict
import numpy as np
from numpy.typing import ArrayLike, NDArray
import subprocess
from typing import Any, Callable, Dict, List, Sequence

try:
    import neurolab.net as nl
except ImportError:
    try:
        subprocess.call(['pip', 'install', 'neurolab'])
        import neurolab.net as nl
    except ImportError:
        raise Exception('??? package neurolab not imported')

from blackboxes.bruteforce import BruteForce


class Neural(BruteForce):
    """
    References:
        - Recommended training algorithms:
              'rprop': resilient backpropagation (NO REGULARIZATION)
                       wikipedia: 'Rprop'
              'bfgs':  Broyden–Fletcher–Goldfarb–Shanno algorithm,
                       see: scipy.optimize.fmin_bfgs() and wikipedia:
                           'Broyden-Fletcher-Goldfarb-Shanno_algorithm'
        - http://neupy.com/docs/tutorials.html#tutorials
    """

    def __init__(self, f: Callable[..., ArrayLike] = None) -> None:
        super().__init__(f)
        self._activation_dict = OrderedDict(
            sigmoid=nl.trans.LogSig,  # default hidden layer
            #
            tanh=nl.trans.TanSig,
            #
            linear=nl.trans.PureLin,  # output layer
        )

        self._backend = 'Nlb'

        self._trainer_dict = OrderedDict(
            rprop=nl.train.train_rprop,  # +++ default trainer
            #
            bfgs=nl.train.train_bfgs,
            cg=nl.train.train_cg,
            gd=nl.train.train_gd,
            gda=nl.train.train_gda,
            gdm=nl.train.train_gdm,
            gdx=nl.train.train_gdx,
        )

    def _create_net(self, n_inp: int,
                    hiddens: Sequence[int],
                    n_out: int,
                    activation: str,
                    output_activation: str,
                    X_stats: Sequence[Dict[str, float]]) -> bool:

        if self._net:
            del self._net

        minmax = [[x['min'], x['max']] for x in X_stats]
        assert len(minmax) == len(self._X_stats), f'{minmax=}, {self._X_stats=}'
        size = np.append(np.atleast_1d(hiddens), [n_out])
        self._net = nl.newff(minmax, size)

        self._net.transf = self._activation_dict[activation]
        self._net.errorf = nl.error.MSE()
        self._net.outputf = nl.trans.PureLin

        return True

    def _predict_scaled(self,x_scaled: NDArray, 
                        **kwargs: Any) -> NDArray | None:
        return self._net.sim(x_scaled)

    def _set_trainer(self, trainer: str | None, **kwargs: Any) -> str:
        assert self._net is not None
        self._net.trainf = self._get_trainer(trainer)
        return ''

    def _train_scaled(self, X: NDArray, Y: NDArray, 
                      **kwargs: Any) -> Dict[str, Any]:
        epochs = kwargs.get('epochs', 300)
        regularization = kwargs.get('regularization', 1.)
        XY_val = kwargs.get('validation_data')

        hist_loss: List[float] = []
        if self._net.trainf == nl.train.train_rprop:
            hist_loss = self._net.train(X, Y,
                                        epochs=epochs,
                                        goal=self._expected,
                                        show=0,
                                        )
        else:
            for i in range(5+1):
                # repetition until success of training because train()
                # crashes sometimes if trainer != 'rprop'
                self._net.init()
                hist_loss = self._net.train(X, Y,
                                            epochs=epochs,
                                            goal=self._expected,
                                            rr=regularization,
                                            show=0,
                                            )
                if len(hist_loss):
                    break
        loss: float = hist_loss[-1]

        if XY_val is None:
            val_loss = np.nan
        else:
            x_val_scaled = self._scale(XY_val[0], self._X_stats,
                                       self._min_max_scale)
            y_val = self._predict_scaled(x_val_scaled)
            Y_val = XY_val[1]
            val_loss = np.mean(np.square(y_val - Y_val))

        # neurolab.train() does not return MSE of the validation
        # MSE history of validation is filled with NaN and completed with
        # MSE of validation data and its prediction
        hist_val_loss: List[float] = [np.nan] * (len(hist_loss)-1) + [val_loss]
        
        return dict(hist_loss=hist_loss,
                    hist_val_loss=hist_val_loss,
                    loss=loss,
                    val_loss=val_loss,
                    )
