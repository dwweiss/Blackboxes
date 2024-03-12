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
      2024-03-05 DWW
"""

__all__ = ['Neural']

import copy
from collections import OrderedDict
import numpy as np
from numpy.typing import ArrayLike, NDArray
import platform
import subprocess
from typing import Any, Callable, Dict, Sequence

try:
    import torch
except ImportError:
    try:
        subprocess.call(['pip', 'install', 'torch'])
        import torch
    except ImportError:
        raise Exception('??? package torch not imported')

import torch.nn as nn
import torch.optim as optim
# TODO from torch.utils.data import DataLoader, TensorDataset

from blackboxes.bruteforce import BruteForce
from blackboxes.tools import EarlyStop


class Neural(BruteForce):
    """
    Wraps neural network implementation with Torch
    """

    def __init__(self, f: Callable[..., ArrayLike] = None) -> None:
        super().__init__(f=f)
        self._backend: str = 'Tch'

        self._activation_dict = OrderedDict(
            elu=nn.ELU(),  # +++ default activation
            #
            leaky=nn.LeakyReLU(),
            leaky_relu=nn.LeakyReLU(),
            relu=nn.ReLU(),
            sigmoid=nn.Sigmoid(),
            tanh=nn.Tanh(),
            linear=nn.Identity()  # +++ output activation
        )
        self._trainer_dict = OrderedDict(
            adam=optim.Adam,  # +++ default trainer
            #
            # adadelta = Adadelta,
            # adagrad = Adagrad,
            adamax=optim.Adamax,
            # ftrl = Ftrl,
            nadam=optim.NAdam,
            rmsprop=optim.RMSprop,
            rprop=optim.RMSprop,
            sgd=optim.SGD,
        )

        self._optimizer = list(self._trainer_dict.values())[0]

    def _create_net(self,
                    n_inp: int,
                    hiddens: Sequence[int],
                    n_out: int,
                    activation: str,
                    output_activation: str,
                    X_stats: Sequence[Dict[str, float]]) -> bool:

        if self._net:
            del self._net

        act = self._activation_dict[activation]

        self._net = nn.Sequential()
        self._net.append(nn.Linear(n_inp, hiddens[0]))
        self._net.append(act)
        for i in range(1, len(hiddens)):
            self._net.append(nn.Linear(hiddens[i - 1], hiddens[i]))
            self._net.append(act)
        self._net.append(nn.Linear(hiddens[-1], n_out))

        return True

    def _device(self) -> Any:
        if platform.system() == 'Darwin':
            # TODO add acceleration on Apple M1, M2 and M3
            return torch.device('cpu')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            torch.device('cpu')

    def _predict_scaled(self, x_scaled: NDArray,
                        **kwargs: Any) -> NDArray | None:
        return self.to_numpy(self._net(self.to_tensor(x_scaled)))

    def _set_trainer(self, trainer: str | None,
                     **kwargs: Any) -> None:
        self._optimizer = self._trainer_dict[trainer](
            params=self._net.parameters(), lr=kwargs.get('lr', 3e-3))

    def to_numpy(self, x: torch.Tensor) -> NDArray:
        # return t.detach().cpu().numpy()
        return x.cpu().detach().numpy()

    def to_tensor(self, x: NDArray) -> torch.Tensor:
        # return torch.from_numpy(x, device=self._device())
        torch.Tensor()
        return torch.Tensor(x)

    def _train_scaled(self, X: NDArray, Y: NDArray,
                      **kwargs: Any) -> Dict[str, Any]:

        batch_size = kwargs.get('batch_size', None)
        epochs = kwargs.get('epochs', 300)
        self._expected = kwargs.get('expected', 0.5e-3)
        learning_rate = kwargs.get('learning_rate', 3e-3)
        patience = kwargs.get('patience', 15)
        self.silent = kwargs.get('silent', self.silent)
        X_val, Y_val = kwargs.get('validation_data')

        early_stop = EarlyStop(patience=patience, min_delta=1e-5)
        hist_loss, hist_val_loss = [], []
        loss_fct = nn.MSELoss()
        loss = np.inf
        val_loss_best = np.inf
        w_best = None

        X_trn = self.to_tensor(X)
        Y_trn = self.to_tensor(Y)
        X_val = self.to_tensor(X_val)
        Y_val = self.to_tensor(Y_val)

        for epoch in range(epochs):
            if batch_size is not None:
                # TODO show process in console as bars
                # batch_starts = torch.arange(0, len(X_trn), batch_size)
                # with tqdm.tqdm(batch_starts, unit='batch', mininterval=0,
                #                disable=True) as bar:
                for start in np.arange(0, len(X_trn), batch_size):
                    # bar.set_description(f'epoch {epoch}')
                    if 1:  # for start in bar:
                        X_bat = X_trn[start:start + batch_size]
                        Y_bat = Y_trn[start:start + batch_size]

                        y_bat = self._net(X_bat)

                        loss = loss_fct(y_bat, Y_bat)
                        self._optimizer.zero_grad()
                        loss.backward()
                        self._optimizer.step()
                        # bar.set_postfix_str(loss=float(loss))
            else:
                y_trn = self._net(X_trn)
                loss = loss_fct(y_trn, Y_trn)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            self._net.train(False)

            y_trn = self._net(X_trn)
            loss = float(loss_fct(y_trn, Y_trn))
            hist_loss.append(loss)
            y_val = self._net(X_val)
            val_loss = float(loss_fct(y_val, Y_val))
            hist_val_loss.append(val_loss)

            if epoch > 30 and early_stop.break_(loss=loss):
                print(f', Early stop {1e3*val_loss:6.2f}m')
                break
            if val_loss_best > val_loss:
                val_loss_best = val_loss
                w_best = copy.deepcopy(self._net.state_dict())
            if loss < self._expected:
                break

            self._net.train(mode=True)  # sets validation mode

            self._net.load_state_dict(w_best)

        return dict(hist_loss=hist_loss,
                    hist_val_loss=hist_val_loss,
                    loss=loss,
                    val_loss=val_loss_best)
