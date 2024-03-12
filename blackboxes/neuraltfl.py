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

from collections import OrderedDict
import logging
import numpy as np
from numpy.typing import ArrayLike, NDArray
import platform
import subprocess
from typing import Any, Callable, Dict, Sequence, List

try:
    import keras
except ImportError:
    try:
        subprocess.call(['pip', 'install', 'keras'])
        import keras
    except ImportError:
        raise Exception('??? package keras not imported')

from keras import Sequential
from keras.callbacks import (Callback, EarlyStopping,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.layers import Activation, Dense, Input
if platform.system() == 'Darwin':
    # M1/M2 Macs
    from keras.optimizers.legacy import (Adam, Adamax, Nadam, SGD, RMSprop)
else:
    from keras.optimizers import (Adam, Adamax, Nadam, SGD, RMSprop)

from blackboxes.bruteforce import BruteForce

# disable tensorflow log
logging.getLogger('tensorflow').disabled = True


class Neural(BruteForce):
    """
    Wraps neural network implementations from Keras (Tensorflow backend)
        
    Literature:
        https://www.tensorflow.org/tutorials/keras/regression
        https://arxiv.org/abs/1609.04747        
    """

    def __init__(self, f: Callable[..., ArrayLike] = None) -> None:
        super().__init__(f=f)
        self._backend = 'Tfl'

        self._activation_dict = OrderedDict(
            elu=Activation('elu'),  # default layer activation
            #
            relu=Activation('relu'),
            sigmoid=Activation('sigmoid'),
            tanh=Activation('tanh'),
            #
            linear=Activation('linear'),  # output layer activation
        )
        self._trainer_dict = OrderedDict(
            adam=Adam,  # default trainer
            #
            # adadelta = Adadelta,
            # adagrad = Adagrad,
            adamax=Adamax,
            # ftrl = Ftrl,
            nadam=Nadam,
            sgd=SGD,
            rprop=RMSprop,
            rmsprop=RMSprop,  # alias
        )

    def _create_callbacks(self, 
                          epochs: int, 
                          silent: bool, 
                          patience: int,
                          best_net_file: str,
                          **kwargs: Any) -> List[Any]:

        class _PrintDot(Callback):
            def on_epoch_end(self, epochs_, logs) -> None:

                """
                if epochs_ == 0:
                    print('        epochs: ', end='')
                if epochs_ % 25 == 0:
                    print(f'{epochs_} ', end='')
                else:
                    if epochs_ + 1 == epochs:
                        print(f'{epochs_+1} ', end='')
                """
                return

        callbacks = []
        if not silent:
            callbacks.append(_PrintDot())

        if patience > 0:
            callbacks.append(EarlyStopping(monitor='val_loss', mode='auto', 
                             patience=patience, min_delta=1e-4, verbose=0))
        if self._best_net_file:
            callbacks.append(ModelCheckpoint(self._best_net_file, 
                             save_best_only=True, monitor='val_loss',
                             mode='auto'))
        if True:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                             mode='auto', factor=0.666, patience=5,
                             min_delta=0., min_lr=5e-4, verbose=0))
            
        return callbacks

    def _create_net(self,
                    n_inp: int,
                    hiddens: Sequence[int],
                    n_out: int,
                    activation: str,
                    output_activation,
                    X_stats: Sequence[Dict[str, float]]
                    ) -> bool:
        if self._net:
            del self._net

        if output_activation not in ('linear',):
            if not self.silent:
                print(f'!!! {output_activation=} is invalid '
                      '==> change to: linear')
            output_activation = 'linear'

        act_fct = self._activation_dict[activation]
        out_act_fct = self._activation_dict[output_activation]

        self._net = Sequential()
        self._net.add(Input(shape=(n_inp,)))
        for hidden in np.atleast_1d(hiddens):
            self._net.add(Dense(units=hidden, activation=act_fct,))
        self._net.add(Dense(units=n_out, activation=out_act_fct,))

        return True

    def _predict_scaled(self, x_scaled: NDArray,
                        **kwargs: Any) -> NDArray | None:
        return self._net.predict(x_scaled,
                                 **self._kwargs_get(kwargs,
                         ('batch_size', 'verbose')))

    def _set_trainer(self, trainer: str | None,
                     **kwargs: Any) -> str:
        opt = dict(learning_rate=kwargs.get('learning_rate', 0.1))
        if trainer == 'sgd':
            # opt['clip_value'] = kwargs.get('clipvalue', 0.667)
            opt['decay'] = kwargs.get('decay', 1e-6)
            opt['momentum'] = kwargs.get('momentum', 0.8)
            opt['nesterov'] = kwargs.get('nesterov', True)

        optimizer = self._get_trainer(trainer)(**opt)

        opt = {k: kwargs[k] for k in kwargs.keys() if k in
               ('loss_weights', 'sample_weight_mode',
                'target_tensors', 'weighted_metrics')
               }
        self._net.compile(loss='mean_squared_error', optimizer=optimizer,
                          **opt)
        return trainer

    def _train_scaled(self, X: NDArray, Y: NDArray,
                      **kwargs: Any) -> Dict[str, Any]:

        hist = self._net.fit(
            X, Y,
            batch_size=kwargs.get('batch_size', None),
            callbacks=kwargs.get('callbacks', None),
            epochs=kwargs.get('epochs', 300),
            shuffle=False,  # shuffling has been done in super().train()
            validation_data=kwargs.get('validation_data'),
            verbose=kwargs.get('verbose', 0),
        ).history

        return dict(hist_loss=hist['loss'],
                    hist_val_loss=hist['val_loss'],
                    loss=hist['loss'][-1],
                    val_loss=hist['val_loss'][-1]
                    )
