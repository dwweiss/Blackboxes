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

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import unittest

from blackboxes.box import Black
from blackboxes.metric import plot_error_bars, plot_hist_loss


def f(x: ArrayLike | None, *c: float) -> ArrayLike:
    """
    Theoretical sub-model

    :param x: input (1D or 2D array of float)
    :param c: tuning parameters
    :return: output (1D or 2D array of float)
    """
  
    c0, c1 = c if len(c) == 2 else 0.1, 0.1
    return np.sin(x) + c0 * x + c1


class TestUM(unittest.TestCase):
    def setUp(self):
        self.save_figures = True

    def tearDown(self):
        pass

    def test1(self):
        print('*** begin_')

        # training data, split into train and validation data in train()
        N = 1000
        absolute_noise = 0.1

        X = np.linspace(-2*np.pi, 2*np.pi, N).reshape(-1, 1)
        Y_tru = np.asfarray([f(item) for item in X])
        Y_nse = Y_tru + np.random.uniform(low=-absolute_noise,
            high=absolute_noise, size=Y_tru.shape)

        # test data
        dx = 0.5 * np.pi
        n = 100
        x = np.linspace(X.min()-dx, X.max()+dx, n).reshape(-1, 1)
        y_tru = np.asfarray([f(item) for item in x])

        neurons = [[10,10,10,10], [10,10,10,10,10],]
        backends = (
            'neurolab',
            'keras',
            'torch',
        )

        all_metrices = []
        phi = Black()
        silent = 0
        plot = 0
        for backend in backends:
            phi.train(X=X, Y=Y_nse,
                      # activation='tanh',
                      activation='sigmoid' if backend == 'neurolab' else 'elu',
                      backend=backend,
                      batch_size=None if backend=='neurolab' else (32, 64, None),
                      early_stop=0,
                      epochs=300,
                      expected=0.5e-3,
                      lr=3e-3,
                      neurons=neurons,
                      output='linear',
                      patience=10,
                      plot=0,
                      silent=silent,
                      tolerated=5e-3,
                      trainer=('adam',),
                      trials=5,
                      verbose=0,
                      )
            all_metrices += phi.metrices

            if phi.ready and plot:
                print('='*40)

                y = phi.predict(x=x, silent=True)
                eval_trn = phi.evaluate(X, Y_tru)
                eval_tst = phi.evaluate(x, y_tru)

                plt.title('mse$^{trn}$: ' f"{1e3*phi.metric['loss']:.2f}e-3, "
                          'mse$^{val}$: ' 
                          f"{1e3*phi.metric['val_loss']:.2f}e-3, "
                          '$L_2^{tst}$: ' f"{1e3*eval_tst['L2']:.2f}e-3 "
                          f"[{phi.metric.get('backend')}]"
                          )
                # plt.ylim(min(-2., Y_nse.min(), y.min()),
                #          max(+2., Y_nse.max(), Y_nse.max()))
                plt.yscale('linear')
                plt.xlim(-0.1 + x.min(), 0.1 + x.max())

                plt.scatter(X, Y_nse, marker='.', c='r', label='trn')
                plt.plot(x, y, c='b', label='prd')
                plt.plot(x, y_tru, linestyle=':', label='tru')

                i_abs_trn = eval_trn['i_abs']
                plt.scatter(X[i_abs_trn], Y_nse[i_abs_trn], marker='o',
                            color='r', s=66, label='max abs trn')

                i_abs_tst = eval_tst['i_abs']
                plt.scatter(x[i_abs_tst], y[i_abs_tst], marker='o',
                            color='b', s=66, label='max abs tst')

                plt.legend(bbox_to_anchor=(1.1, 0), loc='lower left')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.grid()
                plt.show()

        plot_hist_loss(all_metrices, n_best=None,
            expected=phi._model._expected, tolerated=phi._model._tolerated,)

        for n_best in (None, 32):
            plot_error_bars(all_metrices, n_best=n_best,
                expected=phi._model._expected, tolerated=phi._model._tolerated,)

        self.assertTrue(phi.ready)


if __name__ == '__main__':
    unittest.main()
