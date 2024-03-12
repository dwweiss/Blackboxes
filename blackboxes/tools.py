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
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, List, Sequence

__all__ = ['EarlyStop', 'f_rosenbrook', 'f_sine', 'loss_to_str',
           'to_list', 'to_list_of_list']


def f_sine(x: ArrayLike | None, *c: float,
           **kwargs: Any) -> ArrayLike:
    """
    Sine function as demo function f(self, x, c) for single data
    point [y0] = f(x, c)

    recommended plot ranges: [(0, 1), (0, 1)]

    :param x:
            input, shape: (n_inp,)
            or
            None if initial tuning parameters should be returned
    :param c:
            tuning parameters as positional arguments,
            shape: (n_tun,)
    :param kwargs:
            None
    :return:
        output array, shape: (n_out,)
        OR
        initial tuning parameter array if x is None, shape: (n_tun,)
    """
    c_ini = (1., 100., 1., 1.)

    if x is None:
        return c_ini

    if len(c) < len(c_ini):
        c = c_ini

    y0 = (c[0] + c[1] * x[0]) * np.sin(c[2] * x[0]) + c[3] * x[1]

    return [y0]


def f_rosenbrook(x: ArrayLike | None, *c: float,
           **kwargs: Any) -> ArrayLike:
    """
    Rosenbrook function as demo function f(self, x) for single data
    point [y0] = f([x0, x1]) with two coefficients [c0, c1]

    Minimum at f(c0, c0**2) = f(1, 1) = 0 if c0=1 and c1=100
    recommended plot ranges: [(-2, 2), (-1, 3)]

    :param x:
            input, shape: (n_inp,)
            or
            None if initial tuning parameters should be returned
    :param c:
            tuning parameters as positional arguments,
            shape: (n_tun,)
    :param kwargs:
            None
    :return:
        output array, shape: (n_out,)
        OR
        initial tuning parameter array if x is None, shape: (n_tun,)
    """
    c_ini = [1., 100.]

    if x is None:
        return c_ini

    if len(c) < len(c_ini):
        c = c_ini

    y0 = (c[0] - x[0]) ** 2 + c[1] * (x[1] - x[0] ** 2) ** 2

    return [y0]


class EarlyStop:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self._count: int = 0
        self._patience: int = patience
        self._min_delta: float = min_delta
        self._min_loss: float = np.inf

    def reset(self) -> None:
        self._count = 0
        self._min_loss = np.inf

    def break_(self, loss: float) -> bool:
        if loss < self._min_loss:
            self._count = 0
            self._min_loss = loss
        elif loss > self._min_loss + self._min_delta:
            self._count += 1
            if self._count >= self._patience:
                return True
        return False


def loss_to_str(f: float, width: int = 5) -> str:
    """
    :param f: loss
    :param width: width of mantissa
    :return: formatted string: e.g. 12.34e-3
    """
    return f'{1e3*f:{width}.2f}e-3'


def to_list(x: float | int | bool | str | None |
            NDArray |
            Sequence[float | int | bool | str | None]
            ) -> List[float | int | bool | str | None]:
    """
    :param x: scalar or sequence
    :return: list(x)
    """
    if x is None or isinstance(x, (float, int, bool, str)):
        return [x]
    elif isinstance(x, (list, np.ndarray, tuple)):
        return list(x)
    else:
        raise ValueError(f'unsupported {type(x)=}')


def to_list_of_list(x: float | int | bool | str | None |
                    NDArray |
                    Sequence[float | int | bool | str | None] |
                    Sequence[Sequence[float | int | bool | str | None]] |
                    None
                    ) -> List[List[float | int | bool | str | None]] | None:
    """
    :param x: scalar or sequence or sequence of sequence
    :return: list(list(x))
    """
    if x is None:
        return None
    if x is None or isinstance(x, (float, int, bool, str)):
        return [[x]]
    if isinstance(x, (list, np.ndarray, tuple)):
        if all(x is None or isinstance(item, (float, int, bool, str)
                                       ) for item in x):
            return [list(x)]
        else:
            return [list(item) for item in x]
    else:
        raise ValueError(f'unsupported {type(x)=}')
