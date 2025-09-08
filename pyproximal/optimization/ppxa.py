import time
import warnings
from math import sqrt
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pylops.optimization.leastsquares import regularized_inversion
from pylops.utils.backend import get_array_module, to_numpy
from pylops.utils.typing import NDArray

from pyproximal.proximal import L2
from pyproximal.ProxOperator import ProxOperator
from pyproximal.utils.bilinear import BilinearOperator

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


def PPXA(
    prox_ops: List[ProxOperator],
    x0: NDArray | List[NDArray],
    tau: float,
    lmd: float = 1.0,
    weights: NDArray | List[float] | None = None,
    niter: int = 10,
    tol: Optional[float] = None,
    callback: Optional[Callable[..., None]] = None,
    show: bool = False,
) -> Tuple[NDArray, NDArray]:
    r"""Parallel Proximal Algorithm (PPXA) for solving:
        :math:: \min_x  sum_{i=1}^m f_i(x)
        given proximal operators of :math:`f_i`.

    """
    m = len(prox_ops)
    if weights is None:
        weights = np.ones(m) / m

    if isinstance(x0, np.ndarray):
        y = [x0.copy() for _ in range(m)]
        x = x0.copy()
    else:
        assert len(x0) == m
        y = [x0[i].copy() for i in range(m)]
        x = sum(y) / m

    x_old = x.copy()
    for _ in range(niter):

        p = [prox_ops[i].prox(y[i], tau / weights[i]) for i in range(m)]

        pn = sum(weights[i] * p[i] for i in range(m))

        for i in range(m):
            y[i] = y[i] + lmd * (2 * pn - x - p[i])

        x = x + lmd * (pn - x)

        if callback is not None:
            callback(x)

        if np.abs(x - x_old).max() < tol:
            break

        x_old = x

    return x
