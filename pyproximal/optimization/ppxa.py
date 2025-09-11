from typing import Callable, List, Optional

import numpy as np

from pylops.utils.typing import NDArray
from pylops.utils.backend import get_array_module
from pyproximal.ProxOperator import ProxOperator


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
) -> NDArray:
    r"""Parallel Proximal Algorithm (PPXA) for solving:
        :math:: \min_x  sum_{i=1}^m f_i(x)
        given proximal operators of :math:`f_i`.

    """
    ncp = get_array_module(x0)

    m = len(prox_ops)
    if weights is None:
        w = ncp.full(m, 1. / m)
    else:
        w = ncp.asarray(weights)

    if isinstance(x0, list) or x0.ndim == 2:
        y = ncp.asarray(x0)
    else:
        y = ncp.full((m, x0.size), x0)

    x = np.mean(y, axis=0)
    x_old = x.copy()
    for _ in range(niter):

        p = ncp.stack([prox_ops[i].prox(y[i], tau / w[i]) for i in range(m)])

        pn = np.sum(w[:, None] * p, axis=0)

        y = y + lmd * (2 * pn - x - p)

        x = x + lmd * (pn - x)

        if callback is not None:
            callback(x)

        if np.abs(x - x_old).max() < tol:
            break

        x_old = x

    return x
