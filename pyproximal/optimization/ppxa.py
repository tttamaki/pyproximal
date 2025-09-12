import time
from typing import Callable, List, Optional

import numpy as np
from pylops.utils.typing import NDArray
from pylops.utils.backend import get_array_module, to_numpy

from pyproximal.ProxOperator import ProxOperator


def PPXA(  # pylint: disable=invalid-name
    prox_ops: List[ProxOperator],
    x0: NDArray | List[NDArray],
    tau: float,
    eta: float = 1.0,
    weights: NDArray | List[float] | None = None,
    niter: int = 1000,
    tol: Optional[float] = 1e-7,
    callback: Optional[Callable[..., None]] = None,
    show: bool = False,
) -> NDArray:
    r"""Parallel Proximal Algorithm (PPXA) for solving
        :math:: \min_x  sum_{i=1}^m f_i(x)
        given proximal operators of :math:`f_i`.

    Parameters
    ----------
    prox_ops : :obj:`List[ProxOperator]`
        A list of proximable functions :math:`f_1, \ldots, f_m`.
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight
    eta : :obj:`float`, optional
        Relaxation parameter (must be between 0 and 2, 0 excluded).
    weights : :obj:`np.ndarray` or :obj:`List[float]` or :obj:`None`, optional, default=None
        Weights :math:`\sum_{i=1}^m w_i = 1, \ 0 < w_i < 1`,
        Defaults to None, which means :math:`w_1 = \cdots = w_m = \frac{1}{m}.`
    max_iter : :obj:`int`, optional, default=1000
        The maximum number of iterations.
    tol : :obj:`float`, optional, default=1e-7
        Torrelance to stop the iteration.
    callbacky : :obj:`bool`, optional
        Modify callback signature to (``callback(x, y)``)
        when ``callbacky=True``
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model


    Notes
    -----
    TODO: write docstring



    """
    if show:
        tstart = time.time()
        print(
            "Parallel Proximal Algorithm\n"
            "---------------------------------------------------------"
        )
        for i, prox_op in enumerate(prox_ops):
            print(f"Proximal operator (f{i}): {type(prox_op)}")
        print(f"tau = {tau:10e}\tniter = {niter:d}\n")
        head = "   Itn       x[0]          J=sum_i f_i"
        print(head)

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

    for iiter in range(niter):

        p = ncp.stack([prox_ops[i].prox(y[i], tau / w[i]) for i in range(m)])
        pn = np.sum(w[:, None] * p, axis=0)
        y = y + eta * (2 * pn - x - p)
        x = x + eta * (pn - x)

        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf = np.sum([prox_ops[i](x) for i in range(m)])
                print(
                    f"{iiter + 1:6d}  {np.real(to_numpy(x[0])):12.5e}  "
                    f"{pf:10.3e}"
                )

        if np.abs(x - x_old).max() < tol:
            break

        x_old = x

    if show:
        print(f"\nTotal time (s) = {time.time() - tstart:.2f}")
        print("---------------------------------------------------------\n")

    return x
