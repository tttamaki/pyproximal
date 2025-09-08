from typing import List, Callable, Any

import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import ProxOperator, _check_tau
from pyproximal.projection import DykstrasProjection


class DykstrasProjectionProx(ProxOperator):
    r"""The proximal operator corresponding to the convex projection to the
    intersection of convex sets using Dykstra's algorithm.

    Parameters
    ----------
    projections : :obj:`List[Callable[[np.ndarray], np.ndarray]]`
        A list of projection functions :math:`P_1, \ldots, P_m`.
    max_iter : :obj:`int`, optional, default=100
        The maximum number of iterations.
    tol : :obj:`float`, optional, default=1e-6
        Torrelance to stop the iteration.
    use_parallel : :obj:`bool`, optional, default=False
        If True, use the parallel version when $m=2$.

    Notes
    -----
    As the intersection of convex sets is an indicator function,
    the proximal operator corresponds to its convex projection
    (see :class:`pyproximal.projection.DykstrasProjection` for details).

    See also
    --------
    pyproximal.projection.DykstrasProjection :
        The corresponding convex projection.

    """

    def __init__(
        self,
        projections: List[Callable[[NDArray], NDArray]],
        max_iter: int = 100,
        tol: float = 1e-6,
        use_parallel: bool = False,
    ) -> None:
        super().__init__(None, False)
        self.projections = projections

        # The tolerance for the indicator function is set to 10 times larger
        # than the tolerance used in Dykstra's projection. This is because
        # using the same tolerance does not guarantee that the condition
        # will hold even after the convergence of Dykstra's algorithm.
        self.tol = tol * 10

        self.dykstras_projection = \
            DykstrasProjection(
                projections=self.projections,
                max_iter=max_iter,
                tol=tol,
                use_parallel=use_parallel,
            )

    def __call__(self, x: NDArray) -> bool:
        return all(np.abs(x - proj(x)).max() < self.tol
                   for proj in self.projections)

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return self.dykstras_projection(x)
