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
    max_iter : :obj:`int`, optional, default=10
        The maximum number of iterations.
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
        max_iter: int = 10,
        use_parallel: bool = False,
    ) -> None:
        super().__init__(None, False)
        self.projections = projections
        self.max_iter = max_iter
        self.dykstras_projection = \
            DykstrasProjection(
                self.projections,
                self.max_iter,
                use_parallel
            )

    def __call__(self, x: NDArray) -> bool:
        return all(np.allclose(x, proj(x)) for proj in self.projections)

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return self.dykstras_projection(x)
