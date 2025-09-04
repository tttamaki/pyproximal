from typing import List, Callable
import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import ProxOperator, _check_tau
from pyproximal.projection import DykstrasProjection


class DykstrasProjectionProx(ProxOperator):
    r"""The proximal operator corresponding to the convex projection to the
    intersection of convex sets using Dykstra's algorithm.

    Parameters
    ----------
    projections : :obj:`List[Callable[[NDArray], NDArray]]`
        Projection functions :math:`P_i`.
    max_iter : :obj:`int`, optional
        The maximum number of iterations. Default is 10.

    Notes
    -----
    As the intersection of convex sets is an indicator function,
    the proximal operator corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.DykstraProjCompositeProj` for details).

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
    def prox(self, x: NDArray, tau: float) -> NDArray:
        return self.dykstras_projection(x)
