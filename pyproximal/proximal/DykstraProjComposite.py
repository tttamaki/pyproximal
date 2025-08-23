from typing import List, Callable
import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.projection import DykstraProjCompositeProj


class DykstraProjComposite(ProxOperator):

    def __init__(
        self,
        projections: List[Callable[[NDArray], NDArray]],
        max_iter: int = 50
    ) -> None:
        super().__init__(None, False)
        self.projections = projections
        self.max_iter = max_iter
        self.dykstra_proj_coomposite_proj = \
            DykstraProjCompositeProj(
                self.projections,
                self.max_iter
            )

    def __call__(self, x) -> bool:
        return all(np.allclose(x, proj(x)) for proj in self.projections)

    @_check_tau
    def prox(self, x, tau) -> NDArray:
        return self.dykstra_proj_coomposite_proj(x)
