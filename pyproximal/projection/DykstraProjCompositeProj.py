from typing import List, Callable
import numpy as np
from pylops.utils.typing import NDArray


class DykstraProjCompositeProj():

    def __init__(
            self,
            projections: List[Callable[[NDArray], NDArray]],
            max_iter: int = 10
    ) -> None:
        self.projections = projections
        self.max_iter = max_iter

    def __call__(self, x: NDArray) -> NDArray:
        if len(self.projections) == 1:
            return self.projections[0](x)
        if len(self.projections) == 2:
            return self.two_projections(x)

        raise NotImplementedError(
            "projections more than two is not yet ready.")

    def set_projections(
        self,
        projections: List[Callable[[NDArray], NDArray]]
    ) -> None:
        self.projections = projections

    def two_projections(self, x0: NDArray) -> NDArray:
        proj0 = self.projections[0]
        proj1 = self.projections[1]

        x = x0.copy()
        p = np.zeros_like(x)
        q = np.zeros_like(x)

        for _ in range(self.max_iter):
            x_old = x.copy()
            y = proj0(x + p)
            p = x + p - y
            x = proj1(y + q)
            q = y + q - x
            if np.allclose(x, x_old):
                break
        return x
