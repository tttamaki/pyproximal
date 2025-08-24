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
        return self.more_projections(x)

    def set_projections(
        self,
        projections: List[Callable[[NDArray], NDArray]]
    ) -> None:
        self.projections = projections

    def two_projections(self, x0: NDArray) -> NDArray:
        x = x0.copy()
        p = np.zeros_like(x)
        q = np.zeros_like(x)

        for _ in range(self.max_iter):
            x_old = x.copy()

            y = self.projections[0](x + p)
            p = x + p - y
            x = self.projections[1](y + q)
            q = y + q - x

            if np.allclose(x, x_old):
                break
        return x

    def more_projections(self, x0: NDArray) -> NDArray:
        u = x0.copy()
        d = len(self.projections)
        z = [np.zeros_like(u) for _ in range(d)]

        for k in range(self.max_iter):
            u_old = u.copy()

            # Initialize u_0^(k) = u_d^(k-1)
            u_prev = u.copy()

            for i in range(d):
                # u_i^(k) = P_C_i(u_{i-1}^(k) + z_i^(k-1))
                u = self.projections[i](u_prev + z[i])

                # z_i^(k) = u_{i-1}^(k) + z_i^(k-1) - u_i^(k)
                z[i] = u_prev + z[i] - u

                # Update u_{i-1} for next iteration
                u_prev = u

            # Check convergence
            if np.allclose(u, u_old):
                break

        return u
