from typing import List, Callable
import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class DykstraProxComposite(ProxOperator):

    def __init__(
        self,
        prox_ops: List[ProxOperator],
        max_iter: int = 10
    ) -> None:
        super().__init__(None, False)
        self.prox_ops = prox_ops
        self.max_iter = max_iter

    def __call__(self, x: NDArray) -> bool | float:
        """_summary_

        Args:
            x (NDArray): _description_

        Returns:
            bool | float: _description_
            - return False immediately if any boolean prox_ops is False
            - return True if prox_ops are all boolean and True
            - return the sum of numeric prox_ops values
                if all boolean prox_ops are True
        """
        prox_sum = 0.
        has_numeric = False
        for prox_op in self.prox_ops:
            prox_x = prox_op(x)
            if isinstance(prox_x, (bool, np.bool_)):
                if not prox_x:
                    return False
            else:  # float or int
                prox_sum += prox_x
                has_numeric = True
        return prox_sum if has_numeric else True

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        if len(self.prox_ops) == 1:
            return self.prox_ops[0].prox(x, tau)

        if len(self.prox_ops) == 2:
            return self.two_prox_ops(x, tau)

        return self.more_prox_ops(x, tau)

    @_check_tau
    def two_prox_ops(self, x: NDArray, tau: float) -> NDArray:
        return x

    @_check_tau
    def more_prox_ops(self, x: NDArray, tau: float) -> NDArray:
        return x
