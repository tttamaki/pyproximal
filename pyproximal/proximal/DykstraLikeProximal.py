from typing import List
import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import ProxOperator, _check_tau


class DykstraLikeProximal(ProxOperator):
    r"""Proximal operator of a sum of two or more convex functions
    using Dykstra-like algorithm.

    Parameters
    ----------
    ops : :obj:`List[ProxOperator]`
        A list of proximable functions :math:`f_1, \ldots, f_m`.
    max_iter : :obj:`int`, optional, default=10
        The maximum number of iterations.
    use_parallel : :obj:`bool`, optional, default=False
        If True, use the parallel version when $m=2$.
    weights : :obj:`List[float] | None`, optional, default=None
        A list of weights for the weighted sum. Defaults to None, which means
        :math:`w_1 = \cdots = w_m = \frac{1}{m}.`

    Notes
    -----
    Given two :math:`f` and :math:`g`, or a set of proximable functions
    :math:`f_i` and corresponding weights :math:`w_i` for :math:`i=1, \ldots, m`,
    this class computes the proximal operator of the sum of two functions

    .. math:: \prox_{\tau \ f + g}

    using Dykstra-like algorithm, or the weighted sum of functions

    .. math:: \prox_{\tau \ \sum_{i=1}^m w_i f_i}

    using parallel Dykstra-like algorithm.

    For :math:`m=2`:
    The proximal mapping :math:`\prox_{\tau f + g}(\mathbf{x})` of
    :math:`\mathbf{x}` is computed by the Dykstra-like algorithm [1]_, [2]_:

    * :math:`\mathbf{x}^{(0)} = \mathbf{x}, \mathbf{p}^{(0)} = \mathbf{q}^{(0)} = \mathbf{0}`
    * for :math:`k = 1, \ldots`

      * :math:`\mathbf{y}^{(k)} = \prox_{\tau g}(\mathbf{x}^{(k)} + \mathbf{p}^{(k)})`
      * :math:`\mathbf{p}^{(k+1)} = \mathbf{p}_k + (\mathbf{x}_k - \mathbf{y}^{(k)})`
      * :math:`\mathbf{x}^{(k+1)} = \prox_{\tau f}(\mathbf{y}^{(k)} + \mathbf{q}^{(k)})`
      * :math:`\mathbf{q}^{(k+1)} = \mathbf{q}^{(k)} + (\mathbf{y}^{(k)} - \mathbf{x}^{(k+1)})`



    For :math:`m \ge 2`:
    The proximal mapping :math:`\prox_{\tau \sum_{i=1}^m w_i f_i}(\mathbf{x})`
    of :math:`\mathbf{x}` is computed by
    the parallel Dykstra-like algorithm [3]_, [4]_, [5]_,
    where :math:`\sum_{i=1}^m w_i = 1, \ 0 < w_i < 1`:

    * :math:`\mathbf{x}^{(0)} = \mathbf{z}_{1}^{(0)} = \cdots = \mathbf{z}_{m}^{(0)} = \mathbf{x}`
    * for :math:`k = 1, \ldots`

      * :math:`\mathbf{x}^{(k+1)} = \sum_{i=1}^{m} w_i \prox_{\tau f_i} (\mathbf{z}_{i}^{(k)})`
      * for :math:`i = 1, \ldots, m`

        * :math:`\mathbf{z}_{i}^{(k+1)} = \mathbf{z}_{i}^{(k)} + (\mathbf{x}^{(k+1)} - \prox_{\tau f_i} (\mathbf{z}_{i}^{(k)}) )`


    References
    ----------

    .. [1] Combettes, P.L., Pesquet, J.-C., 2011. Proximal Splitting Methods in
        Signal Processing, in Fixed-Point Algorithms for Inverse Problems in
        Science and Engineering, Springer, pp. 185-212. Algorithm 10.18.
        https://doi.org/10.1007/978-1-4419-9569-8_10
    .. [2] Bauschke, H.H., Combettes, P.L., 2008. A Dykstra-like algorithm for two
        monotone operators. Pacific Journal of Pitimization 4, 383-391.
        Theorem 3.3.
        http://www.ybook.co.jp/online-p/PJO/vol4/pjov4n3p383.pdf

    .. [3] Combettes, P.L., Pesquet, J.-C., 2011. Proximal Splitting Methods in
        Signal Processing, in Fixed-Point Algorithms for Inverse Problems in
        Science and Engineering, Springer, pp. 185-212. Algorithm 10.31.
        https://doi.org/10.1007/978-1-4419-9569-8_10
    .. [4] Combettes, P.L., Dũng, Đ., Vũ, B.C., 2011. Proximity for sums of composite
        functions. Journal of Mathematical Analysis and Applications 380, 680-688.
        Eq. (2.26)
        https://doi.org/10.1016/j.jmaa.2011.02.079
    .. [5] Combettes, P.L., 2009. Iterative Construction of the Resolvent of a Sum of
        Maximal Monotone Operators. Journal of Convex Analysis 16, 727-748.
        Theorem 4.2.
        https://www.heldermann.de/JCA/JCA16/JCA163/jca16044.htm


    See also
    --------
    projection.DykstrasProjection :
        The convex projection to the intersection of convex sets
        using Dykstra's algorithm.

    """

    def __init__(
        self,
        ops: List[ProxOperator],
        max_iter: int = 10,
        use_parallel: bool = False,
        weights: List[float] | None = None
    ) -> None:
        super().__init__(None, False)
        assert len(ops) > 0
        self.ops = ops
        self.max_iter = max_iter
        self.use_parallel = use_parallel
        if weights is None:
            self.w = [1. / len(self.ops)] * len(self.ops)
        else:
            self.w = weights

    def __call__(self, x: NDArray) -> bool | float:
        """Proximable function

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector

        Returns
        -------
        :obj:`bool` or  :obj:`float`
            - return False immediately if any boolean ops is False
            - return the sum of numeric ops values if all boolean ops are True
            - return True if ops are all boolean (no numeric ops) and True
        """
        prox_sum = 0.
        has_numeric = False
        for prox_op in self.ops:
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
        if len(self.ops) == 1:
            return self.ops[0].prox(x, tau)

        if len(self.ops) == 2 and not self.use_parallel:
            return self.dykstra_like_proximal_algorithm(x, tau)

        return self.parallel_dykstra_like_proximal_algorithm(x, tau)

    @_check_tau
    def dykstra_like_proximal_algorithm(
        self, x0: NDArray, tau: float
    ) -> NDArray:
        r"""Compute :math:`\prox_{\tau \ f + g}(\mathbf{x})` for :math:`m = 2`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        tau : :obj:`float`
            Positive scalar weight

        Returns
        -------
        :obj:`np.ndarray`
            prox of x

        """
        x = x0.copy()
        p = np.zeros_like(x)
        q = np.zeros_like(x)

        for _ in range(self.max_iter):
            x_old = x.copy()

            y = self.ops[0].prox(x + p, tau)
            p = p + x - y
            x = self.ops[1].prox(y + q, tau)
            q = q + y - x

            if np.allclose(x, x_old):
                break

        return x

    @_check_tau
    def parallel_dykstra_like_proximal_algorithm(
        self, x0: NDArray, tau: float
    ) -> NDArray:
        r"""Compute :math:`\prox_{\tau \ \sum_{i=1}^m w_i f_i}` for :math:`m \ge 2`.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        tau : :obj:`float`
            Positive scalar weight

        Returns
        -------
        :obj:`np.ndarray`
            prox of x

        """
        x = x0.copy()
        m = len(self.ops)
        z = [x0.copy() for _ in range(m)]

        for _ in range(self.max_iter):
            x_old = x.copy()

            for i in range(m):
                x += self.w[i] * self.ops[i].prox(z[i], tau)

            for i in range(m):
                z[i] = z[i] + x - self.ops[i].prox(z[i], tau)

            if np.allclose(x, x_old):
                break

        return x
