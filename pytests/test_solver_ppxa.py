from typing import Dict, Any, Callable

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from pylops.basicoperators import Identity, MatrixMult

from pyproximal.optimization.primal import (
    ADMM,
    DouglasRachfordSplitting,
)
from pyproximal.optimization.ppxa import PPXA

from pyproximal.proximal import L1, L2

par1 = {"n": 8, "m": 10, "dtype": "float32"}  # float64
par2 = {"n": 8, "m": 10, "dtype": "float64"}  # float32


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PPXA(par: Dict[str, Any]) -> None:
    """Check equivalency of ADMM and DouglasRachfordSplitting
    when using a single regularization term
    """
    np.random.seed(0)
    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    # Random mixing matrix
    R = np.random.normal(0.0, 1.0, (n, m))
    Rop = MatrixMult(R)

    y = Rop @ x

    # Step size
    L = (Rop.H * Rop).eigs(1).real.item()
    tau = 0.5 / L

    # ADMM
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xadmm, zadmm = ADMM(l2, l1, x0=np.zeros(m), tau=tau, niter=100, show=True)

    # DRS with g first
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_g = PPXA(
        [l2, l1], x0=np.zeros(m), tau=tau, niter=500, show=True, tol=1e-4
    )

    assert_array_almost_equal(xadmm, xdrs_g, decimal=2)
