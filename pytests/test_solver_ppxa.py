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
def test_ppxa_with_admm(par: Dict[str, Any]) -> None:
    """Check equivalency of PPXA and ADMM
    when using a single regularization term
    """
    rng = np.random.default_rng()

    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    # Random mixing matrix
    R = rng.normal(0.0, 1.0, (n, m))
    Rop = MatrixMult(R)

    y = Rop @ x

    l2 = L2(Op=Rop, b=y, niter=50, warm=True)
    l1 = L1(sigma=5e-1)

    # Step size
    L = (Rop.H * Rop).eigs(1).real.item()
    tau = 0.5 / L

    xadmm, _ = ADMM(
        l2, l1, x0=np.zeros(m), tau=tau, niter=15000, show=True)
    xppxa1 = PPXA(
        [l2, l1], x0=np.zeros(m), tau=tau, niter=15000, show=True, tol=1e-7)

    assert_array_almost_equal(xadmm, xppxa1, decimal=2)

    xppxa2 = PPXA(
        [l1, l2], x0=np.zeros(m), tau=tau, niter=15000, show=True, tol=1e-7)

    assert_array_almost_equal(xadmm, xppxa2, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ppxa_with_drs(par: Dict[str, Any]) -> None:
    """Check equivalency of PPXA and DouglasRachfordSplitting
    when using a single regularization term
    """
    rng = np.random.default_rng()

    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    # Random mixing matrix
    R = rng.normal(0.0, 1.0, (n, m))
    Rop = MatrixMult(R)

    y = Rop @ x

    l2 = L2(Op=Rop, b=y, niter=50, warm=True)
    l1 = L1(sigma=5e-1)

    # Step size
    L = (Rop.H * Rop).eigs(1).real.item()
    tau = 0.5 / L

    xdrs, _ = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m), tau=tau, niter=15000, show=True)
    xppxa1 = PPXA(
        [l2, l1], x0=np.zeros(m), tau=tau, niter=15000, show=True, tol=1e-7)

    assert_array_almost_equal(xdrs, xppxa1, decimal=2)

    xppxa2 = PPXA(
        [l1, l2], x0=np.zeros(m), tau=tau, niter=15000, show=True, tol=1e-7)

    assert_array_almost_equal(xdrs, xppxa2, decimal=2)
