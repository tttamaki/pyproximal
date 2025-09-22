
r"""
Dykstra's algorithms
==============================================

This example showcases two closely related tasks:
projection onto an intersection of convex sets
using the Dykstra's projection algorithm,
and proximal operator of a sum of proximable functions
suing the Dykstra-like proximal algorithm.

"""

###############################################################################
# Here is an example of a projection onto the intersection of convex sets
# using :class:`pyproximal.projection.GenericIntersectionProj`.

import numpy as np
from pyproximal.projection import (
    BoxProj,
    EuclideanBallProj,
    GenericIntersectionProj
)

circle_1 = EuclideanBallProj(np.array([-2.5, 0.0]), 5)
circle_2 = EuclideanBallProj(np.array([2.5, 0.0]), 5)
circle_3 = EuclideanBallProj(np.array([0.0, 3.5]), 5)
box = BoxProj(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

projections = [circle_1, circle_2, circle_3, box]
dykstra_proj = GenericIntersectionProj(projections)

rng = np.random.default_rng(10)
x = rng.normal(0., 3.5, size=2)
print("x            =", x)

xp = dykstra_proj(x)

print("x projection =", xp)

###############################################################################
# Here is the same example of a projection onto the intersection of convex sets
# using :class:`pyproximal.GenericIntersectionProx`.

import numpy as np
from pyproximal.projection import (
    BoxProj,
    EuclideanBallProj,
)
from pyproximal.proximal import GenericIntersectionProx

circle_1 = EuclideanBallProj(np.array([-2.5, 0.0]), 5)
circle_2 = EuclideanBallProj(np.array([2.5, 0.0]), 5)
circle_3 = EuclideanBallProj(np.array([0.0, 3.5]), 5)
box = BoxProj(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

projections = [circle_1, circle_2, circle_3, box]
dykstra_prox = GenericIntersectionProx(projections)

rng = np.random.default_rng(10)
x = rng.normal(0., 3.5, size=2)

print("x            =", x)
print("Is x inside?", dykstra_prox(x))  # x is outside

xp = dykstra_prox.prox(x, 1.0)

print("x projection =", xp)
print("Is x inside?", dykstra_prox(xp))  # xp is outside

###############################################################################
# Here is an example of computing proximal operator of the sum of proximable functions
# using :class:`pyproximal.Sum`.

import numpy as np
from pyproximal.proximal import L1, L2, Sum
from pylops import MatrixMult
rng = np.random.default_rng(10)

A = MatrixMult(rng.normal(0., 1., size=(3, 5)))
b = rng.normal(0., 1., size=3)
sigma = rng.normal(0., 1.)
l2_term = L2(A, b)
l1_term = L1(sigma=sigma)

# for computing prox of 1/2 * ||Ax - b||_2^2 + sigma ||x||_1
dykstra = Sum([l2_term, l1_term])

x = rng.normal(0., 5., size=5)
tau = 1.0

prox_x = dykstra.prox(x, tau)

print("x      =", x)
print("prox(x)=", prox_x)
