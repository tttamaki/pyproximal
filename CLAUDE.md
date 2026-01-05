# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyProximal is a Python library for solving non-smooth convex optimization problems using proximal algorithms. It provides proximal operators, projection operators, and optimization solvers that work with PyLops linear operators.

**Key Design Philosophy**: PyProximal focuses solely on proximal operators and algorithms, leveraging PyLops for all linear operators. This separation keeps the codebase modular.

## Development Commands

### Environment Setup
```bash
# Create development environment with conda
make dev-install_conda

# For ARM-based systems (M1/M2 Macs)
make dev-install_conda_arm

# Activate environment (required for all subsequent commands)
conda activate pyproximal
```

### Testing
```bash
# Run all tests
make tests

# Run specific test file
pytest pytests/test_proximal.py

# Run specific test function
pytest pytests/test_proximal.py::test_Quadratic -v
```

### Code Quality
```bash
# Run flake8 linting
make lint

# Run mypy type checking
make typeannot
```

### Documentation
```bash
# Full documentation build (includes examples/tutorials via sphinx-gallery)
make doc

# Quick rebuild after source changes (does NOT rebuild examples/tutorials)
make docupdate

# Serve documentation locally at http://localhost:8000
make servedoc
```

## Architecture

### Core Components

#### 1. ProxOperator Base Class (`pyproximal/ProxOperator.py`)

All proximal operators inherit from `ProxOperator`. Key methods:
- `prox(x, tau)`: Proximal operator evaluation
- `proxdual(x, tau)`: Dual proximal operator (automatically via Moreau decomposition)
- `__call__(x)`: Function evaluation
- `grad(x)`: Gradient (for differentiable functions with `hasgrad=True`)

**Important**: Subclasses only need to implement either `prox()` or `proxdual()` - the other is computed via Moreau decomposition.

#### 2. Proximal Operators (`pyproximal/proximal/`)

Each operator is in its own file (e.g., `L1.py`, `L2.py`, `Nuclear.py`). Categories:
- **Norms**: L1, L2, L21, Nuclear, TV (Total Variation)
- **Indicators**: Box, Simplex, AffineSet, Intersection (for constrained optimization)
- **Penalties**: Huber, SCAD, Log, ETP, Geman (non-smooth penalties)
- **Composite**: VStack, Sum, GenericIntersection

#### 3. Projection Operators (`pyproximal/projection/`)

Simplified versions of proximal operators for projecting onto sets (indicator functions with tau=1).

#### 4. Optimization Algorithms (`pyproximal/optimization/`)

Organized by algorithm family:

**`primal.py`** - Primal algorithms:
- Basic: `ProximalPoint`, `ProximalGradient`, `AcceleratedProximalGradient`
- ADMM variants: `ADMM`, `ADMML2`, `LinearizedADMM`
- Splitting: `HQS`, `TwIST`, `DouglasRachfordSplitting`
- Multi-term: `PPXA`, `ConsensusADMM`

**`primaldual.py`** - Primal-dual algorithms: `PrimalDual`, `AdaptivePrimalDual`

**`bregman.py`** - Bregman iterations (wraps other solvers)

**`palm.py`** - Alternating minimization: `PALM`, `iPALM`

**`sr3.py`** - Sparse Relaxed Regularized Regression

**`pnp.py`** - Plug-and-Play algorithms

**`segmentation.py`** - Image segmentation via primal-dual

#### 5. Utilities (`pyproximal/utils/`)

- `backend.py`: NumPy/CuPy backend handling for GPU support
- `bilinear.py`: Bilinear operators for matrix factorization problems
- `moreau.py`: Moreau identity verification for testing
- `gradtest.py`: Gradient testing utilities

### Problem Formulation

Solvers typically solve problems of the form:
```
min_x f(x) + g(Lx)
```
where:
- `f`: Differentiable (gradient) or proximable function
- `g`: Proximable function (proximal operator available)
- `L`: Linear operator from PyLops

Different solvers handle different combinations of these components.

## Testing Conventions

- Tests use parametrization: `par1` (even dimensions, float32) and `par2` (odd dimensions, float64)
- Moreau identity verification: `moreau(operator, x, tau)` validates `prox` and `proxdual` relationship
- Test files in `pytests/` directory: `test_proximal.py`, `test_projection.py`, `test_solver.py`, etc.

## Code Style

- **Formatter**: black with 88-char line length
- **Import sorting**: isort with black profile (skip `__init__.py`)
- **Linting**: flake8 - minimize warnings but full compliance not required
- **Type checking**: mypy in strict mode
- **Pre-commit hooks**: Configured for black, isort, trailing whitespace, EOF fixer

## Important Notes

- **Main branch**: Development happens on `dev` branch (not `main`)
- **Python version**: 3.8+ required (v0.3.0+)
- **Core dependency**: Requires `pylops >= 2.0.0` for linear operators
- **Documentation rebuilds**: Changes to examples/tutorials require full `make doc` (not `make docupdate`)
- **Optional acceleration**: Install `numba` and `llvmlite` for performance

## Adding New Operators

When implementing a new proximal operator:

1. Create new file in `pyproximal/proximal/YourOperator.py`
2. Inherit from `ProxOperator`
3. Implement `__call__(self, x)` for function evaluation
4. Implement either `prox()` or `proxdual()` (the other comes free via Moreau)
5. If differentiable, implement `grad()` and set `hasgrad=True`
6. Add to `pyproximal/proximal/__init__.py` imports and `__all__`
7. Create tests in `pytests/test_proximal.py` verifying Moreau identity
8. Add example in `examples/` for documentation

See existing operators for patterns and the documentation for detailed guidance.
