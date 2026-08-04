"""
benchmark problem suite for iterative linear solver discovery.

every problem is a symmetric positive definite (spd) system a x = b, because the
line search step lengths used by the discovered algorithms (and by steepest
descent and conjugate gradient) are only well defined for spd a. each problem
carries a known solution x* so we can measure true error, not just residual.

all matrices are returned as scipy.sparse.csr_matrix so large systems stay cheap.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Problem:
    def __init__(self, name, A, b, x_star, kind, is_spd=True):
        self.name = name
        self.A = A.tocsr()
        self.b = b
        self.x_star = x_star          # ground truth solution
        self.kind = kind              # short family label for analysis
        self.is_spd = is_spd          # symmetric positive definite?
        self.n = A.shape[0]

    def __repr__(self):
        tag = "spd" if self.is_spd else "nonsym"
        return f"<Problem {self.name} n={self.n} kind={self.kind} {tag}>"


def _finish(name, A, kind, rng, is_spd=True):
    """given a, manufacture a random x* and set b = a x*."""
    n = A.shape[0]
    x_star = rng.standard_normal(n)
    b = A @ x_star
    return Problem(name, A, b, x_star, kind, is_spd=is_spd)


def poisson_1d(n=64, rng=None):
    """1d poisson / second difference operator (tridiagonal, spd)."""
    rng = rng or np.random.default_rng(0)
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    return _finish(f"poisson1d_n{n}", A, "poisson1d", rng)


def poisson_2d(m=16, rng=None):
    """2d poisson on an m x m grid via kronecker sum (spd, n = m*m)."""
    rng = rng or np.random.default_rng(0)
    main = 2.0 * np.ones(m)
    off = -1.0 * np.ones(m - 1)
    T = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    I = sp.identity(m, format="csr")
    A = sp.kron(I, T) + sp.kron(T, I)
    return _finish(f"poisson2d_{m}x{m}", A, "poisson2d", rng)


def diag_dominant(n=80, density=0.06, rng=None):
    """random sparse symmetric diagonally dominant spd system."""
    rng = rng or np.random.default_rng(1)
    B = sp.random(n, n, density=density, random_state=rng, format="csr")
    S = (B + B.T) * 0.5
    # make strictly diagonally dominant so the matrix is spd
    row_abs = np.abs(S).sum(axis=1).A1
    A = S + sp.diags(row_abs + 1.0)
    return _finish(f"diagdom_n{n}", A, "diagdom", rng)


def spd_conditioned(n=60, cond=1e3, rng=None):
    """dense spd matrix with a prescribed condition number (stress test)."""
    rng = rng or np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.logspace(0, np.log10(cond), n)
    A = sp.csr_matrix((Q * eigs) @ Q.T)
    return _finish(f"spd_cond{int(cond)}_n{n}", A, "conditioned", rng)


# ----------------------------------------------------------------------------
# nonsymmetric families. here a is not symmetric, so the spd line search step
# is invalid and methods like gmres / bicgstab / orthomin are the human baselines.
# these problems carry is_spd=False so callers can pick the right comparison.
# ----------------------------------------------------------------------------

def convection_diffusion_2d(m=16, pe=20.0, rng=None):
    """2d convection diffusion on an m x m grid, upwind differences.

    the diffusion part is the symmetric poisson operator, the convection part
    adds a first derivative that makes a nonsymmetric. pe is a peclet like knob:
    bigger pe means more convection and a more strongly nonsymmetric problem.
    """
    rng = rng or np.random.default_rng(0)
    n = m * m
    h = 1.0 / (m + 1)
    # diffusion: standard 5 point laplacian (spd)
    main = 2.0 * np.ones(m)
    off = -1.0 * np.ones(m - 1)
    T = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    I = sp.identity(m, format="csr")
    diffusion = sp.kron(I, T) + sp.kron(T, I)
    # convection in the x direction via a nonsymmetric first difference
    c = pe * h
    fwd = sp.diags([-1.0 * np.ones(m - 1), np.ones(m)], [-1, 0], format="csr")
    convection = sp.kron(I, fwd)
    A = (diffusion + c * convection).tocsr()
    return _finish(f"convdiff_{m}x{m}_pe{int(pe)}", A, "convdiff", rng, is_spd=False)


def nonsym_diag_dominant(n=80, density=0.06, skew=0.6, rng=None):
    """random sparse diagonally dominant but nonsymmetric system."""
    rng = rng or np.random.default_rng(1)
    B = sp.random(n, n, density=density, random_state=rng, format="csr")
    # blend a symmetric and an antisymmetric part so a is not symmetric
    S = (B + B.T) * 0.5
    K = (B - B.T) * 0.5
    M = S + skew * K
    row_abs = np.abs(M).sum(axis=1).A1
    A = (M + sp.diags(row_abs + 1.0)).tocsr()
    return _finish(f"nonsymdd_n{n}", A, "nonsymdd", rng, is_spd=False)


def recirculating(n=80, rng=None):
    """a nonsymmetric matrix with a dominant skew (recirculating) part.

    a = diag + gamma * (shift - shift^T). the strong antisymmetric block is a
    stress test for short recurrence solvers.
    """
    rng = rng or np.random.default_rng(2)
    shift = sp.diags([np.ones(n - 1)], [1], format="csr")
    skew = shift - shift.T
    A = (sp.diags([4.0 * np.ones(n)], [0]) + 2.0 * skew).tocsr()
    return _finish(f"recirc_n{n}", A, "recirc", rng, is_spd=False)


def training_suite(rng=None):
    """problems the search is allowed to see (used to score candidates)."""
    rng = rng or np.random.default_rng(10)
    return [
        poisson_1d(48, rng),
        poisson_1d(96, rng),
        poisson_2d(12, rng),
        diag_dominant(70, rng=rng),
        spd_conditioned(50, cond=1e2, rng=rng),
    ]


def test_suite(rng=None):
    """held out problems used only for the final benchmark (never for scoring)."""
    rng = rng or np.random.default_rng(20)
    return [
        poisson_1d(128, rng),
        poisson_2d(16, rng),
        diag_dominant(120, rng=rng),
        spd_conditioned(80, cond=1e3, rng=rng),
        spd_conditioned(80, cond=1e4, rng=rng),
    ]


def nonsym_training_suite(rng=None):
    """nonsymmetric problems the search may see. deliberately a narrow slice
    (moderate convection diffusion + one diag dominant case) so that the test
    suite can probe generalization to unseen families and harder settings."""
    rng = rng or np.random.default_rng(30)
    return [
        convection_diffusion_2d(12, pe=15.0, rng=rng),
        convection_diffusion_2d(12, pe=30.0, rng=rng),
        nonsym_diag_dominant(70, skew=0.6, rng=rng),
    ]


def nonsym_test_suite(rng=None):
    """held out nonsymmetric problems: larger grids, stronger convection, and
    two families (recirculating, stronger skew) never seen during the search."""
    rng = rng or np.random.default_rng(40)
    return [
        convection_diffusion_2d(16, pe=20.0, rng=rng),   # bigger grid, seen family
        convection_diffusion_2d(16, pe=60.0, rng=rng),   # stronger convection
        nonsym_diag_dominant(120, skew=0.9, rng=rng),    # bigger, stronger skew
        recirculating(100, rng=rng),                     # unseen family
    ]


if __name__ == "__main__":
    print("spd suites:")
    for p in training_suite() + test_suite():
        print(" ", p, "  ||b|| =", f"{np.linalg.norm(p.b):.3e}")
    print("nonsymmetric suites:")
    for p in nonsym_training_suite() + nonsym_test_suite():
        print(" ", p, "  ||b|| =", f"{np.linalg.norm(p.b):.3e}")
