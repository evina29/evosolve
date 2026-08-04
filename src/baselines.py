"""
human designed iterative solvers, used as the comparison baselines.

each solver has the same signature and returns a SolveResult so the discovered
algorithms and the classical ones can be scored by identical code.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class SolveResult:
    def __init__(self, name, iters, converged, rel_res, rel_err, res_history,
                 matvecs=None):
        self.name = name
        self.iters = iters              # iterations actually taken
        self.converged = converged      # reached tolerance before maxiter?
        self.rel_res = rel_res          # final ||b-ax|| / ||b||
        self.rel_err = rel_err          # final ||x-x*|| / ||x*||
        self.res_history = res_history  # list of relative residuals per iter
        # matrix vector products, the fair cost metric. defaults to one per
        # iteration, which is right for everything except bicgstab (two per iter).
        self.matvecs = matvecs if matvecs is not None else iters

    def __repr__(self):
        flag = "ok " if self.converged else "NO "
        return (f"[{flag}] {self.name:<22} iters={self.iters:>4}  "
                f"rel_res={self.rel_res:.2e}  rel_err={self.rel_err:.2e}")


def _rel_err(x, x_star):
    d = np.linalg.norm(x_star)
    return np.linalg.norm(x - x_star) / d if d > 0 else np.linalg.norm(x - x_star)


def jacobi(problem, tol=1e-8, maxiter=2000):
    A, b, x_star = problem.A, problem.b, problem.x_star
    d = A.diagonal()
    x = np.zeros_like(b)
    bnorm = np.linalg.norm(b) or 1.0
    hist = []
    for k in range(1, maxiter + 1):
        r = b - A @ x
        rr = np.linalg.norm(r) / bnorm
        hist.append(rr)
        if rr < tol:
            return SolveResult("Jacobi", k, True, rr, _rel_err(x, x_star), hist)
        x = x + r / d
    return SolveResult("Jacobi", maxiter, False, hist[-1], _rel_err(x, x_star), hist)


def gauss_seidel(problem, tol=1e-8, maxiter=2000):
    A, b, x_star = problem.A, problem.b, problem.x_star
    Ad = A.tocsr()
    L = sp.tril(Ad, format="csc")        # lower triangle including the diagonal
    from scipy.sparse.linalg import spsolve_triangular
    x = np.zeros_like(b)
    bnorm = np.linalg.norm(b) or 1.0
    hist = []
    for k in range(1, maxiter + 1):
        r = b - Ad @ x
        rr = np.linalg.norm(r) / bnorm
        hist.append(rr)
        if rr < tol:
            return SolveResult("Gauss-Seidel", k, True, rr, _rel_err(x, x_star), hist)
        x = x + spsolve_triangular(L, r, lower=True)
    return SolveResult("Gauss-Seidel", maxiter, False, hist[-1], _rel_err(x, x_star), hist)


def steepest_descent(problem, tol=1e-8, maxiter=2000):
    A, b, x_star = problem.A, problem.b, problem.x_star
    x = np.zeros_like(b)
    bnorm = np.linalg.norm(b) or 1.0
    hist = []
    r = b - A @ x
    for k in range(1, maxiter + 1):
        rr = np.linalg.norm(r) / bnorm
        hist.append(rr)
        if rr < tol:
            return SolveResult("Steepest Descent", k, True, rr, _rel_err(x, x_star), hist)
        Ar = A @ r
        denom = r @ Ar
        if denom <= 0 or not np.isfinite(denom):
            break
        alpha = (r @ r) / denom
        x = x + alpha * r
        r = r - alpha * Ar
    return SolveResult("Steepest Descent", len(hist), False, hist[-1], _rel_err(x, x_star), hist)


def conjugate_gradient(problem, tol=1e-8, maxiter=2000):
    A, b, x_star = problem.A, problem.b, problem.x_star
    x = np.zeros_like(b)
    bnorm = np.linalg.norm(b) or 1.0
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    hist = []
    for k in range(1, maxiter + 1):
        rr = np.sqrt(rs_old) / bnorm
        hist.append(rr)
        if rr < tol:
            return SolveResult("Conjugate Gradient", k, True, rr, _rel_err(x, x_star), hist)
        Ap = A @ p
        denom = p @ Ap
        if denom <= 0 or not np.isfinite(denom):
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return SolveResult("Conjugate Gradient", len(hist), False, hist[-1], _rel_err(x, x_star), hist)


def orthomin1(problem, tol=1e-8, maxiter=2000):
    """minimal residual short recurrence, also known as gmres(1) / orthomin(1).

    valid for any invertible a (no symmetry needed). the step minimises the
    residual norm along d = r, giving alpha = (r.ad)/(ad.ad). this is the fair
    human short recurrence baseline for a discovered two term nonsymmetric method.
    """
    A, b, x_star = problem.A, problem.b, problem.x_star
    x = np.zeros_like(b)
    bnorm = np.linalg.norm(b) or 1.0
    hist = []
    r = b - A @ x
    for k in range(1, maxiter + 1):
        rr = np.linalg.norm(r) / bnorm
        hist.append(rr)
        if rr < tol:
            return SolveResult("Orthomin(1)", k, True, rr, _rel_err(x, x_star), hist)
        Ar = A @ r
        denom = Ar @ Ar
        if denom <= 0 or not np.isfinite(denom):
            break
        alpha = (r @ Ar) / denom
        x = x + alpha * r
        r = r - alpha * Ar
    return SolveResult("Orthomin(1)", len(hist), False, hist[-1], _rel_err(x, x_star), hist)


def _scipy_result(name, problem, x, hist, tol):
    """wrap a scipy krylov run into a SolveResult using the true final residual."""
    A, b, x_star = problem.A, problem.b, problem.x_star
    bnorm = np.linalg.norm(b) or 1.0
    rr = np.linalg.norm(b - A @ x) / bnorm
    converged = rr < tol
    iters = len(hist) if hist else 0
    if not hist:
        hist = [rr]
    return SolveResult(name, iters, converged, rr, _rel_err(x, x_star), hist)


def gmres_restarted(problem, tol=1e-8, maxiter=2000, restart=30):
    """restarted gmres from scipy. counts inner iterations via the pr_norm callback."""
    from scipy.sparse.linalg import gmres
    A, b = problem.A, problem.b
    bnorm = np.linalg.norm(b) or 1.0
    hist = []
    def cb(pr):                      # pr is the (relative) residual norm per inner step
        hist.append(float(pr))
    outer = max(1, maxiter // restart)
    x, _ = gmres(A, b, rtol=tol, atol=0.0, restart=restart, maxiter=outer,
                 callback=cb, callback_type="pr_norm")
    return _scipy_result("GMRES(30)", problem, x, hist, tol)


def bicgstab(problem, tol=1e-8, maxiter=2000):
    """bicgstab from scipy. the callback receives xk, so we form the residual."""
    from scipy.sparse.linalg import bicgstab as _bicgstab
    A, b = problem.A, problem.b
    bnorm = np.linalg.norm(b) or 1.0
    hist = []
    def cb(xk):
        hist.append(float(np.linalg.norm(b - A @ xk) / bnorm))
    x, _ = _bicgstab(A, b, rtol=tol, atol=0.0, maxiter=maxiter, callback=cb)
    res = _scipy_result("BiCGSTAB", problem, x, hist, tol)
    res.matvecs = 2 * res.iters      # bicgstab applies a twice per iteration
    return res


# spd comparison set
BASELINES = {
    "jacobi": jacobi,
    "gauss_seidel": gauss_seidel,
    "steepest_descent": steepest_descent,
    "conjugate_gradient": conjugate_gradient,
}

# nonsymmetric comparison set (cg is invalid here; these are the fair baselines)
NONSYM_BASELINES = {
    "gmres_restarted": gmres_restarted,
    "bicgstab": bicgstab,
    "orthomin1": orthomin1,
}


if __name__ == "__main__":
    from problems import poisson_1d
    p = poisson_1d(64)
    for fn in BASELINES.values():
        print(fn(p))
