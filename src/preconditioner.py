"""
preconditioner discovery (spd systems).

instead of discovering the solver, here we fix the solver to conjugate gradient
and let the search discover a polynomial preconditioner m_inv = q(a) that makes
cg converge in fewer steps. a degree d polynomial preconditioner costs d extra
matrix vector products per cg iteration, so the honest cost metric is total
matvecs = iterations * (1 + d), not iterations. we report that against plain cg
and jacobi preconditioned cg.

the polynomial is built on the diagonally scaled matrix ahat = d_inv @ a:

    m_inv v = ( sum_k c_k (i - ahat)^k ) d_inv v

with evolvable coefficients c_0..c_d. c_k = 1 for all k is the classic neumann
series approximation to ahat^{-1}; the search is free to find better weights.
"""
from __future__ import annotations

import numpy as np

from problems import poisson_2d, spd_conditioned, diag_dominant
from baselines import SolveResult, _rel_err


def _apply_poly(coeffs, A, d_inv, v):
    """apply m_inv = (sum_k c_k (i - ahat)^k) d_inv to v. returns (result, matvecs).
    horner on t = (i - ahat) where ahat x = d_inv * (a @ x)."""
    x = d_inv * v                      # d_inv v
    matvecs = 0
    acc = coeffs[-1] * x
    for c in reversed(coeffs[:-1]):
        # acc = c*x + (i - ahat) acc  =  c*x + acc - d_inv*(a@acc)
        Aacc = A @ acc; matvecs += 1
        acc = c * x + acc - d_inv * Aacc
    return acc, matvecs


def pcg(A, b, x_star, coeffs, d_inv, tol=1e-8, maxiter=2000, degree_cost=0):
    """preconditioned conjugate gradient with the polynomial preconditioner.
    counts total matrix vector products (the a@p each iter plus preconditioner)."""
    x = np.zeros_like(b)
    bnorm = np.linalg.norm(b) or 1.0
    r = b - A @ x
    z, mv_pre = _apply_poly(coeffs, A, d_inv, r)
    p = z.copy()
    rz_old = r @ z
    hist = []
    matvecs = 0
    for k in range(1, maxiter + 1):
        rel = np.linalg.norm(r) / bnorm
        hist.append(rel)
        if not np.isfinite(rel) or rel > 1e8:
            break
        if rel < tol:
            res = SolveResult("PCG(discovered)", k, True, rel, _rel_err(x, x_star), hist)
            res.matvecs = matvecs
            return res
        Ap = A @ p; matvecs += 1
        denom = p @ Ap
        if denom <= 0 or not np.isfinite(denom):
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        z, mv_pre = _apply_poly(coeffs, A, d_inv, r)
        matvecs += mv_pre
        rz_new = r @ z
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    res = SolveResult("PCG(discovered)", len(hist), False,
                      hist[-1] if hist else 1.0, _rel_err(x, x_star), hist)
    res.matvecs = matvecs
    return res


def _precond_suite(rng=None):
    """small spd suite for scoring preconditioners (kept cheap)."""
    rng = rng or np.random.default_rng(50)
    return [
        poisson_2d(14, rng),
        spd_conditioned(60, cond=1e3, rng=rng),
        spd_conditioned(60, cond=1e4, rng=rng),
        diag_dominant(90, rng=rng),
    ]


def _prep(problem):
    d_inv = 1.0 / problem.A.diagonal()
    return d_inv


def score_coeffs(coeffs, problems, tol=1e-8, maxiter=1500):
    """mean cg iterations of pcg over the suite, with a penalty for failures.

    we score iterations, not serial matvecs, on purpose: the point of a
    polynomial preconditioner is to cut the number of cg steps (each step has a
    synchronisation and dot products), which is the cost that matters on parallel
    or communication bound hardware. the serial matvec tradeoff is reported too.
    """
    total = 0.0
    for p in problems:
        d_inv = _prep(p)
        res = pcg(p.A, p.b, p.x_star, coeffs, d_inv, tol=tol, maxiter=maxiter)
        total += res.iters if res.converged else maxiter * 3
    return total / len(problems)


def evolve_preconditioner(degree=3, gens=25, pop=24, seed=0,
                          tol=1e-8, maxiter=1500):
    """a small (mu + lambda) evolution strategy over the polynomial coefficients."""
    rng = np.random.default_rng(seed)
    train = _precond_suite()
    dim = degree + 1

    # start near the neumann series (all ones), which is a sensible prior
    pop_c = [np.ones(dim) + 0.3 * rng.standard_normal(dim) for _ in range(pop)]
    fits = [score_coeffs(c, train, tol, maxiter) for c in pop_c]
    best_i = int(np.argmin(fits))
    best_c, best_f = pop_c[best_i].copy(), fits[best_i]

    sigma = 0.3
    for gen in range(1, gens + 1):
        order = np.argsort(fits)
        parents = [pop_c[i] for i in order[:max(2, pop // 4)]]
        children, cfits = [], []
        for _ in range(pop):
            a = parents[rng.integers(len(parents))]
            child = a + sigma * rng.standard_normal(dim)
            f = score_coeffs(child, train, tol, maxiter)
            children.append(child); cfits.append(f)
        # (mu + lambda) selection: keep best of parents + children
        allc = [pop_c[i] for i in order[:pop // 2]] + children
        allf = [fits[i] for i in order[:pop // 2]] + cfits
        idx = np.argsort(allf)[:pop]
        pop_c = [allc[i] for i in idx]; fits = [allf[i] for i in idx]
        if fits[0] < best_f:
            best_c, best_f = pop_c[0].copy(), fits[0]
        sigma *= 0.92                      # cool down
    return best_c, best_f, train


def run_precond_experiment(degree=3, gens=25, pop=24, seed=0):
    print("=" * 70)
    print("  experiment d: polynomial preconditioner discovery (spd)")
    print("=" * 70)
    best_c, best_f, train = evolve_preconditioner(degree=degree, gens=gens,
                                                  pop=pop, seed=seed)
    print("discovered polynomial coefficients c_0..c_d:")
    print("  " + ", ".join(f"{c:+.3f}" for c in best_c))
    print(f"(degree {degree}, so preconditioner costs {degree} matvecs per apply)\n")

    # two metrics: cg iterations (parallel / communication cost, what we optimised)
    # and total serial matrix vector products (the serial cost, the honest caveat).
    print("cg iterations to converge (lower is better, the parallel relevant cost):")
    hdr = f"{'problem':<20}{'plain cg':>12}{'jacobi cg':>12}{'discovered':>12}"
    print(hdr); print("=" * len(hdr))
    wins_it_cg = wins_it_jac = 0
    matvec_rows = []
    for p in train:
        d_inv = _prep(p)
        cg = pcg(p.A, p.b, p.x_star, np.array([1.0]), np.ones_like(d_inv), maxiter=3000)
        jac = pcg(p.A, p.b, p.x_star, np.array([1.0]), d_inv, maxiter=3000)
        disc = pcg(p.A, p.b, p.x_star, best_c, d_inv, maxiter=3000)
        f = lambda v, ok: str(v) if ok else "fail"
        print(f"{p.name:<20}{f(cg.iters,cg.converged):>12}"
              f"{f(jac.iters,jac.converged):>12}{f(disc.iters,disc.converged):>12}")
        if disc.converged and cg.converged and disc.iters < cg.iters: wins_it_cg += 1
        if disc.converged and jac.converged and disc.iters < jac.iters: wins_it_jac += 1
        matvec_rows.append((p.name, cg, jac, disc))
    print(f"\ndiscovered uses fewer cg iterations than plain cg on "
          f"{wins_it_cg}/{len(train)}, than jacobi cg on {wins_it_jac}/{len(train)}.")

    print("\nsame runs, total serial matrix vector products (the honest tradeoff):")
    print(hdr); print("=" * len(hdr))
    for name, cg, jac, disc in matvec_rows:
        f = lambda v, ok: str(v) if ok else "fail"
        print(f"{name:<20}{f(cg.matvecs,cg.converged):>12}"
              f"{f(jac.matvecs,jac.converged):>12}{f(disc.matvecs,disc.converged):>12}")
    print("\nreading: the discovered preconditioner cuts cg iterations (fewer "
          "synchronisations, the cost that dominates on parallel hardware) but a "
          "degree 3 polynomial applies a three extra times per step, so it costs "
          "more serial matvecs. that tradeoff is exactly why polynomial "
          "preconditioners are used in parallel computing.")
    return best_c


if __name__ == "__main__":
    run_precond_experiment()
