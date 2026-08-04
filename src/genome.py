"""
the representation of a candidate iterative algorithm (the "language").

every discovered algorithm is a two term recurrence of the general form

    d_k   = r_k  +  beta_k * p_{k-1}          (search direction)
    x_{k+1} = x_k + alpha_k * d_k             (update)
    p_k   = d_k

where r_k = b - a x_k is the residual. the scalars alpha_k and beta_k are not
fixed: each is an evolvable expression

    coeff = scale * feature[num] / (feature[den] + eps)

built from inner product features computed each iteration. this single template
is expressive enough to contain many known methods as special cases, e.g.

    richardson         : beta = 0,               alpha = const
    steepest descent   : beta = 0,               alpha = (r.r)/(r.ar)
    conjugate gradient : beta = (r.r)/(r.r)_prev, alpha = (r.d)/(d.ad)
    heavy ball momentum: beta = const,           alpha = const

so the evolutionary search is free to rediscover those, blend them, or invent
something new. each genome prints itself as readable pseudo math for analysis.
"""
from __future__ import annotations

import numpy as np

# divergent candidate algorithms legitimately overflow or produce nans, we detect
# and penalise those below, so silence the (expected) numerical warnings.
np.seterr(all="ignore")

EPS = 1e-30

# scalar features available when choosing beta (depend only on r, p, history).
BETA_FEATURES = ["one", "rr", "rr_prev", "rp", "pp"]

# scalar features available when choosing alpha (direction d and ad are known).
# rAd and AdAd give the residual minimizing step (r.ad)/(ad.ad), which is what
# nonsymmetric systems need since the spd line search step (r.d)/(d.ad) can blow
# up when a is not symmetric positive definite.
ALPHA_FEATURES = ["one", "rr", "rd", "dAd", "rAr", "pp", "pAd", "rr_prev",
                  "rAd", "AdAd"]


class Coeff:
    """a single evolvable scalar coefficient: scale * feat[num] / feat[den]."""

    def __init__(self, scale, num, den):
        self.scale = float(scale)
        self.num = num
        self.den = den

    def value(self, feats):
        n = feats[self.num]
        d = feats[self.den]
        if abs(d) < EPS:
            d = EPS if d >= 0 else -EPS
        v = self.scale * n / d
        if not np.isfinite(v):
            return 0.0
        return v

    def copy(self):
        return Coeff(self.scale, self.num, self.den)

    def __repr__(self):
        if self.num == "one" and self.den == "one":
            return f"{self.scale:.4g}"
        if self.den == "one":
            return f"{self.scale:.4g}*{self.num}"
        return f"{self.scale:.4g}*{self.num}/{self.den}"


class Genome:
    """a candidate algorithm = (beta coefficient, alpha coefficient)."""

    def __init__(self, beta: Coeff, alpha: Coeff):
        self.beta = beta
        self.alpha = alpha
        self.fitness = None  # filled in by the search

    # factory and variation
    @staticmethod
    def random(rng):
        beta = Coeff(rng.uniform(0.0, 1.0),
                     rng.choice(BETA_FEATURES), rng.choice(BETA_FEATURES))
        alpha = Coeff(rng.uniform(0.2, 1.5),
                      rng.choice(ALPHA_FEATURES), rng.choice(ALPHA_FEATURES))
        return Genome(beta, alpha)

    def copy(self):
        g = Genome(self.beta.copy(), self.alpha.copy())
        return g

    def mutate(self, rng, rate=0.5):
        g = self.copy()
        g.fitness = None
        for coeff, feats in ((g.beta, BETA_FEATURES), (g.alpha, ALPHA_FEATURES)):
            if rng.random() < rate:
                coeff.scale *= float(np.exp(rng.normal(0, 0.4)))   # log normal jitter
            if rng.random() < rate * 0.6:
                coeff.num = rng.choice(feats)
            if rng.random() < rate * 0.6:
                coeff.den = rng.choice(feats)
            if rng.random() < rate * 0.15:                          # occasional reset
                coeff.scale = float(rng.uniform(0.0, 1.5))
        return g

    @staticmethod
    def crossover(a, b, rng):
        """swap whole coefficients between two parents."""
        beta = (a.beta if rng.random() < 0.5 else b.beta).copy()
        alpha = (a.alpha if rng.random() < 0.5 else b.alpha).copy()
        return Genome(beta, alpha)

    def pseudocode(self):
        return (f"beta_k  = {self.beta}\n"
                f"alpha_k = {self.alpha}\n"
                f"d_k     = r_k + beta_k * p_(k-1)\n"
                f"x_(k+1) = x_k + alpha_k * d_k")

    def __repr__(self):
        return f"<Genome beta=({self.beta})  alpha=({self.alpha})>"


def run_genome(genome, problem, tol=1e-8, maxiter=2000):
    """
    execute a genome as an iterative solver. returns a baselines.SolveResult so
    discovered and classical algorithms are scored by identical code.
    """
    from baselines import SolveResult, _rel_err

    A, b, x_star = problem.A, problem.b, problem.x_star
    x = np.zeros_like(b)
    p = np.zeros_like(b)
    bnorm = np.linalg.norm(b) or 1.0
    rr_prev = None
    hist = []

    # only compute ar = a@r (an extra matrix vector product) if the genome's alpha
    # actually uses the rAr feature. most good methods do not, so they cost just
    # one matrix vector product per iteration (the a@d below).
    uses_rAr = "rAr" in (genome.alpha.num, genome.alpha.den)
    matvecs = 0

    r = b - A @ x
    for k in range(1, maxiter + 1):
        rnorm = np.linalg.norm(r)
        rel = rnorm / bnorm
        hist.append(rel)
        if not np.isfinite(rel) or rel > 1e8:            # diverged
            res = SolveResult("Discovered", len(hist), False, min(rel, 1e8),
                              _rel_err(x, x_star), hist)
            res.matvecs = matvecs
            return res
        if rel < tol:
            res = SolveResult("Discovered", k, True, rel, _rel_err(x, x_star), hist)
            res.matvecs = matvecs
            return res

        rr = float(r @ r)
        rp = float(r @ p)
        pp = float(p @ p)
        beta_feats = {"one": 1.0, "rr": rr,
                      "rr_prev": rr_prev if rr_prev is not None else rr,
                      "rp": rp, "pp": pp}
        beta = genome.beta.value(beta_feats)

        d = r + beta * p
        Ad = A @ d; matvecs += 1
        rd = float(r @ d)
        dAd = float(d @ Ad)
        pAd = float(p @ Ad)
        rAd = float(r @ Ad)
        AdAd = float(Ad @ Ad)
        if uses_rAr:
            Ar = A @ r; matvecs += 1
            rAr = float(r @ Ar)
        else:
            rAr = rr                      # placeholder, never used by this genome
        alpha_feats = {"one": 1.0, "rr": rr, "rd": rd, "dAd": dAd,
                       "rAr": rAr, "pp": pp, "pAd": pAd,
                       "rr_prev": rr_prev if rr_prev is not None else rr,
                       "rAd": rAd, "AdAd": AdAd}
        alpha = genome.alpha.value(alpha_feats)

        x = x + alpha * d
        r = r - alpha * Ad     # exact residual update (r = b - a x)
        p = d
        rr_prev = rr

    res = SolveResult("Discovered", maxiter, False, hist[-1], _rel_err(x, x_star), hist)
    res.matvecs = matvecs
    return res
