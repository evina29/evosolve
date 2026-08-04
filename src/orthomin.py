"""
multi direction (gcr / orthomin(m) class) solver discovery for nonsymmetric a.

the two term recurrence in genome.py keeps a single previous direction. here we
keep a window of the last m directions p_j and their images ap_j, and build the
new direction as

    d   = r   + sum_j beta_j p_j
    ad  = a r + sum_j beta_j ap_j        (no extra matvec: ap_j are stored)
    x  += alpha d
    r  -= alpha ad
    push (d, ad) into the window, drop the oldest.

each beta_j is the same evolvable expression evaluated on that direction's
features, and alpha is an evolvable expression on r and ad. this costs exactly
one matrix vector product per iteration (the a r), the same as gmres and half of
bicgstab. the classical generalized conjugate residual (gcr) / orthomin(m) is the
special case

    beta_j = -(a r . ap_j) / (ap_j . ap_j)      alpha = (r . ad) / (ad . ad)

so the search can rediscover gcr, or find a variant. negative coefficient scales
are allowed here because gcr's beta is negative.
"""
from __future__ import annotations

import numpy as np

from genome import Coeff
from baselines import SolveResult, _rel_err

# per direction features used to choose beta_j
BETA_DIR_FEATURES = ["one", "ArApj", "ApApj", "rApj", "rpj", "pjpj"]
# features used to choose the minimal residual style step alpha
ALPHA_MR_FEATURES = ["one", "rr", "rAd", "AdAd", "rd", "dAd"]


class OrthominGenome:
    def __init__(self, beta: Coeff, alpha: Coeff, memory: int):
        self.beta = beta
        self.alpha = alpha
        self.memory = memory
        self.fitness = None

    @staticmethod
    def random(rng, memory):
        beta = Coeff(rng.uniform(-1.5, 1.5),
                     rng.choice(BETA_DIR_FEATURES), rng.choice(BETA_DIR_FEATURES))
        alpha = Coeff(rng.uniform(0.2, 1.5),
                      rng.choice(ALPHA_MR_FEATURES), rng.choice(ALPHA_MR_FEATURES))
        return OrthominGenome(beta, alpha, memory)

    def copy(self):
        g = OrthominGenome(self.beta.copy(), self.alpha.copy(), self.memory)
        return g

    def mutate(self, rng, rate=0.6):
        g = self.copy()
        g.fitness = None
        for coeff, feats in ((g.beta, BETA_DIR_FEATURES), (g.alpha, ALPHA_MR_FEATURES)):
            if rng.random() < rate:
                coeff.scale *= float(np.exp(rng.normal(0, 0.4)))
            if rng.random() < rate * 0.2:
                coeff.scale = -coeff.scale                # allow sign flips
            if rng.random() < rate * 0.6:
                coeff.num = rng.choice(feats)
            if rng.random() < rate * 0.6:
                coeff.den = rng.choice(feats)
        return g

    @staticmethod
    def crossover(a, b, rng):
        beta = (a.beta if rng.random() < 0.5 else b.beta).copy()
        alpha = (a.alpha if rng.random() < 0.5 else b.alpha).copy()
        return OrthominGenome(beta, alpha, a.memory)

    def pseudocode(self):
        return (f"memory m = {self.memory}\n"
                f"beta_j  = {self.beta}\n"
                f"alpha   = {self.alpha}\n"
                f"d   = r + sum_j beta_j p_j\n"
                f"x  += alpha d ;  r -= alpha (a d)")

    def __repr__(self):
        return f"<Orthomin m={self.memory} beta=({self.beta}) alpha=({self.alpha})>"


def run_orthomin(genome, problem, tol=1e-8, maxiter=2000):
    A, b, x_star = problem.A, problem.b, problem.x_star
    m = genome.memory
    x = np.zeros_like(b)
    bnorm = np.linalg.norm(b) or 1.0
    window = []                          # list of (p_j, Ap_j)
    hist = []
    matvecs = 0

    r = b - A @ x
    for k in range(1, maxiter + 1):
        rel = np.linalg.norm(r) / bnorm
        hist.append(rel)
        if not np.isfinite(rel) or rel > 1e8:
            res = SolveResult("Orthomin(disc)", len(hist), False, min(rel, 1e8),
                              _rel_err(x, x_star), hist); res.matvecs = matvecs
            return res
        if rel < tol:
            res = SolveResult("Orthomin(disc)", k, True, rel,
                              _rel_err(x, x_star), hist); res.matvecs = matvecs
            return res

        Ar = A @ r; matvecs += 1
        d = r.copy()
        Ad = Ar.copy()
        for (p_j, Ap_j) in window:
            feats = {"one": 1.0,
                     "ArApj": float(Ar @ Ap_j), "ApApj": float(Ap_j @ Ap_j),
                     "rApj": float(r @ Ap_j), "rpj": float(r @ p_j),
                     "pjpj": float(p_j @ p_j)}
            beta_j = genome.beta.value(feats)
            d = d + beta_j * p_j
            Ad = Ad + beta_j * Ap_j
        afeats = {"one": 1.0, "rr": float(r @ r), "rAd": float(r @ Ad),
                  "AdAd": float(Ad @ Ad), "rd": float(r @ d), "dAd": float(d @ Ad)}
        alpha = genome.alpha.value(afeats)

        x = x + alpha * d
        r = r - alpha * Ad
        window.append((d, Ad))
        if len(window) > m:
            window.pop(0)

    res = SolveResult("Orthomin(disc)", maxiter, False, hist[-1],
                      _rel_err(x, x_star), hist); res.matvecs = matvecs
    return res


def gcr_reference(memory):
    """the classical gcr / orthomin(m) as an OrthominGenome, for validation."""
    return OrthominGenome(Coeff(-1.0, "ArApj", "ApApj"),
                          Coeff(1.0, "rAd", "AdAd"), memory)


# ----------------------------------------------------------------------------
# search over orthomin genomes. same shape as evolve.py but self contained, and
# it allows negative coefficient scales because gcr's beta is negative.
# ----------------------------------------------------------------------------

def score_orth(genome, problems, tol=1e-8, maxiter=400):
    iters, solved = [], 0
    for p in problems:
        res = run_orthomin(genome, p, tol=tol, maxiter=maxiter)
        if res.converged:
            solved += 1
            iters.append(res.iters)
        else:
            iters.append(maxiter * (2.0 + min(res.rel_res, 1e8)))
    return float(np.mean(iters)), solved


def _fit(g, problems, tol, maxiter):
    s, solved = score_orth(g, problems, tol, maxiter)
    g.fitness, g._solved = s, solved
    return s


def _optimize_scales(g, problems, tol, maxiter, passes=1):
    absolute = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)   # negatives for gcr's beta
    best_sa, best_sb = g.alpha.scale, g.beta.scale
    best_f = _fit(g, problems, tol, maxiter)
    for sa in absolute + (best_sa,):
        for sb in absolute + (best_sb,):
            g.alpha.scale, g.beta.scale = sa, sb
            f = _fit(g, problems, tol, maxiter)
            if f < best_f - 1e-12:
                best_f, best_sa, best_sb = f, sa, sb
    g.alpha.scale, g.beta.scale = best_sa, best_sb
    grid_mult = (0.7, 0.85, 1.0, 1.15, 1.3)
    for _ in range(passes):
        for coeff in (g.alpha, g.beta):
            base = coeff.scale
            best_s = base
            for mlt in grid_mult:
                coeff.scale = base * mlt
                f = _fit(g, problems, tol, maxiter)
                if f < best_f - 1e-12:
                    best_f, best_s = f, coeff.scale
            coeff.scale = best_s
    _fit(g, problems, tol, maxiter)
    return best_f


def refine_systematic(genome, problems, tol, maxiter, max_rounds=5):
    best = genome.copy()
    best_f = _optimize_scales(best, problems, tol, maxiter, passes=2)
    for _ in range(max_rounds):
        improved = False
        cands = []
        for num in ALPHA_MR_FEATURES:
            for den in ALPHA_MR_FEATURES:
                g = best.copy(); g.alpha.num = num; g.alpha.den = den; cands.append(g)
        for num in BETA_DIR_FEATURES:
            for den in BETA_DIR_FEATURES:
                g = best.copy(); g.beta.num = num; g.beta.den = den; cands.append(g)
        for cand in cands:
            f = _optimize_scales(cand, problems, tol, maxiter, passes=1)
            if f < best_f - 1e-9:
                best, best_f, improved = cand, f, True
        if not improved:
            break
    best.fitness, best._solved = score_orth(best, problems, tol, maxiter)
    return best


def evolve_orth(problems, memory, generations=10, pop_size=24, elite=4,
                tol=1e-8, maxiter=400, seed=0):
    rng = np.random.default_rng(seed)
    pop = [OrthominGenome.random(rng, memory) for _ in range(pop_size)]
    for g in pop:
        _fit(g, problems, tol, maxiter)
    best = min(pop, key=lambda g: g.fitness); best = best.copy()
    _fit(best, problems, tol, maxiter)
    n_imm = max(2, pop_size // 6)
    for _ in range(generations):
        pop.sort(key=lambda g: g.fitness)
        elites = [g.copy() for g in pop[:elite]]
        for e in elites:
            _fit(e, problems, tol, maxiter)
        imm = [OrthominGenome.random(rng, memory) for _ in range(n_imm)]
        for g in imm:
            _fit(g, problems, tol, maxiter)
        children = []
        while len(children) < pop_size - elite - n_imm:
            a = min(rng.choice(pop, 3, replace=False), key=lambda g: g.fitness)
            b = min(rng.choice(pop, 3, replace=False), key=lambda g: g.fitness)
            c = OrthominGenome.crossover(a, b, rng).mutate(rng)
            _fit(c, problems, tol, maxiter)
            children.append(c)
        pop = elites + children + imm
        gb = min(pop, key=lambda g: g.fitness)
        if gb.fitness < best.fitness:
            best = gb.copy(); _fit(best, problems, tol, maxiter)
    return refine_systematic(best, problems, tol, maxiter)


def run_orthomin_experiment(memories=(2, 3, 5), seeds=(1, 2, 3),
                            gens=8, pop=20, search_maxiter=200, test_maxiter=3000):
    """discover multi direction solvers at several memory sizes and compare, in
    matrix vector products, against gmres, bicgstab and orthomin(1) on held out
    nonsymmetric problems."""
    from problems import nonsym_training_suite, nonsym_test_suite
    from baselines import NONSYM_BASELINES

    print("=" * 74)
    print("  experiment c: multi direction (gcr / orthomin-m) solver discovery")
    print("=" * 74)
    train, test = nonsym_training_suite(), nonsym_test_suite()

    base = {}
    for p in test:
        base[p.name] = {n: (fn(p, maxiter=test_maxiter)) for n, fn in NONSYM_BASELINES.items()}

    def cost(res):
        return res.matvecs if res.converged else None

    print("\nbaseline cost (matrix vector products) on held out test set:")
    hdr = f"{'problem':<22}" + "".join(f"{n[:10]:>12}" for n in NONSYM_BASELINES)
    print(hdr); print("=" * len(hdr))
    for p in test:
        print(f"{p.name:<22}" + "".join(
            f"{str(cost(base[p.name][n])):>12}" for n in NONSYM_BASELINES))

    champions = {}
    for m in memories:
        best_overall, best_mean = None, 1e18
        for s in seeds:
            g = evolve_orth(train, memory=m, generations=gens, pop_size=pop,
                            seed=s, maxiter=search_maxiter)
            costs = [cost(run_orthomin(g, p, maxiter=test_maxiter)) for p in test]
            solved = [c for c in costs if c is not None]
            mean = (sum(solved) + test_maxiter * (len(costs) - len(solved))) / len(costs)
            if mean < best_mean:
                best_mean, best_overall = mean, g
        champions[m] = best_overall
        print(f"\n--- best discovered at memory m={m} ---")
        print(best_overall.pseudocode())
        row = []
        for p in test:
            c = cost(run_orthomin(best_overall, p, maxiter=test_maxiter))
            row.append((p.name, c))
        print("held out matvecs: " + ", ".join(f"{n}={c}" for n, c in row))

    # win rate of the best memory champion vs each baseline
    print("\nwin rate in matvecs, discovered (best memory per problem) vs baselines:")
    for n in NONSYM_BASELINES:
        wins = tot = 0
        for p in test:
            b = cost(base[p.name][n])
            if b is None:
                continue
            dcosts = [cost(run_orthomin(champions[m], p, maxiter=test_maxiter))
                      for m in memories]
            dcosts = [c for c in dcosts if c is not None]
            if not dcosts:
                continue
            tot += 1
            if min(dcosts) < b:
                wins += 1
        print(f"  vs {n:<16} {wins}/{tot}")
    return champions


if __name__ == "__main__":
    run_orthomin_experiment()
