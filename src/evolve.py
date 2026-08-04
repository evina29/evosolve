"""
evolutionary search over algorithm genomes.

fitness = how quickly a genome solves the whole training suite. we use the mean
(over problems) of the iteration count needed to reach the tolerance, with a
heavy penalty for any problem it fails to solve (diverges or hits maxiter). this
is a minimisation problem: lower score = better algorithm.
"""
from __future__ import annotations

import numpy as np

from genome import Genome, run_genome


def score(genome, problems, tol=1e-8, maxiter=500):
    """lower is better. returns (score, per_problem_iters, num_solved)."""
    iters = []
    solved = 0
    for prob in problems:
        res = run_genome(genome, prob, tol=tol, maxiter=maxiter)
        if res.converged:
            solved += 1
            iters.append(res.iters)
        else:
            # penalty: worse than any converging run, scaled by how bad the residual is
            penalty = maxiter * (2.0 + min(res.rel_res, 1e8))
            iters.append(penalty)
    return float(np.mean(iters)), iters, solved


def _fit(g, problems, tol, maxiter):
    s, _, solved = score(g, problems, tol, maxiter)
    g.fitness, g._solved = s, solved
    return s


def _optimize_scales(g, problems, tol, maxiter, passes=1):
    """optimise the two coefficient scales for a fixed structure.

    two phases so that coupled scale optima are reachable:
      1. a joint 2d grid over (alpha.scale, beta.scale) so both scales can snap
         to their natural magnitudes together (e.g. both go to 1.0, which is what
         turns a correct structure into exact conjugate gradient).
      2. per coefficient coordinate descent to fine tune around the joint best.
    """
    absolute = (0.0, 0.5, 1.0, 2.0)          # natural magnitudes for ratio features
    best_sa, best_sb = g.alpha.scale, g.beta.scale
    best_f = _fit(g, problems, tol, maxiter)

    # phase 1: joint grid (includes carrying over current scales)
    for sa in absolute + (best_sa,):
        for sb in absolute + (best_sb,):
            g.alpha.scale, g.beta.scale = sa, sb
            f = _fit(g, problems, tol, maxiter)
            if f < best_f - 1e-12:
                best_f, best_sa, best_sb = f, sa, sb
    g.alpha.scale, g.beta.scale = best_sa, best_sb

    # phase 2: multiplicative fine tuning
    grid_mult = (0.7, 0.85, 1.0, 1.15, 1.3)
    for _ in range(passes):
        for coeff in (g.alpha, g.beta):
            base = coeff.scale
            best_s = base
            for m in grid_mult:
                coeff.scale = base * m
                f = _fit(g, problems, tol, maxiter)
                if f < best_f - 1e-12:
                    best_f, best_s = f, coeff.scale
            coeff.scale = best_s
    _fit(g, problems, tol, maxiter)
    return best_f


def refine(genome, problems, tol, maxiter, rng, steps=25):
    """
    memetic local search. each proposal changes structure (a feature swap) or
    jitters a scale, then re optimises both scales before being judged. this lets
    the search cross coupled optima (e.g. reach the exact conjugate gradient form
    beta=rr/rr_prev, alpha=rd/dAd with unit scales).
    """
    from genome import ALPHA_FEATURES, BETA_FEATURES
    best = genome.copy()
    best_f = _optimize_scales(best, problems, tol, maxiter)
    for _ in range(steps):
        cand = best.copy()
        roll = rng.random()
        if roll < 0.6:                                   # structural feature swap
            which = rng.random()
            if which < 0.25:  cand.alpha.num = rng.choice(ALPHA_FEATURES)
            elif which < 0.5: cand.alpha.den = rng.choice(ALPHA_FEATURES)
            elif which < 0.75:cand.beta.num = rng.choice(BETA_FEATURES)
            else:             cand.beta.den = rng.choice(BETA_FEATURES)
        else:                                            # scale jitter
            if rng.random() < 0.5:
                cand.alpha.scale *= float(np.exp(rng.normal(0, 0.2)))
            else:
                cand.beta.scale *= float(np.exp(rng.normal(0, 0.2)))
        f = _optimize_scales(cand, problems, tol, maxiter)
        if f < best_f - 1e-9:
            best, best_f = cand, f
    best.fitness, best._iters, best._solved = score(best, problems, tol, maxiter)
    return best


def refine_systematic(genome, problems, tol, maxiter, max_rounds=6):
    """
    deterministic memetic hill climb (variable neighbourhood style). each round
    enumerates every single feature neighbour of the current best (change one of
    alpha.num, alpha.den, beta.num, beta.den), optimises both scales for each
    neighbour, and moves to the best improving one. repeats until no single
    structural change helps, i.e. a local optimum. far more reliable than random
    sampling at crossing coupled optima (e.g. climbing into the exact cg form).
    """
    from genome import ALPHA_FEATURES, BETA_FEATURES

    best = genome.copy()
    best_f = _optimize_scales(best, problems, tol, maxiter, passes=2)
    for _ in range(max_rounds):
        improved = False
        # enumerate every (numerator, denominator) pair for each coefficient in
        # turn (block coordinate descent over structure). enumerating both slots
        # together lets the climb make two simultaneous changes, e.g. reach the
        # cg form beta = rr / rr_prev from an unrelated basin in one move.
        candidates = []
        for num in ALPHA_FEATURES:
            for den in ALPHA_FEATURES:
                g = best.copy(); g.alpha.num = num; g.alpha.den = den
                candidates.append(g)
        for num in BETA_FEATURES:
            for den in BETA_FEATURES:
                g = best.copy(); g.beta.num = num; g.beta.den = den
                candidates.append(g)
        for cand in candidates:
            f = _optimize_scales(cand, problems, tol, maxiter, passes=2)
            if f < best_f - 1e-9:
                best, best_f, improved = cand, f, True
        if not improved:
            break
    best.fitness, best._iters, best._solved = score(best, problems, tol, maxiter)
    return best


def evolve(problems, generations=40, pop_size=60, elite=6, tol=1e-8,
           maxiter=500, seed=0, verbose=True):
    rng = np.random.default_rng(seed)

    # seed the population with random genomes
    pop = [Genome.random(rng) for _ in range(pop_size)]
    for g in pop:
        g.fitness, g._iters, g._solved = score(g, problems, tol, maxiter)

    history = []
    best = min(pop, key=lambda g: g.fitness)
    n_immigrants = max(2, pop_size // 6)   # fresh random blood each generation
    stagnation = 0

    for gen in range(1, generations + 1):
        pop.sort(key=lambda g: g.fitness)
        elites = [g.copy() for g in pop[:elite]]
        for e, src in zip(elites, pop[:elite]):
            e.fitness, e._iters, e._solved = src.fitness, src._iters, src._solved
        # memetic refinement of the champion (drives coeffs to exact optima)
        elites[0] = refine(elites[0], problems, tol, maxiter, rng, steps=6)

        # random immigrants preserve diversity and help escape local optima
        immigrants = []
        for _ in range(n_immigrants):
            g = Genome.random(rng)
            g.fitness, g._iters, g._solved = score(g, problems, tol, maxiter)
            immigrants.append(g)

        # children via tournament selection, crossover and mutation
        children = []
        rate = 0.6 + min(0.3, 0.03 * stagnation)   # mutate harder when stuck
        while len(children) < pop_size - elite - n_immigrants:
            a = _tournament(pop, rng)
            b = _tournament(pop, rng)
            child = Genome.crossover(a, b, rng)
            child = child.mutate(rng, rate=rate)
            child.fitness, child._iters, child._solved = score(child, problems, tol, maxiter)
            children.append(child)

        pop = elites + children + immigrants
        gen_best = min(pop, key=lambda g: g.fitness)
        if gen_best.fitness < best.fitness - 1e-9:
            best = gen_best.copy()
            best.fitness, best._iters, best._solved = (
                gen_best.fitness, gen_best._iters, gen_best._solved)
            stagnation = 0
        else:
            stagnation += 1

        history.append(best.fitness)
        if verbose:
            print(f"gen {gen:>3}/{generations}  best_score={best.fitness:8.2f}  "
                  f"solved={best._solved}/{len(problems)}  beta=({best.beta}) alpha=({best.alpha})")

    # final polish: deterministic structural hill climb to the local optimum
    if verbose:
        print("--- final systematic structural refinement ---")
    best = refine_systematic(best, problems, tol, maxiter)
    if verbose:
        print(f"polished best_score={best.fitness:.2f} solved={best._solved}/{len(problems)} "
              f"beta=({best.beta}) alpha=({best.alpha})")
    return best, history


def _tournament(pop, rng, k=3):
    contenders = rng.choice(len(pop), size=k, replace=False)
    return min((pop[i] for i in contenders), key=lambda g: g.fitness)


if __name__ == "__main__":
    from problems import training_suite
    best, hist = evolve(training_suite(), generations=15, pop_size=40, seed=1)
    print("\nbest genome found:")
    print(best.pseudocode())
