"""
end to end driver:

  1. evolve an iterative solver algorithm on the training suite.
  2. benchmark the winner on the held out test suite vs classical methods.
  3. print a table, save convergence plots, and write a summary.

usage:
    python src/main.py                 # default run
    python src/main.py --gens 60 --pop 80 --seed 3
"""
from __future__ import annotations

import argparse
import numpy as np

from problems import training_suite, test_suite
from evolve import evolve, score
from benchmark import benchmark, print_table, save_plots, save_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=500, help="cap during search")
    args = ap.parse_args()

    print("=" * 64)
    print("  automated discovery of iterative linear solver algorithms")
    print("=" * 64)

    train = training_suite()
    test = test_suite()
    print(f"\ntraining problems ({len(train)}): " + ", ".join(p.name for p in train))
    print(f"held out test problems ({len(test)}): " + ", ".join(p.name for p in test))

    print("\nevolutionary search")
    best, hist = evolve(train, generations=args.gens, pop_size=args.pop,
                        seed=args.seed, maxiter=args.maxiter)

    print("\nbest discovered algorithm")
    print(best.pseudocode())
    tr_score, tr_iters, tr_solved = score(best, train, maxiter=args.maxiter)
    print(f"\ntraining score (mean iters): {tr_score:.2f}   solved {tr_solved}/{len(train)}")

    print("\nbenchmark on held out test problems")
    table = benchmark(best, test, tol=1e-8, maxiter=2000)
    print_table(table)

    plot_path = save_plots(best, table, test)
    summ_path, summ_text = save_summary(best, table, hist)
    print("\n" + summ_text)
    print(f"\nsaved convergence plot to {plot_path}")
    print(f"saved summary to {summ_path}")


if __name__ == "__main__":
    main()
