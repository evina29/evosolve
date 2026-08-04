"""
benchmark a discovered genome against the classical baselines on held out
problems, print a comparison table, and save convergence plots plus a summary.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import BASELINES
from genome import run_genome

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def benchmark(genome, problems, tol=1e-8, maxiter=2000):
    """return {problem_name: {method_name: SolveResult}}."""
    table = {}
    for prob in problems:
        row = {}
        for name, fn in BASELINES.items():
            row[name] = fn(prob, tol=tol, maxiter=maxiter)
        row["discovered"] = run_genome(genome, prob, tol=tol, maxiter=maxiter)
        table[prob.name] = row
    return table


def print_table(table):
    methods = ["jacobi", "gauss_seidel", "steepest_descent",
               "conjugate_gradient", "discovered"]
    header = f"{'problem':<20}" + "".join(f"{m[:16]:>18}" for m in methods)
    print("\niterations to converge (rel. residual < 1e-8):")
    print(header)
    print("=" * len(header))
    for pname, row in table.items():
        cells = []
        for m in methods:
            res = row[m]
            cells.append((f"{res.iters}" if res.converged else f"{res.iters}*"))
        print(f"{pname:<20}" + "".join(f"{c:>18}" for c in cells))
    print("(* = did not reach tolerance within maxiter)")


def save_plots(genome, table, problems):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ncols = 2
    nrows = int(np.ceil(len(problems) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, prob in zip(axes, problems):
        row = table[prob.name]
        for m in ["jacobi", "gauss_seidel", "steepest_descent",
                  "conjugate_gradient", "discovered"]:
            h = row[m].res_history
            style = "-" if m != "discovered" else "--"
            lw = 1.4 if m != "discovered" else 2.6
            ax.semilogy(range(1, len(h) + 1), h, style, lw=lw, label=m)
        ax.set_title(prob.name)
        ax.set_xlabel("iteration")
        ax.set_ylabel("relative residual")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[len(problems):]:
        ax.axis("off")
    fig.suptitle("discovered algorithm vs classical solvers", fontsize=14)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "convergence.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def save_summary(genome, table, history=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = []
    lines.append("discovered algorithm")
    lines.append("=" * 60)
    lines.append(genome.pseudocode())
    lines.append("")
    lines.append("head to head vs conjugate gradient (held out problems)")
    lines.append("=" * 60)
    wins = ties = losses = 0
    for pname, row in table.items():
        d = row["discovered"]; c = row["conjugate_gradient"]
        di = d.iters if d.converged else 10 ** 9
        ci = c.iters if c.converged else 10 ** 9
        verdict = "win " if di < ci else ("tie " if di == ci else "loss")
        if di < ci: wins += 1
        elif di == ci: ties += 1
        else: losses += 1
        lines.append(f"{pname:<20} discovered={d.iters}{'' if d.converged else '*'}"
                     f"  cg={c.iters}{'' if c.converged else '*'}  [{verdict}]")
    lines.append("")
    lines.append(f"vs cg    wins:{wins}  ties:{ties}  losses:{losses}")
    text = "\n".join(lines)
    out = os.path.join(RESULTS_DIR, "summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return out, text
