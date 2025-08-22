from maxwell_L import run_maxwell
from div_L import run_div
from kellogg import run_kellogg
from grad import run_grad
import sys
systems = {"curl": run_maxwell, "div": run_div, "kellogg": run_kellogg, "grad": run_grad}

import argparse

def run_system(system="curl", p=None, theta=0.5, lam_alg=0.01, alpha = 2/3, dim=1e5, solver="direct", experiment="bump"):
    if not p:
        for p in range(1,5):
            if system == "curl" or system == "div":
                systems[system](p, theta, lam_alg, alpha, dim, solver, experiment)
            else:
                systems[system](p, theta, lam_alg, dim, solver)
    else:
        if system == "curl" or system == "div":
            systems[system](p, theta, lam_alg, alpha, dim, solver, experiment)
        else:
            systems[system](p, theta, lam_alg, dim, solver)
    return
    
def main():
    parser = argparse.ArgumentParser(description="Run Solver")

    parser.add_argument("--system", type=str, choices=["curl", "div", "grad", "kellogg"],
                        help="Which system do you want to solve")

    parser.add_argument("--p", type=int, default=None, help="Polynomial degree (None runs all)")
    parser.add_argument("--theta", type=float, default=0.5, help="Theta parameter")
    parser.add_argument("--lam_alg", type=float, default=0.01, help="Lambda algorithmic parameter")
    parser.add_argument("--alpha", type=float, default=2/3, help="Alpha parameter (only for curl/div)")
    parser.add_argument("--dim", type=float, default=1e4, help="Dimension of finest function space")
    parser.add_argument("--solver", type=str, default="direct", choices=["direct", "mg"],
                        help="Solver type (for now the only choices are direct or multigrid through patch relaxation).")
    parser.add_argument("--experiment", type=str, default="bump", help="Which experiment to run for div/curl, options=[bump, donut, unknown]")

    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    print(f"SYSTEM: {args.system} for EXPERIMENT {args.experiment} with SOLVER: {args.solver}, P: {args.p}, THETA: {args.theta}, LAM_ALG: {args.lam_alg}, ALPHA: {args.alpha}, DIM: {args.dim}")
    run_system(args.system, args.p, args.theta, args.lam_alg, args.alpha, args.dim, args.solver, args.experiment)

if __name__ == "__main__":
    main()
