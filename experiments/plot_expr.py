import csv
import numpy as np
from matplotlib import pyplot as plt
from itertools import accumulate
import argparse

systems = {"curl": "maxwell_L", "div": "div_L", "kellogg": "kellogg", "grad": "grad"}


def plot_system(system="curl", p=None, theta=0.5, lam_alg=0.01, alpha = 2/3, dim=1e4, solver="direct", dir_name = None, w_uniform=False, unif_dir=None):
    sys_name = systems[system]
    if not dir_name:
        if system == "curl" or system == "div":
            dir_name = f"output/{sys_name}_{solver}/theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}"
        else:
            dir_name = f"output/{sys_name}_{solver}/theta={theta}_lam={lam_alg}_dim={dim}"
    if not unif_dir:
        if system == "curl" or system == "div":
            unif_dir = f"output/{sys_name}_{solver}/theta=0.0_lam={lam_alg}_alpha={alpha}_dim={dim}"
        else:
            unif_dir = f"output/{sys_name}_{solver}/theta=0.0_lam={lam_alg}_dim={dim}"

    dofs = {}
    errors_est = {}
    errors_true = {}
    times = {}
    if p == None:
        for pp in range(1,5):
            print(f"Plotting data for p={pp}")
            with open(f"{dir_name}/{pp}/dat.csv", "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            columns = list(zip(*rows))
            dofs[pp] = np.array(columns[0][1:], dtype=float)
            errors_est[pp] = np.array(columns[1][1:], dtype=float)
            errors_true[pp] = np.array(columns[2][1:], dtype=float)
            times[pp] = np.array(columns[4][1:], dtype=float)

            dof = dofs[pp]
            est = errors_est[pp]
            tim = times[pp]

            plot_single_est_convergence(dir_name, pp, system, solver, dof, est)
            plot_single_time_convergence(dir_name, system, solver, pp, tim, est)

        plot_joint_est_convergence(dir_name, system, solver, dofs, errors_est)
        plot_joint_true_convergence(dir_name, system, solver, dofs, errors_true)
        plot_joint_time_convergence(dir_name, system, solver, times, errors_est)

        if w_uniform:
            udofs = {}
            uerrors_est = {}
            uerrors_true = {}
            utimes = {}
            for pp in range(1,5):
                with open(f"{unif_dir}/{pp}/dat.csv", "r", newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                columns = list(zip(*rows))
                udofs[pp] = np.array(columns[0][1:], dtype=float)
                uerrors_est[pp] = np.array(columns[1][1:], dtype=float)
                uerrors_true[pp] = np.array(columns[2][1:], dtype=float)
                utimes[pp] = np.array(columns[4][1:], dtype=float)

            
            plot_joint_w_uniform(dir_name, system, solver, dofs, udofs, errors_est, uerrors_est)
    else:
        with open(f"{dir_name}/{p}/dat.csv", "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        columns = list(zip(*rows))

        dof = np.array(columns[0][1:], dtype=float)
        est = np.array(columns[1][1:], dtype=float)
        tim = np.array(columns[4][1:], dtype=float)

        plot_single_est_convergence(dir_name, p, system, solver, dof, est)
        plot_single_time_convergence(dir_name, system, solver, p, tim, est)
        
       

def plot_single_est_convergence(dir_name, p, system, solver, dof, est):
        plt.figure(figsize=(8, 6))
        plt.grid(True)
        plt.loglog(dof[1:], est[1:], '-o', alpha = 0.7, markersize = 4)
        scaling = est[1] / dof[1]**-0.5
        plt.loglog(dof[1:], scaling * dof[1:]**-0.5, '--', alpha=0.5, color="lightcoral", label="x^{-0.5}")
        scaling = est[1] / dof[1]**-0.1
        plt.loglog(dof[1:], scaling * dof[1:]**-0.1, '--', alpha = 0.5, color='lawngreen', label = "x^{-0.1}")
        scaling = est[1] / dof[1]**-1
        plt.loglog(dof[1:], scaling * dof[1:]**-1, '--', alpha = 0.5, color = 'aqua', label = "x^{-1}")
        scaling = est[1] / dof[1]**-2
        plt.loglog(dof[1:], scaling * dof[1:]**-2, '--', alpha = 0.5, color = 'indigo', label = "x^{-2}")
        plt.xlabel("Number of degrees of freedom")
        plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
        plt.title(f"Estimated Error Convergence ({system}, {solver}) for p={p}")
        plt.legend()
        plt.savefig(f"{dir_name}/{p}_single_convergence.png")
    
def plot_joint_est_convergence(dir_name, system, solver, dofs, errors_est):
    colors = ['blue', 'green', 'red', 'purple']
    scaling_exp = {1: -0.5, 2: -1, 3: -2, 4: -2}
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    for p in range(4):
        plt.loglog(dofs[p+1], errors_est[p+1], '-o', color=colors[p], alpha = 0.5, markersize=2.5, label=f"p={p+1}")
        # scaling = errors_est[p+1][20] / dofs[p+1][20]**scaling_exp[p+1]
        # plt.loglog(dofs[p+1][20:],scaling * dofs[p+1][20:]**scaling_exp[p+1], '--', alpha=0.5, color= colors[p], label=f"x^{scaling_exp[p+1]}")


    plt.xlabel("Number of degrees of freedom")
    plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
    plt.legend()
    plt.title(f"Estimated Error Convergence ({system}, {solver})")
    plt.savefig(f"{dir_name}/joint_adaptive_convergence_est.png")

def plot_joint_true_convergence(dir_name, system, solver, dofs, errors_true):
    colors = ['blue', 'green', 'red', 'purple']  
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    for p in range(4):
        plt.loglog(dofs[p+1], errors_true[p+1], '-.', color=colors[p], alpha = 0.5, label=f"p={p+1}")

    plt.xlabel("Number of degrees of freedom")
    plt.ylabel(r"True error norm $\|u - u_h\|$")
    plt.legend()
    plt.title(f"True Error Convergence ({system}, {solver})")
    plt.savefig(f"{dir_name}/joint_adaptive_convergence_true.png")


def plot_single_time_convergence(dir_name, system, solver, p, times, errors_est):
    times = np.array(list(accumulate(times)))
    colors = ['blue', 'green', 'red', 'purple'] 
    c = colors[p-1]
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.loglog(times, errors_est, '-o', alpha = 0.6, color=c, markersize = 3)
    scaling = errors_est[0] / times[0]**-0.5
    plt.loglog(times, scaling * times**-0.5, '--', alpha=0.5, color="lightcoral", label="t^{-0.5}")
    scaling = errors_est[0] / times[0]**-0.1
    plt.loglog(times, scaling * times**-0.1, '--', alpha = 0.5, color='lawngreen', label = "t^{-0.1}")
    scaling = errors_est[0] / times[0]**-1
    plt.loglog(times, scaling * times**-1, '--', alpha = 0.5, color = 'aqua', label = "t^{-1}")
    scaling = errors_est[0] / times[0]**-2
    plt.loglog(times, scaling * times**-2, '--', alpha = 0.5, color = 'indigo', label = "t^{-2}")
    plt.xlabel("Cumulative Runtime")
    plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
    plt.title(f"Estimator vs Cumulative Runtime ({system}, {solver}) for p={p}")
    plt.legend()
    plt.savefig(f"{dir_name}/{p}_runtime_convergence.png")

def plot_joint_time_convergence(dir_name, system, solver, times, errors_est):

    colors = ['blue', 'green', 'red', 'purple']  
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    for p in range(4):
        rtimes = np.array(list(accumulate(times[p+1])))
        plt.loglog(rtimes, errors_est[p+1], '-o', color=colors[p], alpha = 0.5, markersize=2.5, label=f"p={p+1}")

    plt.xlabel("Cumulative Runtime")
    plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
    plt.legend()
    plt.title(f"Estimator vs Cumulative Runtime ({system}, {solver})")
    plt.savefig(f"{dir_name}/joint_runtime_convergence.png")

def plot_joint_w_uniform(dir_name, system, solver, dofs, udofs, errors_est, uerrors_est):
    colors = ['blue', 'green', 'red', 'purple']
    scaling_exp = {1: -0.5, 2: -1, 3: -2, 4: -2}
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    for p in range(4):
        plt.loglog(dofs[p+1], errors_est[p+1], '-o', color=colors[p], alpha = 0.7, markersize=3, label=f"adaptive: {p+1}")
        plt.loglog(udofs[p+1], uerrors_est[p+1], '--v', color=colors[p], alpha = 0.5, markersize=3, label=f"uniform: {p+1}")

    plt.xlabel("Number of degrees of freedom")
    plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
    plt.legend()
    plt.title(f"Estimated Error Convergence ({system}, {solver})")
    plt.savefig(f"{dir_name}/joint_adaptive_w_uniform.png")

def main():
    parser = argparse.ArgumentParser(description="Plot system results")
    parser.add_argument("--system", type=str, choices=["curl", "div", "grad", "kellogg"],
                        help="Which system do you want to solve")
    parser.add_argument("--p", type=int, default=None, help="Polynomial degree")
    parser.add_argument("--theta", type=float, default=0.5, help="Theta parameter")
    parser.add_argument("--lam_alg", type=float, default=0.01, help="Lambda algorithmic parameter")
    parser.add_argument("--alpha", type=float, default=2/3, help="Alpha parameter (only for curl/div)")
    parser.add_argument("--dim", type=float, default=1e4, help="Dimension of finest function space")
    parser.add_argument("--solver", type=str, default="direct", choices=["direct", "mg"],
                        help="Solver type (for now the only choices are direct or multigrid through patch relaxation).")
    parser.add_argument("--dir_name", type=str, default=None, help="Custom data directory (if not using default, rename, etc.)")
    parser.add_argument("--w_uniform", type=bool, default=False, help="Plot convergence vs uniform?")
    parser.add_argument("--unif_dir", type=str, default=None, help="Custom data directory, if plotting with uniform and uniform directory has been moved/renamed")

    args = parser.parse_args()

    print(f"SYSTEM: {args.system} with SOLVER: {args.solver}, P: {args.p}, THETA: {args.theta}, LAM_ALG: {args.lam_alg}, ALPHA: {args.alpha}, DIM: {args.dim}")
    plot_system(
        args.system,
        p=args.p,
        theta=args.theta,
        lam_alg=args.lam_alg,
        alpha=args.alpha,
        dim=args.dim,
        solver=args.solver,
        dir_name=args.dir_name,
        w_uniform=args.w_uniform,
        unif_dir=args.unif_dir
    )

if __name__ == "__main__":
    main()
