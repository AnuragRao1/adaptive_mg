import numpy as np
dot_prod = np.dot
from firedrake import *
from netgen.occ import *
from firedrake.mg.ufl_utils import coarsen
from firedrake.dmhooks import get_appctx
from firedrake import dmhooks
from firedrake.solving_utils import _SNESContext
from firedrake.mg.utils import get_level
from itertools import accumulate


import time
import csv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager


    
def run_system(p=1, theta=0.5, lam_alg=0.01, alpha = 2/3, dim=1e3):
    def solve_div(mesh, p, alpha, u_prev, u_real, params, uniform):
        V = FunctionSpace(mesh, "BDM", p)
        W = FunctionSpace(mesh, "BDM", p) 
        uh = u_prev
        v = TestFunction(W)
        bc = DirichletBC(V, u_real, "on_boundary")
        alpha = Constant(alpha)

        x, y = SpatialCoordinate(mesh)
        r = sqrt(x**2 + y**2)
        theta = atan2(y, x)
        theta = conditional(lt(theta, 0), theta + 2 * pi, theta) # map to [0 , 2pi]

        f_expr = u_real - grad(div(u_real))
        f = Function(V).interpolate(f_expr)

        F = (inner(uh,v) + inner(div(uh), div(v)) - inner(f, v)) * dx

        problem = NonlinearVariationalProblem(F, uh, bc)

        if not uniform:
            dm = uh.function_space().dm
            old_appctx = get_appctx(dm)
            mat_type = params["mat_type"]
            appctx = _SNESContext(problem, mat_type, mat_type, old_appctx)
            appctx.transfer_manager = atm
            solver = NonlinearVariationalSolver(problem, solver_parameters=params)
            solver.set_transfer_manager(atm)
            # with dmhooks.add_hooks(dm, solver, appctx=appctx, save=False):
            #     coarsen(problem, coarsen)

        _, level = get_level(mesh)
        with PETSc.Log.Event(f"adaptive_{level}"):
            solver.solve()

        return uh, f


    def estimate_error(mesh, uh, u_real, f):
        W = FunctionSpace(mesh, "DG", 0)
        eta_sq = Function(W)
        w = TestFunction(W)
        h = CellDiameter(mesh) 
        n = FacetNormal(mesh)
        v = CellVolume(mesh)

        G = (
            inner(eta_sq / v, w) * dx 
            - inner(h**2 * (uh - grad(div(uh)) - f)**2, w) * dx
            - inner(h**2 * (grad(uh - grad(div(uh)) - f)**2), w) * dx # added from H(div) norm
            - inner(h('+') * jump(div(uh))**2, w('+')) * dS
            - inner(h('-') * jump(div(uh))**2, w('-')) * dS
            - inner(h * dot(u_real - uh, n)**2, w) * ds
            )
        
        eta_vol = assemble(inner(h**2 * (uh - grad(div(uh)) - f)**2, w) * dx + inner(h**2 * (grad(uh - grad(div(uh)) - f)**2), w) * dx)
        eta_jump = assemble(inner(h('+') * jump(div(uh))**2, w('+')) * dS
            + inner(h('-') * jump(div(uh))**2, w('-')) * dS)
        eta_boundary = assemble(inner(h * dot(grad(u_real - uh), n)**2, w) * ds)
        print(f"Vol: {sqrt(sum(eta_vol.dat.data))}, Jump: {sqrt(sum(eta_jump.dat.data))}, Boundary: {sqrt(sum(eta_boundary.dat.data))}")
        
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)

        eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2
        # VTKFile(f"output/div_L/theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}/{p}/eta_{level}.pvd").write(eta)


        with eta.dat.vec_ro as eta_:  # compute estimate for error in energy norm
            error_est = sqrt(eta_.dot(eta_))
        return (eta, error_est)


    def adapt(mesh, eta):
        W = FunctionSpace(mesh, "DG", 0)
        markers = Function(W)

        # We decide to refine an element if its error indicator
        # is within a fraction of the maximum cellwise error indicator

        # Access storage underlying our Function
        # (a PETSc Vec) to get maximum value of eta
        with eta.dat.vec_ro as eta_:
            eta_max = eta_.max()[1]

        should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
        markers.interpolate(should_refine)

        refined_mesh = mesh.refine_marked_elements(markers)
        return refined_mesh

    def generate_u_real(mesh, p, alpha):
        V = FunctionSpace(mesh, "BDM", p)
        alpha = Constant(alpha)
        x, y = SpatialCoordinate(mesh)
        r = sqrt(x**2 + y**2)
        theta = atan2(y, x)
        theta = conditional(lt(theta, 0), theta + 2 * pi, theta) # map to [0 , 2pi]
        u_real = Function(V).interpolate(as_vector([r**alpha * cos(theta), r**alpha * sin(theta)]))
        return u_real




    rect1 = WorkPlane(Axes((-1,-1,0), n=Z, h=X)).Rectangle(1,2).Face()
    rect2 = WorkPlane(Axes((-1,0,0), n=Z, h=X)).Rectangle(2,1).Face()
    L = rect1 + rect2

    geo = OCCGeometry(L, dim=2)
    ngmsh = geo.GenerateMesh(maxh=0.1)
    mesh = Mesh(ngmsh)


    amh = AdaptiveMeshHierarchy([mesh])
    atm = AdaptiveTransferManager()
    tm = TransferManager()
    
    # ESTABLISH SOLVER PARAMS
    lu = {
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
    assembled_lu = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled": lu
    }
    def mg_params(relax, mat_type="aij"):
        if mat_type == "aij":
            coarse = lu
        else:
            coarse = assembled_lu

        return {
            "mat_type": mat_type,
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_max_it": 1,
                **relax
            },
            "mg_coarse": coarse
        }
    patch_relax = mg_params({
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch": {
        "pc_patch": {
            "construct_type": "star",
            "construct_dim": 0,
            "sub_mat_type": "seqdense",
            "dense_inverse": True,
            "save_operators": True,
            "precompute_element_tensors": True},
        "sub_ksp_type": "preonly",
        "sub_pc_type": "lu"}},
    mat_type="aij")
    
    asm_relax = mg_params({
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMStarPC",
    "pc_star_backend_type": "tinyasm"})

    jacobi_relax = mg_params({"pc_type": "jacobi"}, mat_type="matfree")
    chol = {"mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps"}

    # ITERATIVE LOOP

    # max_iterations = max_iterations

    uniform = False
    if uniform:
        mesh2 = Mesh(ngmsh)
        mh = MeshHierarchy(mesh2, 9)

    error_estimators = {}
    true_errors = []
    dofs = []

    k_l = []
    times = []
    start_time = time.time()
    times.append(start_time)
    level = 0
    csv_file = f"output/div_L/theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}/{p}/dat.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dof", "error_est", "k", "time"])

    while level == 0 or u_k.function_space().dim() <= dim:
        if uniform:
            mesh = mh[level]

        print(f"level {level}")
        V = FunctionSpace(mesh, "BDM", p)
        uh = Function(V, name="solution")
        u_prev = Function(V, name="u_prev")
        dofs.append(uh.function_space().dim())
        
        if level > 0:
            tm.prolong(u_k, uh)

        k = 0
        error_est = 0
        u_real = generate_u_real(mesh, p, alpha)

        while norm(uh - u_prev) > lam_alg * error_est or k == 0:
            k += 1
            u_prev.interpolate(uh)
                        
            start = time.time()
            (uh, f) = solve_div(mesh, p, alpha, uh, u_real, chol, uniform)
            times.append(time.time() - start)
            
            if level % 10 == 0 or level < 15:
                VTKFile(f"output/div_L/theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}/{p}/real_{level}.pvd").write(u_real)
                VTKFile(f"output/div_L/theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}/{p}/{level}_{k}.pvd").write(uh)

            (eta, error_est) = estimate_error(mesh, uh, u_real, f) 
            print("ERROR ESTIMATE: ", error_est)

            if level not in error_estimators:
                error_estimators[level] = [error_est]
            else:
                error_estimators[level].append(error_est)
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dofs[-1], error_est, k, times[-1]])


        u_k = Function(V).interpolate(uh)
        k_l.append(k)

        if not uniform:
            mesh = adapt(mesh, eta)
            if u_k.function_space().dim() <= dim:
                amh.add_mesh(mesh)
                
        
        print(f"DOFS {dofs[-1]}: TIME {times[-1]}")
        level += 1

    final_errors = [error_estimators[key][-1] for key in error_estimators]
    return np.array(dofs, dtype=float), np.array(final_errors), np.array(true_errors), np.array(times)

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    theta = 0.5
    lambda_alg = 0.01
    alpha = 2/3
    dim = 1e4


    errors_true = {}
    errors_est = {}
    dofs = {}
    times = {}
   
    for p in range(1,5):
        (dof, est, true, times) = run_system(p, theta, lambda_alg, alpha, dim)

        with open(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/dat.csv", "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        columns = list(zip(*rows))
        dofs[p] = np.array(columns[0][1:], dtype=float)
        errors_est[p] = np.array(columns[1][1:], dtype=float)
        errors_true[p] = np.array(columns[2][1:], dtype=float)
        times = np.array(columns[3][1:], dtype=float)

        dof = dofs[p]
        est = errors_est[p]

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
        plt.title(f"Estimated Error Convergence p={p}, Multigrid")
        plt.legend()
        plt.savefig(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/single_convergence.png")


    # for p in range(1,5):
    #     dofs[p] = np.load(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/dofs.npy", allow_pickle=True)
    #     errors_est[p] = np.load(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/errors_estimator.npy", allow_pickle=True)
    #     errors_true[p] = np.load(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/errors_true.npy", allow_pickle=True)
    #     times[p] = np.load(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/times.npy", allow_pickle=True)


    # # for i, dat in enumerate(zip(dofs[2].item()[2], errors_est[2])):
    # #     print(i, dat, times[2][i])

    colors = ['blue', 'green', 'red', 'purple']  
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    for p in range(4):
        plt.loglog(dofs[p+1], errors_est[p+1], '-o', color=colors[p], alpha = 0.5, markersize=2.5, label=f"p={p+1}")

    plt.xlabel("Number of degrees of freedom")
    plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
    plt.legend()
    plt.title("Estimated Error Convergence")
    plt.savefig(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/adaptive_convergence_est.png")


    plt.figure(figsize=(8, 6))
    plt.grid(True)
    for p in range(4):
        plt.loglog(dofs[p+1], errors_true[p+1], '-.', color=colors[p], alpha = 0.5, label=f"p={p+1}")

    plt.xlabel("Number of degrees of freedom")
    plt.ylabel(r"True error norm $\|u - u_h\|$")
    plt.legend()
    plt.title("True Error Convergence")
    plt.savefig(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/adaptive_convergence_true.png")

    plt.show()


    # #### SINGLE SAMPLE
    # p = 1
    # (dofs, errors_est, errors_true, times) = run_system(p, theta, lambda_alg, levels)
    # np.save(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/dofs.npy", dofs)
    # np.save(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/errors_estimator.npy", errors_est)
    # np.save(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/errors_true.npy", errors_true)
    # np.save(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/times.npy", times)





   
    # plt.figure(figsize=(8, 6))
    # plt.grid(True)
    # plt.loglog(dofs[1:], errors_est[1:], '-ok')
    # scaling = errors_est[1] / dofs[1]**-0.5
    # plt.loglog(dofs, scaling * dofs**-0.5, '--', alpha=0.5, color="lightcoral", label="x^{-0.5}")
    # scaling = errors_est[1] / dofs[1]**-0.1
    # plt.loglog(dofs, scaling * dofs**-0.1, '--', alpha = 0.5, color='lawngreen', label = "x^{-0.1}")
    # scaling = errors_est[1] / dofs[1]**-1
    # plt.loglog(dofs, scaling * dofs**-1, '--', alpha = 0.5, color = 'aqua', label = "x^{-1}")
    # scaling = errors_est[1] / dofs[1]**-2
    # plt.loglog(dofs, scaling * dofs**-2, '--', alpha = 0.5, color = 'indigo', label = "x^{-2}")




    # plt.xlabel("Number of degrees of freedom")
    # plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
    # plt.title(f"Estimated Error Convergence p={p}")
    # plt.legend()
    # plt.savefig(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/adaptive_convergence_est.png")


    # plt.figure(figsize=(8, 6))
    # plt.grid(True)
    # plt.loglog(dofs[1:], errors_true[1:], '-.')

    # plt.xlabel("Number of degrees of freedom")
    # plt.ylabel(r"True error norm $\|u - u_h\|$")
    # plt.title(f"True Error Convergence p={p}")

    # plt.savefig(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/adaptive_convergence_true.png")

    # plt.show()

    # TIME: 
    # p = 1
    # d_est = np.load(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/errors_estimator.npy", allow_pickle=True)
    # d_times = np.load(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim}/{p}/times.npy", allow_pickle=True)[1:] 
    # d_times = np.array(list(accumulate(d_times)))

    # with open(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim - 1}/{p}/dat.csv", "r", newline="") as f:
    #     reader = csv.reader(f)
    #     rows = list(reader)

    # # Transpose rows to get columns
    # columns = list(zip(*rows))
    # est = np.array(columns[1][1:], dtype=float)
    # times = np.array(columns[4][1:], dtype=float)

    # times = np.array(list(accumulate(times)))
    # print(times)
    # colors = ['blue', 'green', 'red', 'purple'] 
    # c = colors[p-1]
    # plt.figure(figsize=(8, 6))
    # plt.grid(True)
    # plt.loglog(times, est, '-o', alpha = 0.6, color=c, markersize = 3, label = "Multigrid")
    # plt.loglog(d_times, d_est, '-ok', alpha = 0.6, markersize = 3, label = "Direct")
    # scaling = est[0] / times[0]**-0.5
    # plt.loglog(times, scaling * times**-0.5, '--', alpha=0.5, color="lightcoral", label="t^{-0.5}")
    # scaling = est[0] / times[0]**-0.1
    # plt.loglog(times, scaling * times**-0.1, '--', alpha = 0.5, color='lawngreen', label = "t^{-0.1}")
    # scaling = est[0] / times[0]**-1
    # plt.loglog(times, scaling * times**-1, '--', alpha = 0.5, color = 'aqua', label = "t^{-1}")
    # scaling = est[0] / times[0]**-2
    # plt.loglog(times, scaling * times**-2, '--', alpha = 0.5, color = 'indigo', label = "t^{-2}")
    # plt.xlabel("Cumulative Runtime")
    # plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
    # plt.title(f"Estimator vs Cumulative Runtime for p={p}")
    # plt.legend()
    # plt.savefig(f"output/div_L/theta={theta}_lam={lambda_alg}_alpha={alpha}_dim={dim - 1}/{p}/direct_vs_mg_runtime.png")
    # plt.show()
