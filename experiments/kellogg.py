import numpy as np
dot_prod = np.dot
from firedrake import *
from netgen.occ import *
from firedrake.mg.ufl_utils import coarsen
from firedrake.dmhooks import get_appctx
from firedrake import dmhooks
from firedrake.solving_utils import _SNESContext

import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager

    
def run_system(p=2, theta=0.5, lam_alg=0.01, dim=1e3):
    def solve_kellogg(mesh, p, u_prev, u_real, params, uniform):
        V = FunctionSpace(mesh, "CG", p)
        uh = u_prev
        v = TestFunction(V)
        bc = DirichletBC(V, u_real, "on_boundary")

        x = SpatialCoordinate(mesh)
        a = conditional(lt(x[0] * x[1], 0), Constant(1.0), Constant(161.4476387975881)) # Leaving in this format resolves divergence of solver
        F = inner(a * grad(uh), grad(v))*dx # f == 0, 
        
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

        solver.solve()
        return uh


    def estimate_error(mesh, uh, u_boundary):
        W = FunctionSpace(mesh, "DG", 0)
        eta_sq = Function(W)
        w = TestFunction(W)
        h = CellDiameter(mesh) 
        n = FacetNormal(mesh)
        t = as_vector([-n[1], n[0]])
        v = CellVolume(mesh)

        x = SpatialCoordinate(mesh)
        a = conditional(lt(x[0] * x[1], 0), Constant(1.0), Constant(161.4476387975881))
        a = Function(W).interpolate(a)

        G = (
            inner(eta_sq / v, w) * dx 
            - inner(v * div(a * grad(uh))**2, w) * dx 
            - inner(v('+')**0.5 * jump(a * grad(uh), n)**2, w('+')) * dS
            - inner(v('-')**0.5 * jump(a * grad(uh), n)**2, w('-')) * dS
            - inner(v**0.5 * dot(grad(u_boundary - uh), t)**2, w) * ds
            )
        
        eta_vol = assemble(inner(v * div(a * grad(uh))**2, w) * dx)
        eta_jump = assemble(inner(v('+')**0.5 * jump(a * grad(uh), n)**2, w('+')) * dS
            + inner(v('-')**0.5 * jump(a * grad(uh), n)**2, w('-')) * dS)
        eta_boundary = assemble(inner(v**0.5 * dot(grad(u_boundary - uh), t)**2, w) * ds)
        print(f"Vol: {sqrt(sum(eta_vol.dat.data))}, Jump: {sqrt(sum(eta_jump.dat.data))}, Boundary: {sqrt(sum(eta_boundary.dat.data))}")
        
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)

        eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2
        # VTKFile(f"output/kellogg/theta={theta}_lam={lam_alg}_dim={dim}/{p}/eta_{level}.pvd").write(eta)


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
    
    def generate_u_real(mesh, p):
        u_real = Function(FunctionSpace(mesh, "CG", p), name="u_real")
        x, y = SpatialCoordinate(mesh)

        r = sqrt(x**2 + y**2)
        phi = atan2(y, x)
        phi = conditional(lt(phi, 0), phi + 2 * pi, phi) # map to [0 , 2pi]

        alpha = Constant(0.1)
        beta = Constant(-14.92256510455152)
        delta = Constant(pi / 4)

        mu = conditional(
            lt(phi, pi/2),
            cos((pi/2 - beta) * alpha) * cos((phi - pi/2 + delta) * alpha),
            conditional(
                lt(phi, pi),
                cos(delta * alpha) * cos((phi - pi + beta) * alpha),
                conditional(
                    lt(phi, 3*pi/2),
                    cos(beta * alpha) * cos((phi - pi - delta) * alpha),
                    cos((pi/2 - delta) * alpha) * cos((phi - 3*pi/2 - beta) * alpha)
                )
            )
        )

        u_expr = r**alpha * mu
        u_real.interpolate(u_expr)
        return u_real


    # BUILD DOMAIN
    wp = WorkPlane()
    square = wp.Rectangle(2.0,2.0).Face()
    square = square.Move(Vec(-1.0, -1.0, 0))
    geo = OCCGeometry(square, dim=2)
    ngmesh = geo.GenerateMesh(maxh=2) 
    
    # from netgen.meshing import Mesh as NetgenMesh
    # ngmesh = NetgenMesh()
    # ngmesh.Load("square.msh")
    # if ngmesh.Coordinates() is None:
    #     print("COORDINATES NONE")

    mesh = Mesh(ngmesh)
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
    while level == 0 or u_k.function_space().dim() <= dim:
        start = time.time()
        if uniform:
            mesh = mh[level]

        print(f"level {level}")
        V = FunctionSpace(mesh, "CG", p)
        uh = Function(V, name="solution")
        u_prev = Function(V, name="u_prev")
        
        if level > 0:
            # start = time.time()
            tm.prolong(u_k, uh)
            # print("TIME TO PROLONG: ", time.time() - start)      

        k = 0
        error_est = 0

        # start = time.time()
        u_real = generate_u_real(mesh, p)        
        # print("TIME TO GENERATE REAL U: ", time.time() - start)

        while norm(uh - u_prev) > lam_alg * error_est or k == 0:

            k += 1
            u_prev.interpolate(uh)
            # start = time.time()
            uh = solve_kellogg(mesh, p, uh, u_real, patch_relax, uniform)
            # print("TIME TO SOLVE KELLOGG: ", time.time() - start)

            # start = time.time()
            (eta, error_est) = estimate_error(mesh, uh, u_real) 
            # print("TIME TO ESTIMATE ERROR: ", time.time() - start)
            print("ERROR ESTIMATE: ", error_est)

            if level not in error_estimators:
                error_estimators[level] = [error_est]
            else:
                error_estimators[level].append(error_est)


        u_k = Function(V).interpolate(uh)
        VTKFile(f"output/kellogg/theta={theta}_lam={lam_alg}_dim={dim}/{p}/{level}_{k}.pvd").write(uh)
        k_l.append(k)
        dofs.append(uh.function_space().dim())

        err_real = norm(u_k - u_real)
        # print("TRUE ERROR: ", err_real)
        true_errors.append(err_real)
        # start = time.time()
        if not uniform:
            mesh = adapt(mesh, eta)
            if u_k.function_space().dim() <= dim:
                amh.add_mesh(mesh)
        # print("TIME TO REFINE/ADD MESH: ", time.time() - start)
        
        times.append(time.time() - start)
        print(f"DOFS {dofs[-1]}: TIME {times[-1]}")
        level += 1

        
    # print("TIMES FOR LEVELS: ", times)

    final_errors = [error_estimators[key][-1] for key in error_estimators]
    return np.array(dofs, dtype=float), np.array(final_errors), np.array(true_errors), np.array(times)

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    theta = 0.5
    lambda_alg = 0.01
    dim = 1e3


    errors_true = {}
    errors_est = {}
    dofs = {}
    times = {}
    # results = [run_system(i, theta, lambda_alg, levels) for i in range(1, 5)]
    # all_results = comm.gather(results, root=0)
    # print(all_results)

    for p in range(2,3):
        (dof, est, true, times) = run_system(p, theta, lambda_alg, dim)
        dofs[p] = dof
        errors_est[p] = est
        errors_true[p] = true

        np.save(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/dofs.npy", dof)
        np.save(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/errors_estimator.npy", est)
        np.save(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/errors_true.npy", true)
        np.save(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/times.npy", times)

        plt.figure(figsize=(8, 6))
        plt.grid(True)
        plt.loglog(dof[1:], est[1:], '-ok', alpha = 0.7, markersize = 4)
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
        plt.title(f"Estimated Error Convergence p={p}")
        plt.legend()
        plt.savefig(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/single_convergence.png")


    # for p in range(1,5):
    #     dofs[p] = np.load(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/dofs.npy", allow_pickle=True)
    #     errors_est[p] = np.load(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/errors_estimator.npy", allow_pickle=True)
    #     errors_true[p] = np.load(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/errors_true.npy", allow_pickle=True)
    #     times[p] = np.load(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/times.npy", allow_pickle=True)


    # for i, dat in enumerate(zip(dofs[2].item()[2], errors_est[2])):
    #     print(i, dat, times[2][i])

    # colors = ['blue', 'green', 'red', 'purple']  
    # plt.figure(figsize=(8, 6))
    # plt.grid(True)
    # for p in range(4):
    #     plt.loglog(dofs[p+1].item()[p+1], errors_est[p+1], '-o', color=colors[p], alpha = 0.7, markersize=4, label=f"Measured convergence $p={p+1}$")

    # plt.xlabel("Number of degrees of freedom")
    # plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
    # plt.legend()
    # plt.title("Estimated Error Convergence")
    # plt.savefig(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/adaptive_convergence_est.png")


    # plt.figure(figsize=(8, 6))
    # plt.grid(True)
    # for p in range(4):
    #     plt.loglog(dofs[p+1].item()[p+1], errors_true[p+1], '-.', color=colors[p], alpha = 0.7, label=f"True Error Norm $p={p+1}$")

    # plt.xlabel("Number of degrees of freedom")
    # plt.ylabel(r"True error norm $\|u - u_h\|$")
    # plt.legend()
    # plt.title("True Error Convergence")
    # plt.savefig(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/adaptive_convergence_true.png")

    # plt.show()


    # #### SINGLE SAMPLE
    # p = 1
    # (dofs, errors_est, errors_true, times) = run_system(p, theta, lambda_alg, levels)
    # np.save(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/dofs.npy", dofs)
    # np.save(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/errors_estimator.npy", errors_est)
    # np.save(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/errors_true.npy", errors_true)
    # np.save(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/times.npy", times)





   
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
    # plt.savefig(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/adaptive_convergence_est.png")


    # plt.figure(figsize=(8, 6))
    # plt.grid(True)
    # plt.loglog(dofs[1:], errors_true[1:], '-.')

    # plt.xlabel("Number of degrees of freedom")
    # plt.ylabel(r"True error norm $\|u - u_h\|$")
    # plt.title(f"True Error Convergence p={p}")

    # plt.savefig(f"output/kellogg/theta={theta}_lam={lambda_alg}_dim={dim}/{p}/adaptive_convergence_true.png")

    # plt.show()
