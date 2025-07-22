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

    
def run_system(p=2, theta=0.5, lam_alg=0.01, max_iterations=10):
    def solve_kellogg(mesh, p, u_prev, u_real, params):
        V = FunctionSpace(mesh, "CG", p)
        uh = u_prev
        v = TestFunction(V)
        bc = DirichletBC(V, u_real, "on_boundary")

        W = FunctionSpace(mesh, "DG", 0)
        a = Function(V)
        x = SpatialCoordinate(mesh)
        # a.interpolate(conditional(x[0] * x[1] < 0, Constant(1.0), Constant(161.4476387975881)))
        a = conditional(x[0] * x[1] < 0, Constant(1.0), Constant(161.4476387975881))
        F = inner(a * grad(uh), grad(v))*dx # f == 0, trying to map from constant space to CG2
        
        problem = NonlinearVariationalProblem(F, uh, bc)

        dm = uh.function_space().dm
        old_appctx = get_appctx(dm)
        mat_type = params["mat_type"]
        appctx = _SNESContext(problem, mat_type, mat_type, old_appctx)
        appctx.transfer_manager = atm
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)
        solver.set_transfer_manager(atm)
        with dmhooks.add_hooks(dm, solver, appctx=appctx, save=False):
            coarsen(problem, coarsen)

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

        V = FunctionSpace(mesh, "CG", p)
        a = Function(V)
        x = SpatialCoordinate(mesh)
        a.interpolate(conditional(x[0] * x[1] < 0, 1.0, 161.4476387975881))
        a = conditional(x[0] * x[1] < 0, 1.0, 161.4476387975881)

        G = (
            inner(eta_sq / v, w) * dx 
            - inner(v * div(a * grad(uh))**2, w) * dx 
            - inner(v('+')**0.5 * jump(a * grad(uh), n)**2, w('+')) * dS
            - inner(v('-')**0.5 * jump(a * grad(uh), n)**2, w('-')) * dS
            #- inner(v ** 0.5 * (u_deriv - proj_u_deriv)**2, w) * ds
            - inner(v ** 0.5 * dot(grad(u_boundary - uh), t)**2, w) * ds
            )

        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2

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
        phi = conditional(phi < 0, phi + 2 * pi, phi) # map to [0 , 2pi]

        alpha = Constant(0.1)
        beta = Constant(-14.92256510455152)
        delta = Constant(pi / 4)

        mu = conditional(
            phi < pi/2,
            cos((pi/2 - beta) * alpha) * cos((phi - pi/2 + delta) * alpha),
            conditional(
                phi < pi,
                cos(delta * alpha) * cos((phi - pi + beta) * alpha),
                conditional(
                    phi < 3*pi/2,
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
    ngmsh = geo.GenerateMesh(maxh=1) 
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

    # ITERATIVE LOOP
    max_iterations = max_iterations
    error_estimators = {i: [] for i in range(max_iterations)}
    true_errors = []
    dofs = []
    k_l = []
    times = []
    start_time = time.time()
    for level in range(max_iterations):
        print(f"level {level}")
        V = FunctionSpace(amh[-1], "CG", p)
        uh = Function(V, name="solution")
        u_prev = Function(V, name="u_prev")
        
        if level > 0:
            tm.prolong(u_k, uh)       

        k = 0
        error_est = 0
        u_real = generate_u_real(amh[-1], p)

        while norm(uh - u_prev) > lam_alg * error_est or k == 0:
            k += 1
            u_prev.interpolate(uh)
            uh = solve_kellogg(amh[-1], p, uh, u_real, patch_relax)

            (eta, error_est) = estimate_error(amh[-1], uh, u_real) 
            print("ERROR ESTIMATE: ", error_est)

            error_estimators[level].append(error_est)

        u_k = Function(V).interpolate(uh)
        VTKFile(f"output/kellogg/{p}_{theta}_{lam_alg}_levels={max_iterations}/{level}_{k}.pvd").write(uh)
        k_l.append(k)
        dofs.append(uh.function_space().dim())

        err_real = norm(u_k - u_real)
        print("TRUE ERROR: ", err_real)
        true_errors.append(err_real)
        mesh = adapt(amh[-1], eta)
        if level != max_iterations - 1:
            amh.add_mesh(mesh)
        
        times.append(time.time() - start_time)
    print("TIMES FOR LEVELS: ", times)

    import matplotlib.pyplot as plt
    import numpy as np
    final_errors = [error_estimators[key][-1] for key in error_estimators]

    plt.grid()
    plt.loglog(dofs, final_errors, '-ok', label="Measured convergence")
    scaling = 1.5 * error_estimators[0][-1]/dofs[0]**-(0.5)
    plt.loglog(dofs, np.array(dofs)**(-0.5) * scaling, '--', label="Optimal convergence")
    plt.loglog(dofs, true_errors, '-ok', label="True Error Norm")
    plt.xlabel("Number of degrees of freedom")
    plt.ylabel(r"Error estimate of energy norm $\sqrt{\sum_K \eta_K^2}$ for $u_l^{\hat{k}}$")
    plt.legend()
    plt.savefig(f"output/kellogg/{p}_{theta}_{lam_alg}_levels={max_iterations}/adaptive_convergence.png")
    plt.show()

if __name__ == "__main__":
    run_system(1, 0.5, 0.01, 10)
