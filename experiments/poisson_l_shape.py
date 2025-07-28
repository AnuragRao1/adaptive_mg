from firedrake import *
from netgen.occ import *

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager
from firedrake.mg.ufl_utils import coarsen
from firedrake.dmhooks import get_appctx
from firedrake import dmhooks
from firedrake.solving_utils import _SNESContext
import time

def solve_poisson(p,mesh, params):
    V = FunctionSpace(mesh, "CG", p)
    uh = Function(V, name="Solution")
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(0), "on_boundary")
    f = Constant(1)
    F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx
    
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

def estimate_error(mesh, uh):
    W = FunctionSpace(mesh, "DG", 0)
    eta_sq = Function(W)
    w = TestFunction(W)
    f = Constant(1)
    h = CellDiameter(mesh)  # symbols for mesh quantities
    n = FacetNormal(mesh)
    v = CellVolume(mesh)

    G = (  # compute cellwise error estimator
          inner(eta_sq / v, w)*dx
        - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
        - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS
        - inner(h('-')/2 * jump(grad(uh), n)**2, w('-')) * dS
        )

    # Each cell is an independent 1x1 solve, so Jacobi is exact
    sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
    solve(G == 0, eta_sq, solver_parameters=sp)
    eta = Function(W, name="eta").interpolate(sqrt(eta_sq))  # compute eta from eta^2

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

    theta = 0.5
    should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
    markers.interpolate(should_refine)

    refined_mesh = mesh.refine_marked_elements(markers)
    return refined_mesh


rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2

geo = OCCGeometry(L, dim=2)
ngmsh = geo.GenerateMesh(maxh=0.5)
mesh = Mesh(ngmsh)
mesh2 = Mesh(ngmsh)
amh = AdaptiveMeshHierarchy([mesh])
atm = AdaptiveTransferManager()


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

max_iterations = 20
# for p in range(1,5):
#     dofs = []
#     error_estimators = []
#     amh = AdaptiveMeshHierarchy([mesh2])

#     for i in range(max_iterations):
#         start = time.time()
#         print(f"level {i}")
#         mesh = amh[i]

#         uh = solve_poisson(p, mesh, patch_relax)
#         VTKFile(f"output/poisson_l/{max_iterations}_{p}/adaptive_loop_{i}.pvd").write(uh)

#         (eta, error_est) = estimate_error(mesh, uh)
#         VTKFile(f"output/poisson_l/{max_iterations}_{p}/eta_{i}.pvd").write(eta)

#         print(f"  ||u - u_h|| <= C x {error_est}")
#         error_estimators.append(error_est)
#         dofs.append(uh.function_space().dim())

#         mesh = adapt(mesh, eta)
#         if i != max_iterations - 1:
#             amh.add_mesh(mesh)
#         print(f"DoFs: {dofs[-1]}, TIME: {time.time() - start}")
    
#     np.save(f"output/poisson_l/{max_iterations}_{p}/dofs.npy", np.array(dofs))
#     np.save(f"output/poisson_l/{max_iterations}_{p}/error_est.npy", np.array(error_estimators))



import matplotlib.pyplot as plt
import numpy as np
for p in range(1, 5):
    dofs = np.load(f"output/poisson_l/{max_iterations}_{p}/dofs.npy", allow_pickle=True)
    error_estimators = np.load(f"output/poisson_l/{max_iterations}_{p}/error_est.npy", allow_pickle=True)

    plt.grid()
    plt.loglog(dofs, error_estimators, '-ok', label="Measured convergence")
    scaling = 1.5 * error_estimators[0]/dofs[0]**-(1/(2**p))
    plt.loglog(dofs, np.array(dofs)**(-0.5) * scaling, '--', label="Optimal convergence")
    plt.xlabel("Number of degrees of freedom")
    plt.ylabel(r"Error estimate of energy norm $\sqrt{\sum_K \eta_K^2}$")
    plt.title(f"Convergence for p={p}")
    plt.legend()
    plt.savefig(f"output/poisson_l/{max_iterations}_{p}/adaptive_convergence.png")
    plt.close()

plt.grid()
colors = ['blue', 'green', 'red', 'purple']  
for p in range(1,5):
    dofs = np.load(f"output/poisson_l/{max_iterations}_{p}/dofs.npy", allow_pickle=True)
    error_estimators = np.load(f"output/poisson_l/{max_iterations}_{p}/error_est.npy", allow_pickle=True)
    plt.loglog(dofs, error_estimators, '-o', markersize = 3, color = colors[p-1], label=f"p={p}")

plt.xlabel("Number of DoFs")
plt.ylabel(r"Error estimate of energy norm $\sqrt{\sum_K \eta_K^2}$")
plt.title("Convergence of Error Estimator")
plt.legend()
plt.savefig(f"output/poisson_l/{max_iterations}_4/joint_adaptive_convergence.png")
plt.show()


