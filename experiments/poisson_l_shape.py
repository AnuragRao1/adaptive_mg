"""
Poisson experiment on the L-shaped domain. The code commented at the  bottom was to generate the plots
"""
from firedrake import *
from netgen.occ import *

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager
import time
import csv

def solve_poisson(p,mesh, params):
    V = FunctionSpace(mesh, "CG", p)
    uh = Function(V, name="Solution")
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(0), "on_boundary")
    f = Constant(1)
    F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx
    
    problem = NonlinearVariationalProblem(F, uh, bc)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.set_transfer_manager(atm)

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


def adapt(mesh, eta, theta):
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


rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2

geo = OCCGeometry(L, dim=2)
ngmsh = geo.GenerateMesh(maxh=0.5)
mesh = Mesh(ngmsh)
mesh2 = Mesh(ngmsh)
amh = AdaptiveMeshHierarchy(mesh)
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
theta = 0.5
dim = 1e3

for p in range(1,2):
    level = 0

    dofs = []
    error_estimators = []
    amh = AdaptiveMeshHierarchy([mesh2])

    csv_file = f"output/poisson_L/theta={theta}_dim={dim}/{p}/dat.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dof", "error_est", "time"])

    while level == 0 or uh.function_space().dim() < dim:
        print(f"level {level}")
        mesh = amh[level]


        start = time.time()
        uh = solve_poisson(p, mesh, patch_relax)
        run_time = time.time() - start

        print(f"Completed in {run_time}")
        VTKFile(f"output/poisson_L/theta={theta}_dim={dim}/{p}/adaptive_loop_{level}.pvd").write(uh)

        (eta, error_est) = estimate_error(mesh, uh)

        print(f"  ||u - u_h|| <= C x {error_est}")
        error_estimators.append(error_est)
        dofs.append(uh.function_space().dim())

        with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dofs[-1], error_est, run_time])

        print(f"DoFs: {dofs[-1]}, TIME: {time.time() - start}")


        if uh.function_space().dim() < dim:
            mesh = adapt(mesh, eta, theta)
            amh.add_mesh(mesh)
        level += 1


# def single_time_convergence(dof, est, theta, p):
#     plt.figure(figsize=(8, 6))
#     plt.grid(True)
#     plt.loglog(dof[2:], est[2:], '-o', alpha = 0.7, markersize = 4)
#     scaling = est[2] / dof[2]**-0.5
#     plt.loglog(dof[2:], scaling * dof[2:]**-0.5, '--', alpha=0.5, color="lightcoral", label="x^{-0.5}")
#     scaling = est[2] / dof[2]**-0.1
#     plt.loglog(dof[2:], scaling * dof[2:]**-0.1, '--', alpha = 0.5, color='lawngreen', label = "x^{-0.1}")
#     scaling = est[2] / dof[2]**-1
#     plt.loglog(dof[2:], scaling * dof[2:]**-1, '--', alpha = 0.5, color = 'aqua', label = "x^{-1}")
#     scaling = est[2] / dof[2]**-2
#     plt.loglog(dof[2:], scaling * dof[2:]**-2, '--', alpha = 0.5, color = 'indigo', label = "x^{-2}")
#     plt.xlabel("Number of degrees of freedom")
#     plt.ylabel(r"Estimated energy norm $\sqrt{\sum_K \eta_K^2}$")
#     plt.title(f"Estimated Error Convergence ({theta}) for p={p}")
#     plt.legend()
#     plt.savefig(f"output/poisson_L/theta={theta}_dim={dim}/{theta}_{p}_convergence.png")
    

# import matplotlib.pyplot as plt
# import numpy as np
# for p in range(1, 5):
#     for theta in [0, 0.5]:
#         with open(f"output/poisson_L/theta={theta}_dim={dim}/{p}/dat.csv", "r", newline="") as f:
#             reader = csv.reader(f)
#             rows = list(reader)
#         columns = list(zip(*rows))
#         dofs = np.array(columns[0][1:], dtype=float)
#         errors_est = np.array(columns[1][1:], dtype=float)
#         times = np.array(columns[2][1:], dtype=float)

#         single_time_convergence(dofs, errors_est, theta, p)

# def add_triangle(ax, x0, y0, slope, length=0.4, height=0.5, label=0, **kwargs):
#     dx = length
#     dy = -slope * dx   
    
#     x1, y1 = x0, y0
#     x2, y2 = x0 * 10**dx, y0       
#     x3, y3 = x0, y0 * 10**(-dy)    
    
#     ax.plot([x1, x2], [y1, y2], **kwargs)  
#     ax.plot([x1, x3], [y1, y3], **kwargs)  
#     ax.plot([x2, x3], [y2, y3], **kwargs) 
    
#     if label is not None:
#         xm = x0 * 0.9  
#         ym = (y1 * y3)**0.5 
#         yym = y0 * 0.5
#         xxm = (x1 * x2)**0.5 
#         ax.text(xm, ym, label, va="center", ha="right", fontsize=10) 
#         ax.text(xxm, yym, 1, va="bottom", ha="center", fontsize=10)


# plt.figure(figsize=(8,6))
# plt.grid(True)
# colors = ['blue', 'green', 'red', 'purple'] 
# for p in range(1,5):
#     for theta in [0, 0.5]:    
#         with open(f"output/poisson_L/theta={theta}_dim={dim}/{p}/dat.csv", "r", newline="") as f:
#             reader = csv.reader(f)
#             rows = list(reader)
#         columns = list(zip(*rows))
#         dofs = np.array(columns[0][1:], dtype=float)
#         errors_est = np.array(columns[1][1:], dtype=float)
#         times = np.array(columns[2][1:], dtype=float)

#         if theta == 0.5:
#             plt.loglog(dofs, errors_est, '-o', markersize = 3, alpha=0.5, color = colors[p-1], label=f"adaptive: {p}")
#         else:
#             plt.loglog(dofs, errors_est, "--v", markersize = 3, alpha = 0.5, color = colors[p-1], label=f"uniform: {p}")

# ax = plt.gca()
# # add_triangle(ax, x0=500000, y0=0.01, slope=0.1, label=0.1, color="k")
# add_triangle(ax, x0=130000, y0=0.007, slope=0.5, label=0.5 ,color="k")
# add_triangle(ax, x0=300000, y0=4.5 * 1e-5, slope=1, label=1, color="k")
# add_triangle(ax, x0=100000, y0=3.5 * 1e-7, slope=2, label=2, color="k")
# plt.xlabel("Number of DoFs")
# plt.ylabel(r"Error estimate of energy norm $\sqrt{\sum_K \eta_K^2}$")
# plt.title("Convergence of Error Estimator")
# plt.legend()
# plt.savefig(f"output/poisson_L/theta={theta}_dim={dim}/joint_adaptive_convergence.png")
# plt.show()


