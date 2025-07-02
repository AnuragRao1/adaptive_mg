from firedrake import *
from netgen.occ import *
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager
from firedrake.mg.ufl_utils import coarsen
from firedrake.dmhooks import get_appctx
from firedrake import dmhooks
from firedrake.solving_utils import _SNESContext

def solve_poisson(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    uh = Function(V, name="Solution")
    v = TestFunction(V)
    bc = DirichletBC(V, 0, "on_boundary")
    f = Constant(1)
    F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx
    solve(F == 0, uh, bc)
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

    theta = 0.5
    should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
    markers.interpolate(should_refine)

    refined_mesh = mesh.refine_marked_elements(markers)
    return refined_mesh


rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
L = rect1 + rect2

geo = OCCGeometry(L, dim=2)
ngmsh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmsh)
amh = AdaptiveMeshHierarchy([mesh])
atm = AdaptiveTransferManager()


max_iterations = 10
error_estimators = []
dofs = []
for i in range(max_iterations):
    print(f"level {i}")

    uh = solve_poisson(mesh)
    VTKFile(f"output/poisson_l/adaptive_loop_{i}.pvd").write(uh)

    (eta, error_est) = estimate_error(mesh, uh)
    print(f"  ||u - u_h|| <= C x {error_est}")
    error_estimators.append(error_est)
    dofs.append(uh.function_space().dim())

    mesh = adapt(mesh, eta)
    if i != max_iterations - 1:
        amh.add_mesh(mesh)

V_J = FunctionSpace(amh[-1], "CG", 1)
(x,y) = SpatialCoordinate(amh[-1])
f = Constant(1)
u = Function(V_J)
v = TestFunction(V_J)
bc = DirichletBC(V_J, Constant(0), "on_boundary")
F = inner(grad(u), grad(v)) * dx - inner(f, v) * dx

params = {
        "snes_type": "ksponly",
        "ksp_max_it": 20,
        "ksp_type": "cg", 
        "ksp_norm_type": "unpreconditioned",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-8,
        "ksp_view": None,
        "pc_type": "mg",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": 2,
        "mg_levels_ksp_richardson_scale": 1/3,
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps" 
    }

problem = NonlinearVariationalProblem(F, u, bc)
dm = u.function_space().dm
old_appctx = get_appctx(dm)
mat_type = "aij"
appctx = _SNESContext(problem, mat_type, mat_type, old_appctx)
appctx.transfer_manager = atm
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.set_transfer_manager(atm)
with dmhooks.add_hooks(dm, solver, appctx=appctx, save=False):
    coarsen(problem, coarsen)

solver.solve()
VTKFile(f"output/poisson_l/mg_solution_.pvd").write(u)
diff = Function(V_J).assign(u - uh)
VTKFile(f"output/poisson_l/mg_solution_diff.pvd").write(diff)
(eta, error_est) = estimate_error(amh[-1], u)
print("MG Solution:")
print(f"  ||u - u_h|| <= C x {error_est}")


# import matplotlib.pyplot as plt
# import numpy as np

# plt.grid()
# plt.loglog(dofs, error_estimators, '-ok', label="Measured convergence")
# scaling = 1.5 * error_estimators[0]/dofs[0]**-(0.5)
# plt.loglog(dofs, np.array(dofs)**(-0.5) * scaling, '--', label="Optimal convergence")
# plt.xlabel("Number of degrees of freedom")
# plt.ylabel(r"Error estimate of energy norm $\sqrt{\sum_K \eta_K^2}$")
# plt.legend()
# plt.savefig("adaptive_convergence.png")
# #plt.show()