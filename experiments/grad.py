import numpy as np
dot_prod = np.dot
from firedrake import *
from netgen.occ import *
from firedrake.mg.utils import get_level


import time
import csv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager

from firedrake.petsc import PETSc

    
def run_grad(p=2, theta=0.5, lam_alg=0.01, dim=1e3, solver="direct"):
    def solve_grad(mesh, p, u_prev, u_real, params, uniform):
        V = VectorFunctionSpace(mesh, "CG", p)
        uh = u_prev
        v = TestFunction(V)
        bc = DirichletBC(V, u_real, "on_boundary")

        x = SpatialCoordinate(mesh)
        a = conditional(lt(x[0] * x[1], 0), Constant(1.0), Constant(161.4476387975881)) # Leaving in this format resolves divergence of solver
        F = inner(a * grad(uh), grad(v))*dx # f == 0, 
        
        
        problem = NonlinearVariationalProblem(F, uh, bc)

        if not uniform:
            solver = NonlinearVariationalSolver(problem, solver_parameters=params)
            solver.set_transfer_manager(atm)

        _, level = get_level(mesh)
        with PETSc.Log.Event(f"adaptive_{level}"):
            solver.solve()

        return uh


    def estimate_error(mesh, uh, u_boundary):
        W = FunctionSpace(mesh, "DG", 0)
        eta_sq = Function(W)
        w = TestFunction(W)
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
    
    def generate_u_real(mesh):
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

        u_expr = as_vector([r**alpha * mu, r**alpha * mu])
        return u_expr


    from netgen.geom2d import unit_square

    from netgen.meshing import Mesh as NetgenMesh
    from netgen.meshing import MeshPoint, Element2D, FaceDescriptor, Element1D 
    from netgen.csg import Pnt 
    
    ngmesh = NetgenMesh(dim=2) 
    
    fd = ngmesh.Add(FaceDescriptor(bc=1,domin=1,surfnr=1)) 
    
    pnums = [] 
    pnums.append(ngmesh.Add(MeshPoint(Pnt(-1, -1, 0)))) 
    pnums.append(ngmesh.Add(MeshPoint(Pnt(-1, 1, 0))))  
    pnums.append(ngmesh.Add(MeshPoint(Pnt( 1, 1, 0))))  
    pnums.append(ngmesh.Add(MeshPoint(Pnt( 1, -1, 0)))) 
    pnums.append(ngmesh.Add(MeshPoint(Pnt( 0, 0, 0))))  
    
    ngmesh.Add(Element2D(fd, [pnums[0], pnums[1], pnums[4]]))  
    ngmesh.Add(Element2D(fd, [pnums[1], pnums[2], pnums[4]]))  
    ngmesh.Add(Element2D(fd, [pnums[2], pnums[3], pnums[4]]))  
    ngmesh.Add(Element2D(fd, [pnums[3], pnums[0], pnums[4]])) 
    
    ngmesh.Add(Element1D([pnums[0], pnums[1]], index=1)) 
    ngmesh.Add(Element1D([pnums[1], pnums[2]], index=1)) 
    ngmesh.Add(Element1D([pnums[2], pnums[3]], index=1)) 
    ngmesh.Add(Element1D([pnums[0], pnums[3]], index=1))




    for i in range(2):
        for l, el in enumerate(ngmesh.Elements2D()):
            el.refine = 1
        ngmesh.Refine(adaptive=True)
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
    "pc_star_backend": "tinyasm"})

    jacobi_relax = mg_params({"pc_type": "jacobi"}, mat_type="matfree")
    chol = {"mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps"}
    if solver == "direct":
        param_set = chol
    if solver == "mg":
        param_set = patch_relax

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
    csv_file = f"output/grad_{solver}/theta={theta}_lam={lam_alg}_dim={dim}/{p}/dat.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dof", "error_est", "error_true", "k", "time"])

    while level == 0 or u_k.function_space().dim() <= dim:
        if uniform:
            mesh = mh[level]

        print(f"level {level}")
        V = VectorFunctionSpace(mesh, "CG", p)
        uh = Function(V, name="solution")
        u_prev = Function(V, name="u_prev")
        dofs.append(uh.function_space().dim())
        
        if level > 0:
            tm.prolong(u_k, uh)

        k = 0
        error_est = 0

        u_real = generate_u_real(mesh)        

        while norm(uh - u_prev) > lam_alg * error_est or k == 0:
            k += 1
            u_prev.interpolate(uh)
                        
            start = time.time()
            uh = solve_grad(mesh, p, uh, u_real, param_set, uniform)
            times.append(time.time() - start)

            (eta, error_est) = estimate_error(mesh, uh, u_real) 
            print("ERROR ESTIMATE: ", error_est)

            if level not in error_estimators:
                error_estimators[level] = [error_est]
            else:
                error_estimators[level].append(error_est)
            
            err_real = norm(uh - u_real)
            true_errors.append(err_real)
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dofs[-1], error_est, err_real, k, times[-1]])




        u_k = Function(V).interpolate(uh)
        if level % 10 == 0 or level < 15:
            ur = Function(V, name="exact").interpolate(u_real)
            eh = Function(V, name="error").interpolate(ur - uh)
            VTKFile(f"output/grad_{solver}/theta={theta}_lam={lam_alg}_dim={dim}/{p}/{level}_{k}.pvd").write(uh, ur, eh)
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
    theta = 0.5
    lambda_alg = 0.01
    dim = 1e4
    solver = "direct"
   
    for p in range(1,2):
        (dof, est, true, times) = run_grad(p, theta, lambda_alg, dim, solver)