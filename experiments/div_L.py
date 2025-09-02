import numpy as np
dot_prod = np.dot
from firedrake import *
from netgen.occ import *
from firedrake.mg.utils import get_level
from itertools import accumulate


import time
import csv
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager


    
def run_div(p=1, theta=0.5, lam_alg=0.01, alpha = 2/3, dim=1e3, solver = "direct", experiment="bump"):
    def solve_div(mesh, p, u_prev, u_real, params, experiment):
        V = FunctionSpace(mesh, "BDM", p)
        uh = u_prev
        v = TestFunction(V)
        if experiment == "unknown":
            bc = DirichletBC(V, as_vector([Constant(0), Constant(0)]), "on_boundary")
            f_expr = as_vector([Constant(1), Constant(-0.5)])
        else:
            bc = DirichletBC(V, u_real, "on_boundary")
            f_expr = u_real - grad(div(u_real))


        F = (inner(uh,v) + inner(div(uh), div(v)) - inner(f_expr, v)) * dx

        problem = NonlinearVariationalProblem(F, uh, bc)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)
        solver.set_transfer_manager(atm)
    
        _, level = get_level(mesh)
        with PETSc.Log.Event(f"adaptive_{level}"):
            solver.solve()

        return uh, f_expr


    def estimate_error(mesh, uh, u_real, f, experiment):
        W = FunctionSpace(mesh, "DG", 0)
        eta_sq = Function(W)
        w = TestFunction(W)
        h = CellDiameter(mesh) 
        n = FacetNormal(mesh)
        v = CellVolume(mesh)

        eta_vol = assemble(inner(h**2 * (uh - grad(div(uh)) - f)**2, w) * dx)
        eta_jump = assemble(inner(h('+') * jump(div(uh))**2, w('+')) * dS
            + inner(h('-') * jump(div(uh))**2, w('-')) * dS)
        if experiment != "unknown":
            G = (
                inner(eta_sq / v, w) * dx 
                - inner(h**2 * (uh - grad(div(uh)) - f)**2, w) * dx
                - inner(h('+') * jump(div(uh))**2, w('+')) * dS
                - inner(h('-') * jump(div(uh))**2, w('-')) * dS
                - inner(h * dot(u_real - uh, n)**2, w) * ds
                )
            eta_boundary = assemble(inner(h * dot(grad(u_real - uh), n)**2, w) * ds)
            print(f"Vol: {sqrt(sum(eta_vol.dat.data))}, Jump: {sqrt(sum(eta_jump.dat.data))}, Boundary: {sqrt(sum(eta_boundary.dat.data))}")
        else:
            G = (
                inner(eta_sq / v, w) * dx 
                - inner(h**2 * (uh - grad(div(uh)) - f)**2, w) * dx
                - inner(h('+') * jump(div(uh))**2, w('+')) * dS
                - inner(h('-') * jump(div(uh))**2, w('-')) * dS
                )
            print(f"Vol: {sqrt(sum(eta_vol.dat.data))}, Jump: {sqrt(sum(eta_jump.dat.data))}")

        
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)

        eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2


        with eta.dat.vec_ro as eta_:  # compute estimate for error in energy norm
            error_est = sqrt(eta_.dot(eta_))
        return (eta, error_est)

    def generate_u_real(mesh, p, alpha, experiment):
        x, y = SpatialCoordinate(mesh)

        if experiment == "donut":
            r = sqrt(x**2 + y**2)
            chi = conditional(lt(r, 0.1), exp(- (0.1**2) / (0.1**2 - r**2)), 0)
            return as_vector([chi * r**alpha * x, chi * r**alpha * y])
        elif experiment == "bump":
            g = exp(- ((x - 0.5)**2 + (y - 0.5)**2) / 0.2**2)
            return as_vector([g, 1/2 * g])
        else:
            return as_vector([Constant(0), Constant(0)])


    if experiment == "unknown":
        rect1 = WorkPlane(Axes((-1,-1,0), n=Z, h=X)).Rectangle(1,2).Face()
        rect2 = WorkPlane(Axes((-1,0,0), n=Z, h=X)).Rectangle(2,1).Face()
        L = rect1 + rect2

        geo = OCCGeometry(L, dim=2)
        ngmsh = geo.GenerateMesh(maxh=0.1)
        mesh = Mesh(ngmsh)

    else:
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


    amh = AdaptiveMeshHierarchy(mesh)
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
            "ksp_type": "fgmres",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "richardson",
                "ksp_richardson_scale": 1/4,
                # "ksp_monitor_true_residual": None,
                # "mg_levels_ksp_monitor_true_residual": None,
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
    json_str = json.dumps(param_set, indent=4)
    os.makedirs(f"output/div_L_{solver}/{experiment}_theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}/{p}", exist_ok=True)
    with open(f"output/div_L_{solver}/{experiment}_theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}/{p}/param.json", "w") as f:
        f.write(json_str)


    error_estimators = {}
    true_errors = []
    dofs = []

    k_l = []
    times = []
    start_time = time.time()
    times.append(start_time)
    level = 0
    csv_file = f"output/div_L_{solver}/{experiment}_theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}/{p}/dat.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dof", "error_est", "error_true", "k", "time"])

    while level == 0 or u_k.function_space().dim() <= dim:
        print(f"level {level}")
        V = FunctionSpace(mesh, "BDM", p)
        uh = Function(V, name="solution")
        u_prev = Function(V, name="u_prev")
        dofs.append(uh.function_space().dim())
        
        if level > 0:
            tm.prolong(u_k, uh)

        k = 0
        error_est = 0
        u_real = generate_u_real(mesh, p, alpha, experiment)

        while norm(uh - u_prev) > lam_alg * error_est or k == 0:
            k += 1
            u_prev.interpolate(uh)
                    
            start = time.time()
            (uh, f) = solve_div(mesh, p, uh, u_real, param_set, experiment)
            times.append(time.time() - start)
            
            
            ur = Function(V, name="exact").interpolate(u_real)
            eh = Function(V, name="error").interpolate(ur - uh)
            VTKFile(f"output/div_L_{solver}/{experiment}_theta={theta}_lam={lam_alg}_alpha={alpha}_dim={dim}/{p}/{level}_{k}.pvd").write(uh, ur, eh)

            (eta, error_est) = estimate_error(mesh, uh, u_real, f, experiment) 
            print("ERROR ESTIMATE: ", error_est)

            if level not in error_estimators:
                error_estimators[level] = [error_est]
            else:
                error_estimators[level].append(error_est)
            
            error_real = norm(uh - u_real)
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dofs[-1], error_est, error_real, k, times[-1]])


        u_k = Function(V).assign(uh)
        k_l.append(k)

        if u_k.function_space().dim() <= dim:
            mesh = amh.adapt(eta, theta)
            
        
        print(f"DOFS {dofs[-1]}: TIME {times[-1]}")
        level += 1

    final_errors = [error_estimators[key][-1] for key in error_estimators]
    return np.array(dofs, dtype=float), np.array(final_errors), np.array(true_errors), np.array(times)

    

if __name__ == "__main__":
    theta = 0.5
    lambda_alg = 0.01
    alpha = 2/3
    dim = 1e5 - 3

    for p in range(1,2):
        (dof, est, true, times) = run_div(p, theta, lambda_alg, alpha, dim)