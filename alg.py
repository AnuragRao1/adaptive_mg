from netgen.occ import *
from firedrake import *
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager
import random
import gc
from firedrake.mg.ufl_utils import coarsen
from firedrake.dmhooks import get_appctx
from firedrake import dmhooks
from firedrake.solving_utils import _SNESContext
from firedrake.mg.utils import get_level






def p_robust_alg(p=1, levels=2, s=0):
    # Construct Mesh Hierarchy
    random.seed(1234)
    wp = WorkPlane()
    wp.Rectangle(2,2)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    maxh = 0.5
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = Mesh(ngmesh)
    amh = AdaptiveMeshHierarchy([mesh])
    atm = AdaptiveTransferManager()
    
    for i in range(levels):
        for l, el in enumerate(ngmesh.Elements2D()):
            el.refine = 0
            if random.random() < 0.5:
                el.refine = 1
        ngmesh.Refine(adaptive=True)
        mesh = Mesh(ngmesh)
        amh.add_mesh(mesh)

    # solve initial guess on coarse mesh
    V_0 = FunctionSpace(amh[0], "CG", p)
    (x,y) = SpatialCoordinate(amh[0])
    u_ex = Function(V_0, name="u_0_real").interpolate(sin(2 * pi * x) * sin(2 * pi * y))
    f_0 = Function(V_0, name="f_0").interpolate(- div(grad(u_ex)))
    u_0 = Function(V_0, name="u_0")
    v = TestFunction(V_0)

    bc = DirichletBC(V_0, u_ex, "on_boundary")
    F =  inner(grad(u_0),grad(v)) * dx - inner(f_0, v) * dx
    # solve(F == 0, u_0, bc)

    

    # initialize quantities for mg loop
    V_J = V_0.reconstruct(mesh=amh[-1])



    f_j = [Function(V_0.reconstruct(mesh=amh[j]), name=f"f_{j}") for j in range(0, levels+1)]
    for f in f_j:
        atm.prolong(f_0, f)
    
    
    (x,y) = SpatialCoordinate(amh[-1])
    u_ex = Function(V_J, name="u_0_real").interpolate(sin(2 * pi * x) * sin(2 * pi * y))
    u = Function(V_J)
    v = TestFunction(V_J)
    bc = DirichletBC(V_J, Constant(0), "on_boundary")
    F = inner(grad(u - u_ex), grad(v)) * dx
    attempt_solve(F, u, bc, atm)
    return




    rho_0_0 = Function(V_0, name=f"rho_0^{i}")
    rho_0_bc = DirichletBC(V_0, 0, "on_boundary")
    v_rho = TestFunction(V_0)

    rho_J_alg = Function(V_J, name="rho_J,alg^i")
    rho_J_alg.assign(1) # dummy initialization for first loop

    rho_j = [Function(V_J.reconstruct(mesh=amh[j]), name=f"rho_{j}^i") for j in range(len(amh.meshes))]

    u_J = Function(V_J, name="u_J^i")
    atm.prolong(u_0, u_J, amh)

    patches = {i: generate_submeshes(amh[i]) for i in range(levels+1)}
    #CHECK SUBMESHES HAPPENING PROPERLY
    # for key, val in patches.items():
    #     for i,mesh in enumerate(val):
    #         VTKFile(f"output/neighboring/{key}/{i}.pvd").write(Function(V_0.reconstruct(mesh=mesh)))

    rho_j_a = {i: [Function(V_0.reconstruct(mesh=patch)) for patch in patches[i]] for i in range(levels+1)}

    
    iteration = -1 # iteration tracker
    w1 = (levels + 1) * 3 # J(d+1)
    w2 = 1

    # metrics
    u_real = Function(V_J, name="u_real")
    atm.prolong(u_ex, u_real, amh)
    contraction_errors = []
    rel_errors = []

    while norm(rho_J_alg) > 1e-10 :
        gc.collect()
        iteration += 1
        print("Iteration: ", iteration)
        prev_err = norm(grad(u_real - u_J)) # compute for contraction error

        atm.inject(u_J, u_0, amh)

        #compute rho_0^i
        F = inner(grad(rho_0_0), grad(v_rho)) * dx - (inner(f_0, v_rho) * dx - inner(grad(u_0), grad(v_rho)) * dx)
        # solve(F == 0, rho_0_0, rho_0_bc)
        attempt_solve(F, rho_0_0, rho_0_bc)
        atm.prolong(rho_0_0, rho_j[0], amh)

        # compute local contributions from patch problems (rho_j,a^i)
        # submesh?, find node, locations elements surrounding, construct submesh for all functions,
        #  need to check for parallelization
        
        for j in range(1, levels+1):
            submeshes = patches[j-s]

            # prolong current rho predictions into 
            gc.collect()
            prev_rho_patches = [Function(V_0.reconstruct(mesh=amh[j-s])) for _ in range(j-s)] # for prolonged rho_j^i to compute rho_j,a^i
            for i in range(len(prev_rho_patches)):
                atm.prolong(rho_j[i], prev_rho_patches[i], amh)

            print(f"PATCHES FOR LEVEL {j}: ", len(submeshes))
            for i, patch in enumerate(submeshes):
                print(f"PATCH {i}")
                V_a = V_0.reconstruct(mesh=patch)
                v_patch = TestFunction(V_a)

                f_patch = Function(V_a).interpolate(f_j[j-s])

                u_patch = Function(V_a).interpolate(u_J)

                rho_patch = rho_j_a[j-s][i]
                assert rho_patch.function_space().mesh() == patch

                prev_rho_patches = [Function(V_a).interpolate(prev_rho) for prev_rho in prev_rho_patches]
                patch_bc = DirichletBC(V_a, 0, "on_boundary")

                F = inner(grad(rho_patch), grad(v_patch)) * dx - inner(f_patch, v_patch) * dx + inner(grad(u_patch), grad(v_patch)) * dx + 1 / w2 * inner(grad(sum(prev_rho_patches)), grad(v_patch)) * dx
                # solve(F == 0, rho_patch, patch_bc)
                attempt_solve(F, rho_patch, patch_bc)

            # sum patchwise for level residual estimate (rho_j^i)
            rho_j_a_mesh = [Function(V_0.reconstruct(mesh=amh[j-s])).interpolate(rho_j_a[j-s][i], allow_missing_dofs=True, default_missing_val=0) for i in range(amh[j-s].coordinates.dat.data.shape[0])]
            rho_j[j].assign(1 / w1 * sum(rho_j_a_mesh))
                



        # recombine rho_J,alg^i
        rho_j_prolonged = []
        for i, func in enumerate(rho_j):
            prolonged_rho_j = Function(V_J)
            atm.prolong(func, prolonged_rho_j, amh)
            rho_j_prolonged.append(prolonged_rho_j)
        rho_J_alg.assign(sum(rho_j_prolonged))


        #update solution
        lmbda = (assemble(f_j[-1] * rho_J_alg * dx) - assemble(inner(grad(rho_J_alg), inner(u_J)) * dx)) / norm(grad(rho_J_alg))**2
        u_J.assign(u_J + lmbda * rho_J_alg)

        contraction_errors.append(norm(grad(u_real - u_J)) / prev_err)
        rel_errors.append(norm(grad(u_real - u_J)) / norm(grad(u_real)))
        print("ERRORS: ", contraction_errors[-1], rel_errors[-1])

        #TRACK METRICS HERE
    return u_J, iteration, contraction_errors, rel_errors

def find_neighboring_elements(mesh, vertex_index):
    # given vertex, find all the neighboring elements
    cell_vertices = mesh.coordinates.function_space().cell_node_map().values
    
    neighbor_cells = np.where(np.any(cell_vertices == vertex_index, axis=1))[0]
    return neighbor_cells.tolist()

def generate_submeshes(mesh):
    # find all patches, return submeshes to solve over
    V = FunctionSpace(mesh, "DG", 0)
    elements = []
    for i, _ in enumerate(mesh.coordinates.dat.data):
        elements.append(Function(V))
        elements[i].assign(0)
        neighboring_elements = find_neighboring_elements(mesh, i)

        for el in neighboring_elements:
            elements[i].dat.data[el] = 1

        mesh.mark_entities(elements[i], i)

    mesh = RelabeledMesh(mesh, elements, list(range((mesh.coordinates.dat.data.shape[0]))))
    submeshes = [Submesh(mesh, mesh.topology_dm.getDimension(), i) for i in range((mesh.coordinates.dat.data.shape[0]))]
    return submeshes

def attempt_solve(F, u, bcs, atm):
    solver_configs = [
        # {
        #     "snes_type": "newtonls",
        #     "snes_max_it": 50,
        #     "snes_rtol": 1e-8
        # },
        # {
        #     "snes_type": "newtonls",
        #     "snes_linesearch_type": "bt",
        #     "snes_max_it": 100,
        #     "snes_rtol": 1e-8
        # },
        # {
        #     "snes_type": "newtontr",
        #     "snes_max_it": 200,
        #     "snes_rtol": 1e-8
        # },
        {
            "snes_type": "ksponly",
            "ksp_max_it": 20,
            "ksp_type": "cg", 
            "ksp_monitor": None,
            "snes_monitor": None,
            "ksp_norm_type": "unpreconditioned",
            "ksp_rtol": 1e-6,
            "ksp_atol": 1e-6,
            "pc_type": "mg",
            "mg_levels_pc_type": "jacobi",
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_ksp_max_it": 2,
            "mg_levels_ksp_richardson_scale": 1/3,
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "lu",
            "mg_coarse_pc_factor_mat_solver_type": "mumps" 
        }
    ]
    
    for i, params in enumerate(solver_configs):

        problem = NonlinearVariationalProblem(F, u, bcs)
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
        print("SOLVED")
        # solve(F == 0, u, bcs, solver_parameters=params)
        return True
    
    return False




if __name__ == "__main__":
    _ = p_robust_alg()
    # print(contraction_err)
    # print(rel_err)
