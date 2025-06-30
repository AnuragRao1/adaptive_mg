from netgen.occ import *
from firedrake import *
from adaptive import AdaptiveMeshHierarchy
from adaptive_transfer_manager import AdaptiveTransferManager
import random


def p_robust_alg(p=1, levels=2, s=0):
    # Construct Mesh Hierarchy
    random.seed(1234)
    wp = WorkPlane()
    wp.Rectangle(2,2)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    maxh = 0.8
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
    F =  inner(grad(u_0),grad(v)) * dx - f_0 * v * dx
    solve(F == 0, u_0, bc)

    # initialize quantities for mg loop
    V_J = V_0.reconstruct(mesh=amh[-1])

    f_j = [Function(V_0.reconstruct(mesh=amh[j]), name=f"f_{j}") for j in range(1, levels+1)]
    for f in f_j:
        atm.prolong(f_0, f, amh)
    

    rho_0_0 = Function(V_0, name=f"rho_0^{i}")
    #rho_0_bc = DirichletBC(V_0, 0, "on_boundary")
    v_rho = TestFunction(V_0)

    rho_J_alg = Function(V_0, name="rho_J,alg^i")
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
        iteration += 1
        print("Iteration: ", iteration)
        prev_err = norm(grad(u_real - u_J)) # compute for contraction error

        atm.inject(u_J, u_0, amh)

        #compute rho_0^i
        F = inner(grad(rho_0_0), grad(v_rho)) * dx - (f_0 * v_rho * dx - inner(grad(u_0), grad(v_rho)) * dx)
        solve(F == 0, rho_0_0)
        atm.prolong(rho_0_0, rho_j[0], amh)

        # compute local contributions from patch problems (rho_j,a^i)
        # submesh?, find node, locations elements surrounding, construct submesh for all functions,
        #  need to check for parallelization
        
        for j in range(1, levels+1):
            submeshes = patches[j-s]

            # prolong current rho predictions into 
            prev_rho_patches = [Function(V_0.reconstruct(mesh=amh[j-s])) for _ in range(j-s)] # for prolonged rho_j^i to compute rho_j,a^i
            for i in range(len(prev_rho_patches)):
                atm.prolong(rho_j[i], prev_rho_patches[i], amh)

            for i, patch in enumerate(submeshes):
                V_a = V_0.reconstruct(mesh=patch)
                v_patch = TestFunction(V_a)

                f_patch = Function(V_a).interpolate(f_j[j-s])

                u_patch = Function(V_a).interpolate(u_J)

                rho_patch = rho_j_a[j-s][i]
                assert rho_patch.function_space().mesh() == patch

                prev_rho_patches = [Function(V_a).interpolate(prev_rho) for prev_rho in prev_rho_patches]
                #patch_bc = DirichletBC(V_a, 0, "on_boundary")

                print("rho prevs: ", len(prev_rho_patches))
                F = inner(grad(rho_patch), grad(v_patch)) * dx - f_patch * v_patch * dx + inner(grad(u_patch), grad(v_patch)) * dx + 1 / w2 * inner(grad(sum(prev_rho_patches)), grad(v_patch)) * dx
                solve(F == 0, rho_patch)

            # sum patchwise for level residual estimate (rho_j^i)
            rho_j[j].assign(1 / w1 * sum(rho_j_a[j-s]))
                



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

    marked_mesh = RelabeledMesh(mesh, elements, list(range((mesh.coordinates.dat.data.shape[0]))))
    submeshes = [Submesh(marked_mesh, marked_mesh.topology_dm.getDimension(), i) for i in range((mesh.coordinates.dat.data.shape[0]))]
    return submeshes



if __name__ == "__main__":
    (u, iterations, contraction_err, rel_err) = p_robust_alg()
    print(contraction_err)
    print(rel_err)
