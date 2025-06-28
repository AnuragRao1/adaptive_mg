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
    maxh = 0.1
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
    (x,y) = SpatialCoordinate(amh[0])
    u_ex = sin(2 * pi * x) * sin(2 * pi * y)
    f_0 = - div(grad(u_ex))
    V_0 = FunctionSpace(amh[0], "CG", p)
    u_0 = Function(V_0, name="u_0")
    v = TestFunction(V_0)

    bc = DirichletBC(V_0, u_ex, "BCs")
    F = inner(grad(u_0),grad(v)) * dx - f_0 * v * dx
    solve(F == 0, u_0, bc)

    # initialize quantities for mg loop
    V_J = V_0.reconstruct(mesh=amh[-1])

    f_j = [Function(V_0.reconstruct(mesh=amh[j]), name=f"f_{j}") for j in range(1, levels+1)]
    for f in f_j:
        atm.prolong(f_0, f, amh)
    

    rho_0_0 = Function(V_0, name=f"rho_0^{i}")
    rho_0_bc = DirichletBC(V_0, 0, "boundary")
    v_rho = TestFunction(V_0)

    rho_J_alg = Function(V_0, name="rho_J,alg^i")
    rho_J_alg.assign(1) # dummy initialization for first loop

    rho_j = [Function(V_J.reconstruct(mesh=amh[j]), name=f"rho_{j}^i") for j in range(len(amh.meshes))]

    u_J = Function(V_J, name="u_J^i")
    atm.prolong(u_0, u_J, amh)

    patches = {i: generate_submeshes(amh[i]) for i in range(levels+1)}
    rho_j_a = {i: [Function(V_0.reconstruct(mesh=patch)) for patch in patches[i]] for i in range(levels+1)}

    
    iteration = -1 # iteration tracker
    w1 = 0
    w2 = 0

    while norm(rho_J_alg) > 1e-10 :
        iteration += 1

        atm.inject(u_J, u_0)

        #compute rho_0^i
        F = inner(grad(rho_0_0), grad(v_rho)) * dx - (f_0 * v_rho * dx - inner(grad(u_0), grad(v_rho)) * dx)
        solve(F == 0, rho_0_0, rho_0_bc)
        atm.prolong(rho_0_0, rho_j[0], amh)

        # compute local contributions from patch problems
        # submesh?, find node, locations elements surrounding, construct submesh for all functions,
        #  need to check for parallelization
        
        for j in range(1, levels+1):
            submeshes = patches[j-s]

            # prolong current rho predictions into 
            rho_prev_patches = [Function(V_0.reconstruct(mesh=amh[j-s])) for _ in range(j-s)] # for prolonged rho_j^i to compute rho_j,a^i
            for i in range(len(rho_prev_patches)):
                atm.prolong(rho_j[i], rho_prev_patches[i], amh)

            for i, patch in enumerate(submeshes):
                V_a = V_0.reconstruct(mesh=patch)
                v_patch = TestFunction(V_a)

                f_patch = Function(V_a).interpolate(f_j[j-s])

                u_patch = Function(V_a).interpolate(u_J)

                rho_patch = rho_j_a[j-s][i]

                rho_prev_patches = [Function(V_a).interpolate(rho_prev) for rho_prev in rho_prev_patches]
                patch_bc = DirichletBC(V_a, 0, "patch boundary condition")

                F = inner(grad(rho_patch), grad(v_patch)) * dx - f_patch * v_patch * dx + inner(grad(u_patch), grad(v_patch)) * dx + 1 / w2 * inner(sum(grad(rho_prev_patches)), grad(v_patch)) * dx
                solve(F == 0, rho_patch, patch_bc)

            rho_j[j].assign(sum(rho_j_a[j-s]))
                # sum patchwise for level residual estimate



        # recombine rho_J,alg
        rho_j_prolonged = []
        for i, func in enumerate(rho_j):
            prolonged_rho_j = Function(V_J)
            atm.prolong(func, prolonged_rho_j, amh)
            rho_j_prolonged.append(prolonged_rho_j)
        rho_J_alg.assign(sum(rho_j_prolonged))


        #update solution
        lmbda = (assemble(f_j[-1] * rho_J_alg * dx) - assemble(inner(grad(rho_J_alg), inner(u_J)) * dx)) / norm(grad(rho_J_alg))**2
        u_J.assign(u_J + lmbda * rho_J_alg)

        #TRACK METRICS HERE

def find_neighboring_elements(mesh, vertex_index):
    # given vertex, find all the neighboring elements
    cell_vertices = mesh.coordinates.function_space().cell_node_map().values
    
    neighbor_cells = np.where(np.any(cell_vertices == vertex_index, axis=1))[0]
    return neighbor_cells.tolist()

def generate_submeshes(mesh):
    # find all patches, return submeshes to solve over
    V = FunctionSpace(mesh, "DG", 0)
    elements = []
    for i, _ in mesh.coordinates.dat.data:
        elements.append(Function(V))
        elements[i].assign(0)
        neighboring_elements = find_neighboring_elements(mesh, i)

        for el in neighboring_elements:
            elements[i].dat.data[el] = 1

        mesh.mark_entities(elements, i)

    marked_mesh = RelabeledMesh(mesh, elements, list(range((mesh.coordinates.dat.data.shape[0]))))
    submeshes = [Submesh(marked_mesh, marked_mesh.topology_dm.getDimension(), i) for i in range((mesh.coordinates.dat.data.shape[0]))]
    return submeshes



if __name__ == "__main__":
    random.seed(1234)
    wp = WorkPlane()
    wp.Rectangle(2,2)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    maxh = 0.1
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = Mesh(ngmesh)
    amh = AdaptiveMeshHierarchy([mesh])
    atm = AdaptiveTransferManager()
    
    for i in range(2):
        for l, el in enumerate(ngmesh.Elements2D()):
            el.refine = 0
            if random.random() < 0.5:
                el.refine = 1
        ngmesh.Refine(adaptive=True)
        mesh = Mesh(ngmesh)
        amh.add_mesh(mesh)

    vertex_coordinates = amh[-2].coordinates.dat.data     # Read-write access

    # Access individual vertices
    for i, vertex in enumerate(vertex_coordinates):
        print(f"Vertex {i}: {vertex}")
        if i > 5:  # Just show first few
            break
