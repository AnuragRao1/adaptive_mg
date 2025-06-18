from netgen.occ import *
import pdb 
import numpy as np

import firedrake
from firedrake.utils import cached_property
from firedrake.cython import mgimpl as impl
from firedrake.mg import HierarchyBase
from firedrake import *
from fractions import Fraction
from adaptive_transfer_manager import AdaptiveTransferManager
from firedrake.mg.utils import set_level, get_level


""" Implementing Pablo's method here, we split our refined meshes into the unsplit and split submeshes, we have fixed sized arrays for both parts which resolves the transfer manager issues."""


class AdaptiveMeshHierarchy(HierarchyBase):
    def __init__(self, mesh, refinements_per_level=1, nested=True):
        self.meshes = tuple(mesh)
        self._meshes = tuple(mesh)
        self.submesh_hierarchies = []
        self.coarse_to_fine_cells = None
        self.fine_to_coarse_cells = None
        self.refinements_per_level = refinements_per_level
        self.nested = nested
        set_level(mesh[0], self, Fraction(1, 1))

        # Implementing setlevel might mess with the adjusted AdaptiveTransferManager
        
    def add_mesh(self, mesh, netgen_flags=True):
        #check new mesh is fd mesh or netgen, if not reject
        if (isinstance(netgen_flags, bool) and netgen_flags) and not hasattr(mesh, "netgen_mesh"):
            raise RuntimeError("NO SUPPORT FOR OTHER MESH TYPES OR NEED TO INPUT NG MESH USING Mesh()")
        
        self._meshes += tuple(mesh)
        self.meshes += tuple(mesh)
        coarse_mesh = self.meshes[-2]
        level = len(self.meshes)
        set_level(self.meshes[-1], self, level)
        # self._shared_data_cache ???

        if len(self.meshes) <= 2: 
            self.coarse_to_fine_cells = {}
            self.fine_to_coarse_cells = {}
            self.fine_to_coarse_cells[Fraction(0,1)] = None

        if (isinstance(netgen_flags, bool) and netgen_flags) and hasattr(mesh, "netgen_mesh"):
            # extract parent child relationships from netgen meshes,
            num_parents = self.meshes[-2].num_cells()
            (c2f, f2c) = get_c2f_f2c_fd(mesh, num_parents, coarse_mesh)

       
        self.coarse_to_fine_cells[Fraction(len(self.meshes) - 2,1)] = c2f # for now fix refinements_per_level to be 1
        self.fine_to_coarse_cells[Fraction(len(self.meshes) - 1,1)] = np.array(f2c)

        # split both the fine and coarse meshes into the componenets that will be split, store those 
        print("CONSTRUCTING SUBMESHES")
        (v, z, w, coarse_intersect, v_fine, z_fine, w_fine, fine_intersect, num_children) = split_to_submesh(mesh, coarse_mesh, c2f, f2c)
        coarse_mesh.mark_entities(coarse_intersect, 1)
        coarse_mesh.mark_entities(v, 2)
        coarse_mesh.mark_entities(z, 3)
        coarse_mesh.mark_entities(w, 4)
        coarse_mesh = RelabeledMesh(coarse_mesh, [coarse_intersect, v, z, w], [1, 2, 3, 4], name="Relabeled_coarse")
        #c_subm = [Submesh(coarse_mesh, coarse_mesh.topology_dm.getDimension(), j) for j in [1,2,3,4]]
        c_subm = {j: Submesh(coarse_mesh, coarse_mesh.topology_dm.getDimension(), j) for j in [1,2,3,4] if any(num_children == j)}
       
        mesh.mark_entities(fine_intersect, 11)
        mesh.mark_entities(v_fine, 22)
        mesh.mark_entities(z_fine, 33)
        mesh.mark_entities(w_fine, 44)
        mesh = RelabeledMesh(mesh, [fine_intersect, v_fine, z_fine, w_fine], [11, 22, 33, 44])
        #f_subm = [Submesh(mesh, mesh.topology_dm.getDimension(), j) for j in [11,22,33,44]]
        f_subm = {int(str(j)[0]): Submesh(mesh, mesh.topology_dm.getDimension(), j) for j in [11, 22, 33, 44] if any(num_children == int(str(j)[0]))}
        
        print("FINISHED CONSTRUCTING SUBMESHES")
        # update c2f and f2c for submeshes by mapping numberings on full mesh to numberings on coarse mesh
        n = [len([el for el in c2f if len(el) == j]) for j in [1,2,3,4]] # number of parents for each category
        c2f_adjusted = {j: np.zeros((n,j)) for n,j in zip(n,[1,2,3,4]) if n != 0}
        f2c_adjusted = {j: np.zeros((n * j, 1)) for n,j in zip(n,[1,2,3,4]) if n != 0}

        print("GENERATING NUMERBING MAP")
        coarse_full_to_sub_map = {i: full_to_sub(coarse_mesh, c_subm[i]) for i in c_subm}
        fine_full_to_sub_map = {j: full_to_sub(mesh, f_subm[j]) for j in f_subm}
    
        print("CONSTRUCTING PROPER NUMBERINGS FOR SUBMESHES")
        for i in range(len(c2f)):
            n = len(c2f[i])
            if 1 <= n <= 4:
                c2f_adjusted[n][coarse_full_to_sub_map[n](i)] = fine_full_to_sub_map[n](np.array(c2f[i]))

        for j in range(len(f2c)):
            n = int(num_children[f2c[j]].item())
            if 1 <= n <= 4:
                f2c_adjusted[n][fine_full_to_sub_map[n](j), 0] = coarse_full_to_sub_map[n](f2c[j].item())

        # HAD TO CHANGE THIS, the getlevel() function used in mg/utils.py checks for level of submesh hierarchy, since we only have two levels per submesh hierarchy it will only look for 0 and 1 for each pair
        # What might happen if the submesh for when its the fine and coarse are the same???
        # c2f_subm = [{Fraction(len(self.meshes) - 2, 1): c2f_n} for c2f_n in c2f_adjusted]
        # f2c_subm = [{Fraction(len(self.meshes) - 1, 1): f2c_n} for f2c_n in f2c_adjusted]
        c2f_subm = [{Fraction(0, 1): c2f_n} for c2f_n in c2f_adjusted]
        f2c_subm = [{Fraction(1, 1): f2c_n} for f2c_n in f2c_adjusted]

        
        #hierarchy_dict = {f"{i+1}": HierarchyBase([c_subm[i], f_subm[i]], c2f_subm[i], f2c_subm[i], nested=True) for i in range(4) if c2f_subm[i][Fraction(len(self.meshes) - 2, 1)].shape[0] != 0}
        hierarchy_dict = {i+1: HierarchyBase([c_subm[i], f_subm[i]], c2f_subm[i], f2c_subm[i], nested=True) for i in c_subm}
        self.submesh_hierarchies.append(hierarchy_dict)

    def refine(self, refinements):
        ngmesh = self.meshes[-1].netgen_mesh
        for l, el in enumerate(ngmesh.Elements2D()):
            el.refine = 0
            if refinements[l] == 1:
                el.refine = 1

        ngmesh.Refine(adaptive=True)
        mesh = Mesh(ngmesh)
        self.add_mesh(mesh)
    
    def split_function(self, u, child=True):
        V = u.function_space()
        full_mesh = V.mesh()
        _, level = get_level(full_mesh)

        # to decide which submesh split to take
        ind = 1 if child else 0
        hierarchy_dict = self.submesh_hierarchies[int(level)-1-ind]
        
        split_functions = {}
        for i in hierarchy_dict:
            V_split = V.reconstruct(mesh=hierarchy_dict[i].meshes[ind])
            split_functions[i] = Function(V_split, name=str(i)).interpolate(u)
        return split_functions

    def recombine(self, split_funcs, f, child=True):      
        V = f.function_space()  


        mesh_label = split_funcs[1].function_space().mesh().submesh_parent
        V_label = V.reconstruct(mesh=mesh_label)
        f_label = Function(V_label, val=f.dat)
        for split_label, val in split_funcs.items():
            assert val.function_space().mesh().submesh_parent == mesh_label
            if child:
                split_label = int(str(split_label)*2)
                f_label.interpolate(val, subset=mesh_label.cell_subset(split_label))
            else:
                f_label.interpolate(val, subset=mesh_label.cell_subset(split_label))
        return f


def get_c2f_f2c_fd(mesh, num_parents, coarse_mesh):
    V = FunctionSpace(mesh, "DG", 0)
    V2 = FunctionSpace(coarse_mesh, "DG", 0) # dummy mesh for ng -> fd mapping
    ngmesh = mesh.netgen_mesh
    P = ngmesh.Coordinates()
    parents = ngmesh.GetParentSurfaceElements()
    fine_mapping = lambda x: mesh._cell_numbering.getOffset(x)
    coarse_mapping = lambda x: coarse_mesh._cell_numbering.getOffset(x)
    u = Function(V); u.rename("fd_parent_element") # store what netgen returns
   
    c2f = [[] for _ in range(num_parents)]
    f2c = [[] for _ in range(len(ngmesh.Elements2D()))]
    for l, el in enumerate(ngmesh.Elements2D()):
        pts = [P[k.nr-1] for k in list(el.vertices)] 
        bary = (1/3)*sum(pts)
        k = mesh.locate_cell(bary) # compute center of current element and locate index of element in mesh
        if parents[l] == -1 or l < num_parents: # need the second statement if multiple refinements occur on the same parent mesh
            u.dat.data[k] = coarse_mapping(l)
            f2c[fine_mapping(l)].append(coarse_mapping(l))
            c2f[coarse_mapping(l)].append(fine_mapping(l))
        elif parents[l] < num_parents:
            u.dat.data[k] = coarse_mapping(parents[l])
            f2c[fine_mapping(l)].append(coarse_mapping(parents[l]))
            c2f[coarse_mapping(parents[l])].append(fine_mapping(l))
        else: # correct mapping from Umberto
            u.dat.data[k] = coarse_mapping(parents[parents[l]])
            f2c[fine_mapping(l)].append(coarse_mapping(parents[parents[l]]))
            c2f[coarse_mapping(parents[parents[l]])].append(fine_mapping(l))
        
    #VTKFile(f"output/fd_mesh_test_{num_parents}.pvd").write(u)
    return c2f, np.array(f2c)

def split_to_submesh(mesh, coarse_mesh, c2f, f2c):
    V = FunctionSpace(mesh, "DG", 0)
    V2 = FunctionSpace(coarse_mesh, "DG", 0) 
    v = Function(V2, name="bisected_elements")
    z = Function(V2, name="trisected_elements")
    w = Function(V2, name="quadrisected_elements")
    coarse_intersect = Function(V2, name="unsplit_elements")
    v_fine = Function(V, name="bisected_children")
    z_fine = Function(V, name="trisected_children")
    w_fine = Function(V, name="quadrisected_children")
    fine_intersect = Function(V, name="unsplit_children")
    
    num_children = np.zeros((len(c2f)))

    for i in range(len(c2f)):
        v.dat.data[i], z.dat.data[i], w.dat.data[i], num_children[i] = 0, 0, 0, 1
        if len(c2f[i]) == 2: 
            v.dat.data[i] = 1
            num_children[i] = 2
        if len(c2f[i]) == 3:
            z.dat.data[i] = 1
            num_children[i] = 3
        if len(c2f[i]) == 4: 
            w.dat.data[i] = 1
            num_children[i] = 4 

    v_fine.dat.data[:], w_fine.dat.data[:], z_fine.dat.data[:] = np.zeros(v_fine.dat.data.shape), np.zeros(w_fine.dat.data.shape), np.zeros(z_fine.dat.data.shape)
    v_fine.dat.data[num_children[f2c.squeeze()] == 2] = 1
    z_fine.dat.data[num_children[f2c.squeeze()] == 3] = 1
    w_fine.dat.data[num_children[f2c.squeeze()] == 4] = 1

    coarse_intersect.dat.data[:] = np.logical_and(np.logical_and(v.dat.data == 0, w.dat.data == 0), z.dat.data == 0)
    fine_intersect.dat.data[:] = np.logical_and(np.logical_and(v_fine.dat.data == 0, w_fine.dat.data == 0), z_fine.dat.data == 0)
    
    return v, z, w, coarse_intersect, v_fine, z_fine, w_fine, fine_intersect, num_children
 
def full_to_sub(mesh, submesh):
    # returns the submesh element number associated with the mesh number
    V1=FunctionSpace(mesh, "DG", 0)
    V2=FunctionSpace(submesh,  "DG", 0)
    u1=Function(V1)
    u2=Function(V2)
    u2.dat.data[:] = np.arange(len(u2.dat.data))
    u1.interpolate(u2)
    
    return lambda x: u1.dat.data[x].astype(int)
       

if __name__ == "__main__":
    # mesh = UnitSquareMesh(4, 4)
    # mh = MeshHierarchy(mesh, 2)
    # print(mh.fine_to_coarse_cells)

    ## EXAMPLE 1: NG UNIFORM W TRANSFERMANAGER
    # wp = WorkPlane()
    # wp.Rectangle(2,2)
    # face = wp.Face()
    # geo = OCCGeometry(face, dim=2)
    # maxh = 0.1
    # ngmesh = geo.GenerateMesh(maxh=maxh)
    # mesh = Mesh(ngmesh)
    # mh = AdaptiveMeshHierarchy([mesh])
    # for i in range(2):
    #     ngmesh.Refine(adaptive=True)
    #     mesh = Mesh(ngmesh)
    #     mh.add_mesh(mesh, netgen_flags=True)

    # amh = mh.build()
    # xcoarse, ycoarse = SpatialCoordinate(amh[0])
    # xfine, yfine = SpatialCoordinate(amh[-1]) 
    # Vcoarse = FunctionSpace(amh[0], "DG", 0)
    # Vfine = FunctionSpace(amh[-1], "DG", 0)
    # u = Function(Vcoarse)
    # v = Function(Vfine)
    # u.rename("coarse")
    # v.rename("fine")
    # #Evaluate sin function on coarse mesh
    # u.interpolate(sin(pi*xcoarse)*sin(pi*ycoarse))
    # tm = TransferManager()
    # tm.prolong(u, v)
    # File("output_coarse.pvd").write(u)
    # File("output_fine.pvd").write(v)


    # ## EXAMPLE 2: NG NON-UNIFORM W TRANSFERMANAGER, won't work as of right now since requires fixed length arrays for c2f & f2c
    # import random
    # random.seed(1234)
    # wp = WorkPlane()
    # wp.Rectangle(2,2)
    # face = wp.Face()
    # geo = OCCGeometry(face, dim=2)
    # maxh = 0.1
    # ngmesh = geo.GenerateMesh(maxh=maxh)
    # mesh = Mesh(ngmesh, name = "coarse original")
    # amh = AdaptiveMeshHierarchy([mesh])
    # for_ref = np.zeros((len(ngmesh.Elements2D())))
    # for i in range(1):
    #     for l, el in enumerate(ngmesh.Elements2D()):
    #         el.refine = 0
    #         if random.random() < 0.5:
    #             el.refine = 1
    #             for_ref[l] = 1
    #     # el.refine = 1
    #     ngmesh.Refine(adaptive=True)
    #     mesh = Mesh(ngmesh, name="original")
    #     amh.add_mesh(mesh)
    #     #amh.refine(for_ref)

    # xcoarse, ycoarse = SpatialCoordinate(amh[0])
    # xfine, yfine = SpatialCoordinate(amh[-1]) 
    # Vcoarse = FunctionSpace(amh[0], "DG", 0)
    # Vfine = FunctionSpace(amh[-1], "DG", 0)

    # u = Function(Vcoarse)
    # v = Function(Vfine)
    # u.rename("coarse")
    # v.rename("fine")
    # #Evaluate sin function on coarse mesh
    # u.interpolate(sin(pi*xcoarse)*sin(pi*ycoarse))
    # # u.interpolate(xcoarse)


    # coarse_split = amh.split_function(u, child=False)
    # fine_split = amh.split_function(v, child=True)

    # tm = TransferManager()

    # # Interpolate
    # for i in range(1,5):
    #     if i in coarse_split:
    #         tm.prolong(coarse_split[i], fine_split[i])

    #         VTKFile(f"output/coarse_split_recomb_test_{i}.pvd").write(coarse_split[i])
    #         VTKFile(f"output/split_recomb_test_{i}.pvd").write(fine_split[i])
    # # tm.prolong(u_bi, v_bi)
    # # tm.prolong(u_tri, v_tri)
    # # tm.prolong(u_quad, v_quad)
    # # tm.prolong(u_unsplit, v_unsplit)
    # amh.recombine(coarse_split, u, child=False)

    # amh.recombine(fine_split, v, child=True)
    # VTKFile("output/output_coarse_mgtest.pvd").write(u)
    # VTKFile("output/output_fine_mgtest.pvd").write(v)

    # File("output/output_coarse_bi.pvd").write(u_bi)
    # File("output/output_coarse_tri.pvd").write(u_tri)
    # File("output/output_coarse_quad.pvd").write(u_quad)
    # File("output/output_coarse_unsplit.pvd").write(u_unsplit)
    # File("output/output_fine_bi.pvd").write(v_bi)
    # File("output/output_fine_tri.pvd").write(v_tri)
    # File("output/output_fine_quad.pvd").write(v_quad)
    # File("output/output_fine_unsplit.pvd").write(v_unsplit)



    # EXAMPLE 3
    import random
    random.seed(1234)
    wp = WorkPlane()
    wp.Rectangle(2,2)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    maxh = 1
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = Mesh(ngmesh)
    amh = AdaptiveMeshHierarchy([mesh])
    
    for i in range(2):
        for_ref = np.zeros((len(ngmesh.Elements2D())))
        for l, el in enumerate(ngmesh.Elements2D()):
            el.refine = 0
            if random.random() < 0.5:
                el.refine = 1
                for_ref[l] = 1
        # el.refine = 1
        ngmesh.Refine(adaptive=True)
        mesh = Mesh(ngmesh)
        amh.add_mesh(mesh)
        #amh.refine(for_ref)

    xcoarse, ycoarse = SpatialCoordinate(amh[0])
    xfine, yfine = SpatialCoordinate(amh[-1]) 
    Vcoarse = FunctionSpace(amh[0], "CG", 1)
    Vfine = FunctionSpace(amh[-1], "CG", 1)

    u = Function(Vcoarse)
    v = Function(Vfine)
    u.rename("coarse")
    v.rename("fine")
    #Evaluate sin function on coarse mesh
    u.interpolate(sin(pi*xcoarse)*sin(pi*ycoarse))

    atm = AdaptiveTransferManager()

    targets = atm.prolong(u, v, amh)
    
    VTKFile("output/output_coarse_atmtest.pvd").write(u)
    VTKFile("output/output_fine_atmtest.pvd").write(v)
    
   