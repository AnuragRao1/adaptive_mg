from netgen.occ import *
import pdb 
import numpy as np

import firedrake
from firedrake.utils import cached_property
from firedrake.cython import mgimpl as impl
from firedrake.mg.utils import set_level
from firedrake.mg import HierarchyBase
from firedrake import *
from fractions import Fraction


class AdaptiveMeshHierarchy():
    def __init__(self, mesh):
        self.meshes = tuple(mesh)
        self._meshes = tuple(mesh)
        
    def add_mesh(self, mesh, netgen_flags=False):
        #check new mesh is fd mesh or netgen, if not reject
        if (isinstance(netgen_flags, bool) and netgen_flags) and not hasattr(mesh, "netgen_mesh"):
            raise RuntimeError("NO SUPPORT FOR OTHER MESH TYPES OR NEED TO INPUT NG MESH USING Mesh()")
        
        self._meshes += tuple(mesh)
        self.meshes += tuple(mesh)
        # self._shared_data_cache ???

        if len(self.meshes) <= 2: 
            self.coarse_to_fine_cells = {}
            self.fine_to_coarse_cells = {}
            self.fine_to_coarse_cells[Fraction(0,1)] = None

        if (isinstance(netgen_flags, bool) and netgen_flags) and hasattr(mesh, "netgen_mesh"):
            # extract parent child relationships from netgen meshes,
            num_parents = self.meshes[-2].num_cells() # store number of parent elements from previous mesh
            (c2f, f2c) = get_c2f_f2c_fd(mesh, num_parents, self.meshes[-2])
            print(get_c2f_f2c_netgen(mesh, num_parents)) # run for sanity

        ## TO DO: FIX FD MESH SECTION OR ASK ABOUT IT
        # else:
        #     lgmaps = []
        #     for i in range(len(self.meshes) - 2, len(self.meshes)):
        #         # only care for the last two since we have everything else already
        #         m = self.meshes[i]
        #         no = impl.create_lgmap(m.topology_dm)
        #         m.init()
        #         o = impl.create_lgmap(m.topology_dm)
        #         m.topology_dm.setRefineLevel(i)
        #         lgmaps.append((no, o))
            
        #     c2f, f2c = impl.coarse_to_fine_cells(self.meshes[len(self.meshes)-2], self.meshes[len(self.meshes)-1] , lgmaps[0], lgmaps[1])
        
        self.coarse_to_fine_cells[Fraction(len(self.meshes) - 2,1)] = np.array(c2f) # for now fix refinements_per_level to be 1
        self.fine_to_coarse_cells[Fraction(len(self.meshes) - 1,1)] = np.array(f2c)


    def build(self):
        return HierarchyBase(self.meshes, self.coarse_to_fine_cells, self.fine_to_coarse_cells, 1, nested=True)

def get_c2f_f2c_netgen(mesh, num_parents):
    V = FunctionSpace(mesh, "DG", 0)
    ngmesh = mesh.netgen_mesh
    P = ngmesh.Coordinates()
    parents = ngmesh.GetParentSurfaceElements()
    print(parents)
    u = Function(V); u.rename("netgen_parent_element") # store what netgen returns
    v = Function(V); v.rename("parent_element") # store the correctly mapped parent elements

    c2f = [[] for _ in range(num_parents)]
    f2c = [[] for _ in range(len(ngmesh.Elements2D()))]
    for l, el in enumerate(ngmesh.Elements2D()):
        pts = [P[k.nr-1] for k in list(el.vertices)] 
        bary = (1/3)*sum(pts)
        k = mesh.locate_cell(bary) # compute center of current element and locate index of element in mesh
        u.dat.data[k] = parents[l] # store netgen res
        if parents[l] == -1 or l < num_parents: # need the second statement if multiple refinements occur on the same parent mesh
            v.dat.data[k] = l
            f2c[l].append(l)
            c2f[l].append(l)
        elif parents[l] < num_parents:
            v.dat.data[k] = parents[l]
            f2c[l].append(parents[l])
            c2f[parents[l]].append(l)
        else:
            v.dat.data[k] = parents[parents[l]] # correct mapping to parent from Umberto
            f2c[l].append(parents[parents[l]])
            c2f[parents[parents[l]]].append(l)

    VTKFile(f"output/ng_correct_parent_{num_parents}.pvd").write(u, v)
    return c2f, f2c


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

    VTKFile(f"output/fd_mesh_test_{num_parents}.pvd").write(u)
    return c2f, f2c



if __name__ == "__main__":
    # mesh = UnitSquareMesh(4, 4)
    # mh = MeshHierarchy(mesh, 2)
    # print(mh.fine_to_coarse_cells)

    ## EXAMPLE 1: NG UNIFORM W TRANSFERMANAGER
    wp = WorkPlane()
    wp.Rectangle(2,2)
    face = wp.Face()
    geo = OCCGeometry(face, dim=2)
    maxh = 0.1
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = Mesh(ngmesh)
    mh = AdaptiveMeshHierarchy([mesh])
    for i in range(2):
        ngmesh.Refine(adaptive=False)
        mesh = Mesh(ngmesh)
        mh.add_mesh(mesh, netgen_flags=True)

    amh = mh.build()
    xcoarse, ycoarse = SpatialCoordinate(amh[0])
    xfine, yfine = SpatialCoordinate(amh[-1]) 
    Vcoarse = FunctionSpace(amh[0], "DG", 0)
    Vfine = FunctionSpace(amh[-1], "DG", 0)
    u = Function(Vcoarse)
    v = Function(Vfine)
    u.rename("coarse")
    v.rename("fine")
    #Evaluate sin function on coarse mesh
    u.interpolate(sin(pi*xcoarse)*sin(pi*ycoarse))
    tm = TransferManager()
    tm.prolong(u, v)
    File("output_coarse.pvd").write(u)
    File("output_fine.pvd").write(v)


    ## EXAMPLE 2: NG NON-UNIFORM W TRANSFERMANAGER, won't work as of right now since requires fixed length arrays for c2f & f2c
    # import random
    # random.seed(1234)
    # wp = WorkPlane()
    # wp.Rectangle(2,2)
    # face = wp.Face()
    # geo = OCCGeometry(face, dim=2)
    # maxh = 0.1
    # ngmesh = geo.GenerateMesh(maxh=maxh)
    # mesh = Mesh(ngmesh)
    # mh = AdaptiveMeshHierarchy([mesh])
    # for i in range(2):
    #     for el in ngmesh.Elements2D():
    #         el.refine = 0
    #         if random.random() < 0.5:
    #             el.refine = 1
        
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

    ## EXAMPLE 3: TEST c2f & f2c
    # mh = MeshHierarchy(base, 2) # 2 refinements
    # print(mh.coarse_to_fine_cells)
    # print(mh.fine_to_coarse_cells)

    # wp = WorkPlane()
    # wp.Rectangle(2,2)
    # face = wp.Face()
    # geo = OCCGeometry(face, dim=2)
    # maxh = 2
    # ngmesh = geo.GenerateMesh(maxh=maxh)
    # mesh = Mesh(ngmesh) # NEED TO USE THIS FIRST
    
    # amh = AdaptiveMeshHierarchy([mesh]) # need to input singular mesh as list
    # for n in range(2):
    #     for i,el in enumerate(ngmesh.Elements2D()):
    #         el.refine = 1

    #     ngmesh.Refine(adaptive=True)
    #     amh.add_mesh(Mesh(ngmesh), netgen_flags=True)
    # print(amh.coarse_to_fine_cells)
    # print(amh.fine_to_coarse_cells)
    
   