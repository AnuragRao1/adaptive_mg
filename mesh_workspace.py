from netgen.occ import *
import numpy as np

import firedrake
from firedrake import *
from adaptive_transfer_manager import AdaptiveTransferManager
from adaptive import AdaptiveMeshHierarchy
from firedrake.mg.utils import get_level


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
# cube = Box(Pnt(0,0,0), Pnt(2,2,2))
# geo = OCCGeometry(cube, dim=3)
maxh = 1
ngmesh = geo.GenerateMesh(maxh=maxh)
mesh = Mesh(ngmesh)
mesh2 = Mesh(ngmesh)
amh = AdaptiveMeshHierarchy([mesh])
# VTKFile("output/meshes/initial_mesh.pvd").write(Function(FunctionSpace(mesh, "DG", 0)))

for i in range(1):
    # for_ref = np.zeros((len(ngmesh.Elements2D())))
    # for l, el in enumerate(ngmesh.Elements3D()):
    #     el.refine = 0
    #     # if random.random() < 0.05:
    #     #     print(l)
    #     #     el.refine = 1
    # el.refine = 1
    #         # for_ref[l] = 1
    for l, el in enumerate(ngmesh.Elements2D()):
        el.refine = 0
        if random.random() < 0.4:
            el.refine = 1
    
    ngmesh.Refine(adaptive=True)
    mesh = Mesh(ngmesh)
    amh.add_mesh(mesh)
    # amh.refine(for_ref)

# mh = MeshHierarchy(mesh2, 1)
# u = Function(FunctionSpace(mh[-1], "DG", 0))
# v = Function(FunctionSpace(amh[-1], "DG", 0))

# for i in range(1,5):
#     coarse_mesh = amh.submesh_hierarchies[0][i].meshes[0]
#     fine_mesh = amh.submesh_hierarchies[0][i].meshes[1]
#     u = Function(FunctionSpace(coarse_mesh, "DG", 0))
#     v = Function(FunctionSpace(fine_mesh, "DG", 0))


#     VTKFile(f"output/meshes/csubm_{i}.pvd").write(u)
#     VTKFile(f"output/meshes/fsubm_{i}.pvd").write(v)


# for i in range(2):
#     refs = np.ones(len(ngmesh.Elements2D()))
#     amh.refine(refs)
    

xcoarse, _ = SpatialCoordinate(amh[0])
xfine, _ = SpatialCoordinate(amh[-1]) 
Vcoarse = FunctionSpace(amh[0], "CG", 1)
Vfine = FunctionSpace(amh[-1], "CG", 1)
u = Function(Vcoarse)
v = Function(Vfine)
u.rename("coarse")
v.rename("fine")

# PROLONG

#Evaluate sin function on coarse mesh
u.interpolate(xcoarse)
u.interpolate(sin(pi * xcoarse))
atm = AdaptiveTransferManager()

atm.prolong(u, v)
VTKFile("output/split_transfer/output_coarse_atmtest.pvd").write(u)
VTKFile("output/split_transfer/output_fine_atmtest.pvd").write(v)


# # RESTRICT
# u.interpolate(xcoarse)
# atm = AdaptiveTransferManager()
# atm.prolong(u, v)


# # rf = Cofunction(Vfine.dual()).assign(1)
# rf = assemble(TestFunction(Vfine)*dx)
# rc = Cofunction(Vcoarse.dual()) 
# atm.restrict(rf, rc)

# assembled_rc = assemble(TestFunction(Vcoarse)*dx)
# print("Adaptive TM: ", assemble(action(rc, u)), assemble(action(rf, v)))
# assert (assemble(action(rc, u)) - assemble(action(rf, v))) / assemble(action(rf, v)) <= 1e-2

