from firedrake import *
mesh = UnitSquareMesh(2, 2)
M = FunctionSpace(mesh, "DG", 0)
m = Function(M)
m.dat.data[0] = 1

rmesh = RelabeledMesh(mesh, [m], [100])
submesh = Submesh(rmesh, rmesh.geometric_dimension(), 100)

V = FunctionSpace(rmesh, "DG", 0).dual()
Vsub = V.reconstruct(mesh=submesh)
u = Function(V)
usub = Function(Vsub)
usub.assign(1)

u.assign(usub, allow_missing_dofs=True)
print(u.dat.data)