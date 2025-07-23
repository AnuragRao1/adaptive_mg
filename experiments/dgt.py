from firedrake import *

mesh = UnitSquareMesh(1, 1)
n = FacetNormal(mesh)
t = as_vector([-n[1], n[0]])

p = 1
V = FunctionSpace(mesh, "CG", p)

(x, y) = SpatialCoordinate(mesh)
#u = Function(V).interpolate(sin(x)*cos(y))
u = y

W = FunctionSpace(mesh, "DGT", p)

w = Function(W, name="TangentialGradient")
v = TestFunction(W)
both = lambda v: v("+") + v("-")

F = (
      inner(w, v)*ds
    - inner(dot(grad(u), t), v)*ds
    + Constant(1.0e-16)*inner(both(w), both(v))*dS
    )

solve(F == 0, w)
print(w.dat.data)

#for pt in [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]:
#    print(f"w({pt}) = {w.at(pt)}")
