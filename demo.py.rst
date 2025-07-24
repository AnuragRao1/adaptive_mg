Adaptive Multigrid Methods using the AdaptiveMeshHierarchy
=================================================

The purpose of this demo is to show how to use Firedrake's multigrid solver on a hierarchy of adaptively refined Netgen meshes.
We will first have a look at how to use the AdaptiveMeshHierarchy to construct the mesh hierarchy with Netgen meshes, then we will consider a solution to the Poisson problem on an L shaped domain.
Finally, we will show how to use the AdaptiveMeshHierarchy and AdaptiveTransferManager as arguments to Firedrake solvers. The AdaptiveMeshHierarchy contains information of the mesh hierarchy and the parent child relations between the meshes.
The AdaptiveTransferManager deals with the transfer operator logic across any given levels in the hierarchy.
We begin by importing the necessary libraries ::

   from firedrake import *
   from netgen.occ import *

Constructing the Mesh Hierarchy
---------------------------
We first must construct the domain over which we will solve the problem. For a more comprehensive demo on how to use Open Cascade Technology (OCC) and Constructive Solid Geometry (CSG), see `here <https://www.firedrakeproject.org/demos/netgen_mesh.py.html>`_ . 
We begin with the L-shaped domain, which we build as the union of two rectangles: ::
  rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
  rect2 = WorkPlane(Axes((0,1,0), n=Z, h=X)).Rectangle(2,1).Face()
  L = rect1 + rect2
  
  geo = OCCGeometry(L, dim=2)
  ngmsh = geo.GenerateMesh(maxh=0.1)
  mesh = Mesh(ngmsh)

It is important to convert the initial Netgen mesh into a Firedrake mesh before constructing the AdaptiveMeshHierarchy. To call the constructor to the hierarchy, we must pass the initial mesh as a list. 
We will also initialize the AdaptiveTransferManager here: ::
  
  amh = AdaptiveMeshHierarchy([mesh])
  atm = AdaptiveTransferManager()

Poisson Problem
-------------------------
Now we can define a simple Poisson Problem

.. math::

   - \nabla^2 u = f \text{ in } \Omega, \quad u = 0 \text{ on } \partial \Omega

We define the function solve_poisson. The first lines correspond to finding a solution in the CG1 space. The variational problem is formulated in F, where f is the constant function equal to 1: ::

   def solve_poisson(mesh, params):
    V = FunctionSpace(mesh, "CG", 1)
    uh = Function(V, name="Solution")
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(0), "on_boundary")
    f = Constant(1)
    F = inner(grad(uh), grad(v))*dx - inner(f, v)*dx

    problem = NonlinearVariationalProblem(F, uh, bc)

    dm = uh.function_space().dm
    old_appctx = get_appctx(dm)
    mat_type = params["mat_type"]
    appctx = _SNESContext(problem, mat_type, mat_type, old_appctx)
    appctx.transfer_manager = atm
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.set_transfer_manager(atm)
    with dmhooks.add_hooks(dm, solver, appctx=appctx, save=False):
        coarsen(problem, coarsen)

    solver.solve()
    return uh

Note the code after the construction of the NonlinearVariationalProblem(). To use the AdaptiveMeshHierarchy with the existing Firedrake solver, we have to execute the following lines before we call solver.solve(). 
The AdaptiveTransferManager has to be set to the appctx and solver in order to use a multigrid solver. ::
Now we are ready to assemble the stiffness matrix for the problem. Since we want to enforce Dirichlet boundary conditions we construct a `DirichletBC` object and we use the `GetRegionNames` method from the Netgen mesh in order to map the label we have given when describing the geometry to the PETSc `DMPLEX` IDs. In particular if we look for the IDs of boundary element labeled either "line" or "curve" we would get::

   labels = [i+1 for i, name in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ["line","curve"]]
   bc = DirichletBC(V, 0, labels)
   print(labels)

We then proceed to solve the problem::

   sol = Function(V)
   solve(a == L, sol, bcs=bc)
   VTKFile("output/Poisson.pvd").write(sol)


Adaptive Mesh Refinement
-------------------------
In this section we will discuss how to use the mesh refinement methods wrapped from Netgen C++ interface.
In particular we will be considering a Laplace eigenvalue problem on the same PacMan domain presented above, i.e.:

.. math::

   \text{Find } u \in H^1_0(\Omega) \text{ and } \lambda \in \mathbb{R} \text{ s.t. } \int_{\Omega} \nabla u\cdot\nabla v\;d\vec{x} = \lambda \int_{\Omega}uv\;d\vec{x}\qquad \forall v\in H^1_0(\Omega).

This script is based on a code developed by Professor Daniele Boffi and based on a code from Professor Douglas Arnold for the source problem.
We begin by defining some quantities of interest such as the desired tolerance, the maximum number of iterations and the exact eigenvalue::

   from firedrake.petsc import PETSc
   from slepc4py import SLEPc
   import numpy as np

   tolerance = 1e-16
   max_iterations = 10
   exact = 3.375610652693620492628**2

We create a function to solve the eigenvalue problem using SLEPc. We begin initialising the `FunctionSpace`, the bilinear forms and linear functionals needed in the variational problem.
Then a SLEPc Eigenvalue Problem Solver (`EPS`) is initialised and set up to use a shift and invert (`SINVERT`) spectral transformation where the preconditioner factorisation is computed using MUMPS::

   def Solve(msh, labels):
        V = FunctionSpace(msh, "CG", 2)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        m = (u*v)*dx
        uh = Function(V)
        bc = DirichletBC(V, 0, labels)
        A = assemble(a, bcs=bc)
        M = assemble(m, bcs=bc, weight=0.)
        Asc, Msc = A.M.handle, M.M.handle
        E = SLEPc.EPS().create()
        E.setType(SLEPc.EPS.Type.ARNOLDI)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        E.setDimensions(1, SLEPc.DECIDE)
        E.setOperators(Asc, Msc)
        ST = E.getST()
        ST.setType(SLEPc.ST.Type.SINVERT)
        PC = ST.getKSP().getPC()
        PC.setType("lu")
        PC.setFactorSolverType("mumps")
        E.setST(ST)
        E.solve()
        vr, vi = Asc.getVecs()
        with uh.dat.vec_wo as vr:
            lam = E.getEigenpair(0, vr, vi)
        return (lam, uh, V)

We will also need a function that mark the elements that need to be marked according to an error indicator, i.e.

.. math::
   \eta = \sum_{K\in \mathcal{T}_h(\Omega)} h^2\int_K|\lambda u_h + \Delta u_h|^2\;d\vec{x}+\frac{h}{2}\int_{E\subset \partial K} | [\![ \nabla u\cdot n_E]\!] | ^2\; ds

In order to do so we begin by computing the value of the indicator using a piecewise constant function space::

   def Mark(msh, uh, lam):
      W = FunctionSpace(msh, "DG", 0)
      eta_sq = Function(W)
      w = TestFunction(W)
      f = Constant(1)
      h = CellDiameter(msh)  # symbols for mesh quantities
      n = FacetNormal(msh)
      v = CellVolume(msh)

      G = (  # compute cellwise error estimator
            inner(eta_sq / v, w)*dx
          - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
          - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS
          - inner(h('-')/2 * jump(grad(uh), n)**2, w('-')) * dS
          )

      # Each cell is an independent 1x1 solve, so Jacobi is exact
      sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
      solve(G == 0, eta_sq, solver_parameters=sp)
      eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2

      with eta.dat.vec_ro as eta_:  # compute estimate for error in energy norm
          error_est = sqrt(eta_.dot(eta_))
      markers = Function(W)

      # We decide to refine an element if its error indicator
      # is within a fraction of the maximum cellwise error indicator

      # Access storage underlying our Function
      # (a PETSc Vec) to get maximum value of eta
      with eta.dat.vec_ro as eta_:
          eta_max = eta_.max()[1]

      theta = 0.5
      should_refine = conditional(gt(eta, theta*eta_max), 1, 0)
      markers.interpolate(should_refine)
      return markers

It is now time to define the solve, mark and refine loop that is at the heart of the adaptive method described here::


   for i in range(max_iterations):
        print("level {}".format(i))
        lam, uh, V = Solve(msh, labels)
        mark = Mark(msh, uh, lam)
        msh = msh.refine_marked_elements(mark)
        VTKFile("output/AdaptiveMeshRefinement.pvd").write(uh)

Note that the mesh conforms to the CAD geometry as it is adaptively refined.


.. figure:: Adaptive.png
   :align: center
   :alt: Outcome of the adaptive mesh refinement process.

Constructive Solid Geometry in 3D
---------------------------------
In this section we will focus our attention on three dimensional constructive solid geometry. In particular we will look at the operators `+,-,*~`, which have been overridden to have a special meaning when applied to two instances of the class `CSGeometry`.
It is important to notice that the same operators can be used also when working with a `SplineGeometry` and their action will have the same meaning that is presented here.
The `+,-,*` operators have respectively the meaning of union, set difference, and intersection. We will build a cube using the planes intersection and remove from it a portion of sphere::

   from netgen.csg import *
   left = Plane(Pnt(0, 0, 0), Vec(-1, 0, 0))
   right = Plane(Pnt(1, 1, 1), Vec(1, 0, 0))
   front = Plane(Pnt(0, 0, 0), Vec(0, -1, 0))
   back = Plane(Pnt(1, 1, 1), Vec(0, 1, 0))
   bot = Plane(Pnt(0, 0, 0), Vec(0, 0, -1))
   top = Plane(Pnt(1, 1, 1), Vec(0, 0, 1))
   cube = left * right * front * back * bot * top
   cube.bc("cube")
   sphere = Sphere(Pnt(0.6, 0.6, 0.6), 0.5)
   geo = CSGeometry()
   geo.Add(cube-sphere)
   ngmsh = geo.GenerateMesh(maxh=0.1)
   msh = Mesh(ngmsh)
   VTKFile("output/MeshExample3.pvd").write(msh)


Open Cascade Technology
-----------------------
Last we will have a look at the Netgen Open Cascade Technology interface, which has been recently included. We will follow the tutorial presented in the `NetGen docs <https://docu.ngsolve.org/nightly/i-tutorials/unit-4.4-occ/bottle.html>`__, which itself comes from the OCCT tutorial `here <https://dev.opencascade.org/doc/overview/html/occt__tutorial.html>`__.
The idea is to draw a "flask" using the OCCT interface and solve the linear elasticity equations to compute the stress tensor on the flask subject to gravity.
We begin importing the Netgen Open Cascade interface and constructing the bottom of the flask using many different method such as `Axes, Face, Pnt, Segment, ...` (all the details this methods can be found in `NetGen docs <https://docu.ngsolve.org/nightly/i-tutorials/unit-4.4-occ/bottle.html>`__

::

   from netgen.occ import *
   myHeight = 70
   myWidth = 50
   myThickness = 30
   pnt1 = Pnt(-myWidth / 2., 0, 0)
   pnt2 = Pnt(-myWidth / 2., -myThickness / 4., 0)
   pnt3 = Pnt(0, -myThickness / 2., 0)
   pnt4 = Pnt(myWidth / 2., -myThickness / 4., 0)
   pnt5 = Pnt(myWidth / 2., 0, 0)
   seg1 = Segment(pnt1, pnt2)
   arc = ArcOfCircle(pnt2, pnt3, pnt4)
   seg2 = Segment(pnt4, pnt5)
   wire = Wire([seg1, arc, seg2])
   mirrored_wire = wire.Mirror(Axis((0, 0, 0), X))
   w = Wire([wire, mirrored_wire])
   f = Face(w)
   f.bc("bottom")

Once the bottom part of the flask has been constructed we then extrude it to create the main body. We now construct the neck of the flask and fuse it with the main body::

   body = f.Extrude(myHeight*Z)
   body = body.MakeFillet(body.edges, myThickness / 12.0)
   neckax = Axes(body.faces.Max(Z).center, Z)
   myNeckRadius = myThickness / 4.0
   myNeckHeight = myHeight / 10
   neck = Cylinder(neckax, myNeckRadius, myNeckHeight)
   body = body + neck
   fmax = body.faces.Max(Z)
   thickbody = body.MakeThickSolid([fmax], -myThickness / 50, 1.e-3)

Last we are left to construct the threading of the flask neck and fuse it to the rest of the flask body. In order to do this we are going to need the value of pi, which we grab from the Python math package::

   import math
   cyl1 = Cylinder(neckax, myNeckRadius * 0.99, 1).faces[0]
   cyl2 = Cylinder(neckax, myNeckRadius * 1.05, 1).faces[0]
   aPnt = Pnt(2 * math.pi, myNeckHeight / 2.0)
   aDir = Dir(2 * math.pi, myNeckHeight / 4.0)
   anAx2d = gp_Ax2d(aPnt, aDir)
   aMajor = 2 * math.pi
   aMinor = myNeckHeight / 10
   arc1 = Ellipse(anAx2d, aMajor, aMinor).Trim(0, math.pi)
   arc2 = Ellipse(anAx2d, aMajor, aMinor/4).Trim(0, math.pi)
   seg = Segment(arc1.start, arc1.end)
   wire1 = Wire([Edge(arc1, cyl1), Edge(seg, cyl1)])
   wire2 = Wire([Edge(arc2, cyl2), Edge(seg, cyl2)])
   threading = ThruSections([wire1, wire2])
   bottle = thickbody + threading
   geo = OCCGeometry(bottle)

As usual, we generate a mesh for the described geometry and use the Firedrake-Netgen interface to import as a PETSc DMPLEX::

   ngmsh = geo.GenerateMesh(maxh=5)
   msh = Mesh(ngmsh)
   VTKFile("output/MeshExample4.pvd").write(msh)

.. figure:: Bottle.png
   :align: center
   :alt: Example of the mesh generated from a bottle geometry described using Open Cascade.
