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
We first must construct the domain over which we will solve the problem. For a more comprehensive demo on how to use Open Cascade Technology (OCC) and Constructive Solid Geometry (CSG), see `Netgen integration in Firedrake <https://www.firedrakeproject.org/demos/netgen_mesh.py.html>`_ . 
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

Our approach strongly follows the similar problem in this `lecture course <https://github.com/pefarrell/icerm2024>`_. We define the function solve_poisson. The first lines correspond to finding a solution in the CG1 space. The variational problem is formulated with F, where f is the constant function equal to 1. Since we want Dirichlet boundary conditions, we construct the DirichletBC object and apply it to the entire boundary: ::

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

Note the code after the construction of the NonlinearVariationalProblem(). To use the AdaptiveMeshHierarchy with the existing Firedrake solver, we have to execute the intermediate lines before we call solver.solve(). 
The AdaptiveTransferManager has to be set to the appctx and solver in order to use a multigrid solver.
For the parameters of the multigrid solver, we will be using patch relaxation, which we define with ::
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
           "ksp_type": "cg",
           "pc_type": "mg",
           "mg_levels": {
               "ksp_type": "chebyshev",
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

For more information about patch relaxation, see `Using patch relaxation for multigrid <https://www.firedrakeproject.org/demos/poisson_mg_patches.py.html>`_.

Adaptive Mesh Refinement
-------------------------
In this section we will discuss how to adaptively refine select elements and add the newly refined mesh into the AdaptiveMeshHierarchy.
For this problem, we will be using the Babuška-Rheinbolt a-posteriori estimate for an element:

.. math::
   \eta_K^2 = h_K^2 \int_K | f + \nabla^2 u_h |^2 \mathrm{d}x + \frac{h_K}{2} \int_{\partial K \setminus \partial \Omega} \llbracket \nabla u_h \cdot n \rrbracket^2 \mathrm{d}s,

where :math:`K` is the element, :math:`h_K` is the diameter of the element, :math:`n` is the normal, and :math:`\llbracket \cdot \rrbracket` is the jump operator. The a-posteriori estimator is computed using the solution at the current level :math:`h`. We can use a trick to compute the estimator on each element. We transform the above estimator into the variational problem 

.. math::
   \int_\Omega \eta_K^2 w \mathrm{d}x = \int_\Omega \sum_K h_K^2 \int_K (f + \text{div} (\text{grad} u_h) )^2 \mathrm{d}x w \mathrm{d}x + \int_\Omega \sum_K \frac{h_K}{2} \int_{\partial K \setminus \partial \Omega} \llbracket \nabla u_h \cdot n \rrbracket^2 \mathrm{d}s w \mathrm{d}x

Our approach will be to compute the estimator over all elements and selectively choose to refine only those that contribute most to the error. To compute the error estimator, we use the function below to solve the variational formulation of the error estimator. Since our estimator is a constant per element, we use a DG0 function space.  ::

   def estimate_error(mesh, uh):
       W = FunctionSpace(mesh, "DG", 0)
       eta_sq = Function(W)
       w = TestFunction(W)
       f = Constant(1)
       h = CellDiameter(mesh)  # symbols for mesh quantities
       n = FacetNormal(mesh)
       v = CellVolume(mesh)
   
       G = (  # compute cellwise error estimator
             inner(eta_sq / v, w)*dx
           - inner(h**2 * (f + div(grad(uh)))**2, w) * dx
           - inner(h('+')/2 * jump(grad(uh), n)**2, w('+')) * dS
           - inner(h('-')/2 * jump(grad(uh), n)**2, w('-')) * dS
           )
   
       sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
       solve(G == 0, eta_sq, solver_parameters=sp)
       eta = Function(W).interpolate(sqrt(eta_sq))  # compute eta from eta^2
   
       with eta.dat.vec_ro as eta_:  # compute estimate for error in energy norm
           error_est = sqrt(eta_.dot(eta_))
       return (eta, error_est)

The next step is to choose which elements to refine. For this we Dörfler marking, developed by Professor Willy Dörfler:  

.. math::
   \eta_K \geq \theta \text{max}_L \eta_L

The logic is to select an element :math:`K` to refine if the estimator is greater than some factor :math:`\theta` of the maximum error estimate of the mesh, where :math:`\theta` ranges from 0 to 1. In our code we choose :math:`theta=0.5`. We implement this in the following function::

   def adapt(mesh, eta):
       W = FunctionSpace(mesh, "DG", 0)
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
   
       refined_mesh = mesh.refine_marked_elements(markers)
       return refined_mesh

With these helper functions complete, we can solve the system iteratively. In the max_iterations is the number of total levels we want to perform multigrid on. We will solve for 10 levels. At every level :math:`l`, we first compute the solution using multigrid with patch relaxation up till level :math:`l`. We then use the current approximation of the solution to estimate the error across the mesh. Finally, we refine the mesh and repeat. ::

   max_iterations = 10
   error_estimators = []
   dofs = []
   for i in range(max_iterations):
       print(f"level {i}")
   
       uh = solve_poisson(mesh, patch_relax)
       VTKFile(f"output/poisson_l/{max_iterations}/adaptive_loop_{i}.pvd").write(uh)
   
       (eta, error_est) = estimate_error(mesh, uh)
       VTKFile(f"output/poisson_l/{max_iterations}/eta_{i}.pvd").write(eta)
   
       print(f"  ||u - u_h|| <= C x {error_est}")
       error_estimators.append(error_est)
       dofs.append(uh.function_space().dim())
   
       mesh = adapt(mesh, eta)
       if i != max_iterations - 1:
           amh.add_mesh(mesh)

To add the mesh to the AdaptiveMeshHierarchy, we us the amh.add_mesh() method. In this method the input is the refined mesh. There is another method for adding a mesh to the hierarchy. This is the amh.refine([to_refine]). In this method, to_refine is a list of 1's or 0's, where a 1 at index i means the elements[i] should be refined. It is important to note that this method assumes the input list considers the elements in the order than Netgen enumerates them, not Firedrake. This enumeration can be found with ::

   for l, el in enumerate(ngmesh.Elements2D()):

or alternatively ::

   for l, el in enumerate(mesh.netgen_mesh.Elements2D()):

for 2D elements for example. The convergence of the  

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
