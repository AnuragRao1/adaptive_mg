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
  ngmsh = geo.GenerateMesh(maxh=0.5)
  mesh = Mesh(ngmsh)

It is important to convert the initial Netgen mesh into a Firedrake mesh before constructing the AdaptiveMeshHierarchy. To call the constructor to the hierarchy, we must pass the initial mesh as a list. Our initial mesh looks like this:

.. figure:: initial_mesh.png
   :align: center
   :alt: Initial mesh.

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

For more information about patch relaxation, see `Using patch relaxation for multigrid <https://www.firedrakeproject.org/demos/poisson_mg_patches.py.html>`_. The initial solution is shown below

.. figure:: solution_l1.png
   :align: center
   :alt: Initial Solution from multigrid with initial mesh.


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

With these helper functions complete, we can solve the system iteratively. In the max_iterations is the number of total levels we want to perform multigrid on. We will solve for 15 levels. At every level :math:`l`, we first compute the solution using multigrid with patch relaxation up till level :math:`l`. We then use the current approximation of the solution to estimate the error across the mesh. Finally, we refine the mesh and repeat. ::

   max_iterations = 15
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

or alternatively to access it from the Firedrake mesh, ::

   for l, el in enumerate(mesh.netgen_mesh.Elements2D()):

The meshes now refine according to the error estimator. The error estimators levels 3,5, and 15 are shown below. Zooming into the vertex of the L at level 15 shows the error indicator remains strongest there. Further refinements will focus on that area.

+-------------------------------+-------------------------------+-------------------------------+
| .. figure:: eta_l3.png        | .. figure:: eta_l6.png        | .. figure:: eta_l15.png       |
|    :align: center             |    :align: center             |    :align: center             |
|    :width: 100%               |    :width: 100%               |    :width: 100%               |
|    :alt: Eta at level 3       |    :alt: Eta at level 6       |    :alt: Eta at level 15      |
|                               |                               |                               |
|    *Level 3*                  |     *Level 6*                 |    *Level 15*                 |
+-------------------------------+-------------------------------+-------------------------------+

The solutions at level 4 and 15 are shown below.

+------------------------------------+------------------------------------+
| .. figure:: solution_l4.png        | .. figure:: solution_l15.png       |
|    :align: center                  |    :align: center                  |
|    :width: 90%                     |    :width: 90%                     |
|    :alt: Solution, level 4         |    :alt: Solution, level 15        |
|                                    |                                    |
|    *MG solution at level 4*        |    *MG solution at level 15*       |
+------------------------------------+------------------------------------+


The convergence follows the expected behavior:

.. figure:: adaptive_convergence_9.png
   :align: center
   :alt: Convergence of the error estimator.

