For a demo for how to use the AdaptiveMeshHierarchy and AdaptiveTransferManager, see demo/demo.py.rst. \
To run the experiments, first move into the experiments directory with `cd experiments`.
The central command for running each: \
`python run_expr.py --system curl --p None --theta 0.5 --lam_alg 0.01 --alpha 2/3 --dim 1e4 --solver direct`

The parameters are as follows:
- system: Which system wants to be run. The options are
  - curl
  - div
  - grad
  - kellogg
- p: Polynomial Degree. Specify which polynomial basis to run experiments for. p=None will run all p=1-4
- theta: Dorfler marking parameter, theta in [0,1]
- lam_alg: iterative solver parameter that controls convergence, lam_alg > 0.
- alpha: exponent to radial component for curl/div problems. alpha in (0,1)
- dim: Number of maximum degrees of freedom to refine mesh until
- solver: There are two options (specific parameter options can be found/changed in each experiment script):
  -direct: Direct solvers
  -mg: multigrid (implemented with patch relaxation)

To generate the plots for each system, run \
`python plot_expr.py --system curl --p None --theta 0.5 --lam_alg 0.01 --alpha 2/3 --dim 1e4 --solver direct --dir_name None`

The parameters are the same as above, with the only new one being `dir_name`, which allows the user to specify the directory if the generated results were renamed/moved into a different location.
