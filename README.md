For a demo for how to use the AdaptiveMeshHierarchy and AdaptiveTransferManager, see [demo/demo.py.rst](https://github.com/AnuragRao1/adaptive_mg/blob/main/demo/demo.py.rst). \
To use the code, the dev version of Firedrake must be installed. Installation instructions can be found [here](https://www.firedrakeproject.org/install.html). The branch ['arao/atm'](https://github.com/firedrakeproject/firedrake/tree/arao/atm) is needed to run this code. Another version of Netgen different than the one in the main installation instructions is also necessary. This can be installed with the command `pip install --pre netgen-mesher`.

To run the experiments, first move into the experiments directory with `cd experiments`.
The central command for running each: \
`python run_expr.py --system curl --p None --theta 0.5 --lam_alg 0.01 --alpha 2/3 --dim 1e4 --solver direct --experiment bump`

The parameters are as follows:
- system: which system wants to be run. The options are
  - curl
  - div
  - grad
  - kellogg
- p: polynomial degree. Specify which polynomial basis to run experiments for. p=None will run all p=1-4
- theta: Dorfler marking parameter, theta in [0,1]
- lam_alg: iterative solver parameter that controls convergence, lam_alg > 0.
- alpha: exponent to radial component for curl/div problems. alpha in (0,1)
- dim: Number of maximum degrees of freedom to refine mesh until
- solver: There are two options (specific parameter options can be found/changed in each experiment script):
  -direct: Direct solvers
  -mg: multigrid (implemented with patch relaxation)
-experiment: which experiment to run, input as a string. Options are "bump", "donut", and "unknown"

To generate the plots for each system, run \
`python plot_expr.py --system curl --p None --theta 0.5 --lam_alg 0.01 --alpha 2/3 --dim 1e4 --solver direct --experiment bump --dir_name None --w_uniform False --unif_dir None`

The parameters are the same as above, with the only new ones being
- dir_name: str value. allows the user to specify the directory of data if the generated results were renamed/moved into a different location. To be input as a string "path/to/data"
- w_uniform: Boolean value. Controls whether convergence plots for adaptively refined (theta != 0) are to be plotted against uniform (theta = 0)
- unif_dir: str value. allows user to specify directory of uniformly refined data if renamed/moved. Input as string "path/to/uniform/data"
