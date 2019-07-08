# GaBP_solvers
## Generalized Gaussian belief propagation solvers

This repository accompanies the [article](https://arxiv.org/abs/1904.04093) where we proposed new linear solvers based on the generalized belief propagation.

Implementation of solvers and discretization can be found in bvp.py, equations.py, variance_solvers.py, and solvers.py.

Explanations of how to define and solve equations are in [template for your equations.ipynb](https://github.com/VLSF/GaBP_solvers/blob/master/template%20for%20your%20equations.ipynb).

Experiments with GaBP as a geometric multigrid solver are in [multigrid and GaBP.ipynb](https://github.com/VLSF/GaBP_solvers/blob/master/multigrid%20and%20GaBP.ipynb)

In the article, you can find an example of the matrix that is walk-summable in the generalized sense. File [generalized walk-summable model.ipynb](https://github.com/VLSF/GaBP_solvers/blob/master/generalized%20walk-summable%20model.ipynb) contains comments about the matrix.

Notebook [GaBP for non positive definite matrices.ipynb](https://github.com/VLSF/GaBP_solvers/blob/master/GaBP%20for%20non%20positive%20definite%20matrices.ipynb) contains numerical evidence that GaBP can handle real matrices with complex spectrum.

Two notebooks [condition number.ipynb](https://github.com/VLSF/GaBP_solvers/blob/master/condition%20number.ipynb), [convergence and refinement.ipynb](https://github.com/VLSF/GaBP_solvers/blob/master/convergence%20and%20refinement.ipynb) reproduce all figures from the article.

Sometimes github fails to render .ipynb, use [nbviewer](https://nbviewer.jupyter.org) in this case.
