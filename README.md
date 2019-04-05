# GaBP_solvers
## Generalized Gaussian belief propagation solvers

This repository accompanies the article (link) where we proposed new linear solvers based on the generalized belief propagation.

There are two jupyter notebooks: "reproduce figures" and "template for your equations". Name are self-explanatory. The second notebook is of use for those who wants to check the performance of our solvers on his/her favourite linear elliptic equation.

Most of the code utilizes numpy and scipy, but BP.py script which is based on graph [graph tool library](https://graph-tool.skewed.de). We also provide pure python (numpy and scipy) implementation of the belief propagation (python_BP.py). Whilst it is far slower, there is no need to deal with graph tools which is not straightforward to install sometimes.
