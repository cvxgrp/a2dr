### Examples

This repo contains scripts and notebooks for additional tests and reproducing the plots in the paper.

To reproduce the plots in the paper, simply run [notebooks/paper_plots.ipynb](https://github.com/cvxgrp/a2dr/edit/master/examples/notebooks/), which imports [paper_examples/paper_plots.py](https://github.com/cvxgrp/a2dr/edit/master/examples/paper_examples/), run the examples, and store the plots in [figures/](https://github.com/cvxgrp/a2dr/edit/master/examples/figures).

You can also play with individual examples by running commands like the following:
```python
python nnls.py
```
which will show the plots but won't store it automatically. The sizes in the individual example scripts are deliberately set differently from the paper to encourage exploration.
