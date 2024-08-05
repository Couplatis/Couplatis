"""
Couplatis
=========

Couplatis is a deep learning-based study of the dynamical
properties of nonlinear coupled neural networks.
"""

import torch

MODE = "cuda" if torch.cuda.is_available() else "cpu"
