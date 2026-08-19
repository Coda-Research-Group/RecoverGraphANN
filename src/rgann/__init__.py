"""Reproducibility artifact for the CIKM '26 paper.

*Recovering Graph ANN Search on Attention-Derived Workloads via a Query-Agnostic
Spherical Transformation* — https://doi.org/10.1145/3799682.3839984
"""

from rgann.transform import Normalization, apply_normalization, bachrach_transform, l2_normalize

__all__ = ['Normalization', 'apply_normalization', 'bachrach_transform', 'l2_normalize']
__version__ = '1.0.0'
