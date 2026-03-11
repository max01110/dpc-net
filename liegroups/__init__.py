"""Vendored liegroups package (minimal subset) for DPC-Net.

This is a local copy of the upstream `liegroups` package to avoid relying on
pip installation on Narval. Only the pieces required by DPC-Net are included.
"""

from .numpy import SO3 as SO3
from .numpy import SE3 as SE3

