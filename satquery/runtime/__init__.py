"""
Runtime model management (PLAN.md §4.3): one heavy model resident at a time.

OWNED BY W0 AND FROZEN (PLAN.md §5.1).

Imports ModelPool lazily-ish: constructing the singleton is free (no GPU touch,
no weight download), so importing this package is safe anywhere.
"""

from satquery.runtime.modelpool import ModelPool, RoleSpec, model_pool

__all__ = ["ModelPool", "RoleSpec", "model_pool"]
