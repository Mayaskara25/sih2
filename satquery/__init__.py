"""
SatQuery AI — agentic vision-language assistant for remote sensing.
SIH 2026, ISRO problem statement 26167. See PLAN.md for the build plan.

OWNED BY W0 AND FROZEN (PLAN.md §5.1). Every module's exports are pre-declared
here and in the sub-package __init__ files, pointing at the W0 stubs. This is
deliberate: when W3/W4/W5 replace a stub body, nothing changes at package level,
so no two agents ever edit the same __init__.py and there is no merge conflict
by construction.

Deliberately does NOT eagerly import specialists or runtime — importing
`satquery` must stay cheap and must never touch torch or the GPU. Import the
sub-packages explicitly.
"""

__version__ = "0.1.0"

from satquery import contracts

__all__ = ["contracts", "__version__"]
