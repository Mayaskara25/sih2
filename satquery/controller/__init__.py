"""
Agentic controller: query routing, input validation, execution trace.

OWNED BY W0 AND FROZEN (PLAN.md §5.1). W6's modules (router.py, validate.py,
trace.py) were added by W6 — W6 is the only writer of this file, so this
carries no conflict risk. The top-level orchestration entry point is
``run_query`` (route -> validate -> dispatch five specialists -> trace).
"""

from satquery.controller.router import (
    SINGLE_IMAGE_TASKS,
    RouteResult,
    Tier1Rule,
    extract_grounding_target,
    recommended_input_counts,
    route,
)
from satquery.controller.trace import run_query, write_trace
from satquery.controller.validate import (
    PLAUSIBLE_BANDS,
    SUPPORTED_EXTENSIONS,
    ValidationResult,
    check_compatibility,
    inspect_input,
    validate_inputs,
)

__all__ = [
    "RouteResult",
    "Tier1Rule",
    "SINGLE_IMAGE_TASKS",
    "route",
    "extract_grounding_target",
    "recommended_input_counts",
    "ValidationResult",
    "SUPPORTED_EXTENSIONS",
    "PLAUSIBLE_BANDS",
    "inspect_input",
    "check_compatibility",
    "validate_inputs",
    "run_query",
    "write_trace",
]