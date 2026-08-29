"""
Specialist tools. Contracts in PLAN.md §4.1 / satquery.contracts.

OWNED BY W0 AND FROZEN (PLAN.md §5.1). All five entry points are declared here
against the W0 stubs. W3/W4/W5 replace the stub BODIES in the individual
modules; they must not edit this file. If you need a new export, that is a §5.5
contract change — stop and report it.

  run_vqa, run_caption  -> vqa.py        [W3]
  run_grounding         -> grounding.py  [W3]
  run_change            -> change.py     [W4]
  run_fusion            -> fusion.py     [W5]

Per PLAN.md §5.3 no specialist imports another specialist; shared code lives in
satquery.io / satquery.runtime / satquery.adapters.
"""

from satquery.specialists.change import run_change
from satquery.specialists.fusion import run_fusion
from satquery.specialists.grounding import run_grounding
from satquery.specialists.vqa import run_caption, run_vqa

__all__ = ["run_vqa", "run_caption", "run_grounding", "run_change", "run_fusion"]
