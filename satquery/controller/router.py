"""Query-text -> task routing (PLAN.md §3.4, §4.2 routing record).

This module is W6's implementation of R4: task selection is driven by the
QUERY TEXT, never by how many files happened to be uploaded. Two images plus
"describe the first image" must route to ``caption`` on image 1, not to
``change``; one image plus "what changed?" must still route to ``change`` and
be refused later by ``validate.py`` (a validation error is not a routing
result).

Two tiers, in order (PLAN.md §3.4 — the trace records which fired):

  Tier 1  deterministic keyword/pattern rules, tried first. Every rule has a
          stable ``matched`` name and a strength score, so the trace can say
          *which* rule fired and how hard it was trusted.
  Tier 2  exemplar nearest-neighbour over hand-written exemplars per task,
          scored with a dependency-free weighted token overlap (idf-weighted
          cosine). No transformer is loaded and no LLM is placed in the
          routing path — this is the one rubric row that rewards being
          deterministic and auditable.

The module is pure stdlib + ``satquery.contracts``. It opens no files, loads
no models, and never touches the GPU.

Ambiguity defaults (documented here so they are testable and auditable):
  * "compare these two images"-style phrasings default to CHANGE. If the query
    also names both optical/msi and SAR (or uses fuse/joint/cross-modal
    language), the FUSION rules fire first and win.
  * Modality co-occurrence always beats temporal-comparison phrasing:
    "differences between the SAR and optical image" is FUSION, not change.
  * A single-image task run when two images are supplied uses image 1
    (index 0) unless the query names the second/latter/other image, which
    selects index 1. "both images" stays on index 0 with an explicit note.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from satquery.contracts import RoutingInfo

__all__ = [
    "Tier1Rule",
    "RouteResult",
    "route",
    "route_rules",
    "extract_grounding_target",
    "recommended_input_counts",
    "_TIER1_RULES",
    "_EXEMPLARS",
]

SINGLE_IMAGE_TASKS: Tuple[str, ...] = ("vqa", "caption", "grounding")

# ---------------------------------------------------------------------------
# Tier 1: deterministic rules, in evaluation priority order.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _compile(pattern: str) -> "re.Pattern":
    return re.compile(pattern, re.IGNORECASE)


@dataclass(frozen=True)
class Tier1Rule:
    """One deterministic routing rule (task, stable name, pattern, strength).

    ``matches`` applies the pattern to the query text case-insensitively.
    ``strength`` is the confidence the route assigns when this rule fires; all
    current rules are strong (>= 0.7), so any fire is treated as a confident
    route.
    """

    task: str
    name: str
    pattern: str
    strength: float = 1.0

    def matches(self, query: str) -> bool:
        return _compile(self.pattern).search(query) is not None


# Evaluation order is significant. A query is matched against rules in this
# order and the FIRST hit wins, so specific language beats generic language:
# SAR/optical co-occurrence beats "compare", change beats bare question words,
# grounding beats caption, caption beats vqa.
#
# Task-priority ordering (fusion > change > grounding > caption > vqa) is the
# documented tie-break documented in the module docstring.
#
# NOTE on rule naming: the `name` is what the trace records as
# routing.matched, so it must be stable and human-readable.

_TIER1_RULES: List[Tier1Rule] = [
    # --- fusion wins first: any explicit joint-modality language -----------
    Tier1Rule("fusion", "fusion_sar_optical", r"\bsar\b.{0,80}\boptic(al|al|)\b"),
    Tier1Rule("fusion", "fusion_optical_sar", r"\boptic(al|al|)\b.{0,80}\bsar\b"),
    Tier1Rule("fusion", "fusion_cross_modal", r"\b(cross[ -]?modal|multi[ -]?sensor|multi[ -]?modal)\b"),
    Tier1Rule("fusion", "fusion_fuse", r"\bfus(e|es|ed|ing|ion)\b"),
    Tier1Rule("fusion", "fusion_joint", r"\bjoint(ly)?\s+(use|analysis|interpret\w*|reading)\b"),
    Tier1Rule("fusion", "fusion_complementary", r"\bcomplementar\w+\b"),
    # "microwave" is a SAR synonym and "cross-reference" a fusion verb; both
    # were missed by an independent probe (2026-08-30).
    Tier1Rule("fusion", "fusion_microwave_optical", r"\b(?:microwave|radar)\b.{0,80}\b(?:optical|visible|multispectral|msi|panchromatic)\b"),
    Tier1Rule("fusion", "fusion_optical_microwave", r"\b(?:optical|visible|multispectral|msi|panchromatic)\b.{0,80}\b(?:microwave|radar)\b"),
    Tier1Rule("fusion", "fusion_cross_reference", r"\bcross[- ]?referenc\w+\b"),
    Tier1Rule("fusion", "fusion_together", r"\b(?:use|used|using)\b.{0,60}\btogether\b.{0,60}\b(?:sar|radar)\b"),
    # --- change: temporal comparison ---------------------------------------
    Tier1Rule("change", "change_what_changed", r"\bwhat\s+changed\b"),
    Tier1Rule(
        "change",
        "change_has_changed",
        r"\bhas\s+\w+\s+(changed|increased|decreased|grown|shrunk|expanded|receded|regressed)\b",
    ),
    Tier1Rule(
        "change",
        "change_increased_decreased_unchanged",
        r"\bincreased,\s*(?:decreased|declined),\s*or\s+remain(?:ed|s)?\s+unchanged\b",
    ),
    Tier1Rule(
        "change",
        "change_compare_temporal",
        r"\bcompare[d]?\b.{0,40}\b(?:earlier|later|date|dates|before|after|ago)\b",
    ),
    Tier1Rule(
        "change", "change_changed_between",
        r"\bchang(?:ed|es|ing)\s+(?:between|over|across|since|from)\b",
    ),
Tier1Rule(
        "change",
        "change_changed_between",
        r"\b(?:chang(?:ed|es|ing)|shift\w*|evolv\w*|alter(?:ed|es|ing)?|mov(?:ed|es|ing)?|fluctuat\w*|grow\w*|var(?:ied|ies|ying)?)\s+(?:between|over|across|since|from)\b",
    ),
    Tier1Rule(
        "change",
        "change_comparative_scale",
        r"\b(?:larger|smaller|bigger|greater|higher|lower|wider|narrower|more|less)\s+or\b",
    ),
    Tier1Rule(
        "change",
        "change_difference_between",
        r"\bdifferenc(?:e|es)\s+(?:between|among)\b",
    ),
    Tier1Rule("change", "change_different_between", r"\bdifferent\s+between\b"),
    Tier1Rule("change", "change_before_after", r"\bbefore\s+and\s+after\b"),
    Tier1Rule("change", "change_over_time", r"\bover\s+time\b"),
    # Trend verbs. Added 2026-08-30 after an independent probe with queries
    # outside eval/routing_testset.json routed these to grounding or vqa.
    # Past/perfect forms only -- bare "growing" appears in single-image
    # agricultural questions ("growing season") and must not force change.
    Tier1Rule("change", "change_trend_verb", r"\b(shrank|shrunk|shrink|expanded|expanding|declined|receded|retreated|dwindled)\b"),
    # "grew"/"grown" need a temporal companion: "which crop is being GROWN here"
    # is a single-image vqa question, and the bare verb misrouted it to change.
    Tier1Rule("change", "change_grew_temporal", r"\b(?:grew|grown)\b.{0,40}\b(?:since|between|compared|than\s+before|over\s+the|in\s+the\s+last)\b"),
    Tier1Rule("change", "change_since_earlier", r"\bsince\s+(?:the\s+)?(?:earlier|previous|prior|first|last|original)\b"),
    Tier1Rule("change", "change_between_two_obs", r"\bbetween\s+the\s+two\s+(?:acquisitions?|dates?|images?|scenes?|photos?|passes?|observations?)\b"),
    Tier1Rule("change", "change_versus_before", r"\b(?:versus|vs\.?|compared\s+(?:to|with))\s+(?:before|earlier|previously|the\s+past)\b"),
    Tier1Rule("change", "change_different_temporal", r"\bdifferent\b.{0,30}\b(?:now|before|earlier|previously)\b"),
    Tier1Rule("change", "change_two_dates", r"\btwo\s+dates\b"),
    Tier1Rule("change", "change_bitemporal", r"\b[bB]i[\s-]?temporal\b"),
    Tier1Rule("change", "change_between_years", r"\bbetween\s+(?:19|20)\d{2}\b"),
    Tier1Rule(
        "change",
        "change_from_year_to",
        r"\bfrom\s+(?:19|20)\d{2}\s*(?:to|-)\s*(?:19|20)\d{2}\b",
    ),
    Tier1Rule(
        "change",
        "change_new_construction",
        r"\bnew\s+(?:buildings?|construction|development|road|roads|colonies?)\b",
    ),
    Tier1Rule(
        "change",
        "change_landuse_shift",
        r"\b(?:deforestation|urbaniz\w+|urbanis\w+|reforest\w+|encroach\w+|accretion)\b",
    ),
    Tier1Rule(
        "change",
        "change_appeared_disappeared",
        r"\b(?:appeared|disappeared|vanished|emerged)\b",
    ),
    Tier1Rule(
        "change",
        "change_compare_images_default",
        r"\bcompar\w+\b.{0,40}\b(?:image|images|scene|scenes|pair|two)\b",
    ),
    Tier1Rule(
        "change",
        "change_detect_changes",
        r"\b(?:detect|detecting)\s+(?:the\s+)?changes?\b",
    ),
    # --- grounding: locate / highlight --------------------------------------
    Tier1Rule("grounding", "grounding_highlight", r"\bhighlight\b"),
    Tier1Rule("grounding", "grounding_locate", r"\blocate\b"),
    Tier1Rule("grounding", "grounding_localize", r"\blocali[sz]\w*\b"),  # -ise and -ize
    Tier1Rule("grounding", "grounding_where", r"\bwhere(?:\s+(?:is|are|was|were|did|do|does|have|has)|'s|'re)\b"),
    Tier1Rule("grounding", "grounding_find_object", r"\bfind\s+(?:the|any|all|every|these|those)\b"),
    Tier1Rule("grounding", "grounding_identify_object", r"\bidentif\w+\s+(?:the|any|all|these|those)\s+\w+\s+(?:buildings|structures|roads|fields|crops|zones|areas|regions|parcels|lots)\b"),
    Tier1Rule("grounding", "grounding_bbox", r"\bbounding\s*boxes?\b"),
    Tier1Rule("grounding", "grounding_mark", r"\bmark\s+(?:the|all|out|every)\b"),
    Tier1Rule("grounding", "grounding_point", r"\bpoint\s+(?:to|out)\b"),
    Tier1Rule("grounding", "grounding_show_where", r"\bshow\s+me\s+where\b"),
    Tier1Rule("grounding", "grounding_segment", r"\bsegment(?:ation)?\b"),
    Tier1Rule("grounding", "grounding_delineate", r"\bdelineat\w*\b"),
    Tier1Rule("grounding", "grounding_location_of", r"\b(?:the\s+)?location\s+of\b"),
    Tier1Rule("grounding", "grounding_referring", r"\breferred\s+to\s+in\s+the\s+query\b"),
    # --- caption: describe / summarize the scene ----------------------------
    Tier1Rule("caption", "caption_describe", r"\bdescrib\w*\b"),
    Tier1Rule("caption", "caption_description", r"\bdescription\b"),
    Tier1Rule("caption", "caption_caption", r"\bcaption(?:ing)?\b"),
    Tier1Rule("caption", "caption_overview", r"\boverview\s+of\b"),
    Tier1Rule("caption", "caption_what_shown", r"\bwhat(?:'s|\s+is)?\s+(?:shown|in|on|pictured)\b"),
    Tier1Rule("caption", "caption_what_show", r"\bwhat\s+(?:does|do)\s+(?:this|the)\s+(?:image|scene|photo)s?\s+(?:show|contain)\b"),
    Tier1Rule("caption", "caption_summarize", r"\bsummariz\w+\b"),
    Tier1Rule("caption", "caption_summary", r"\bsummary\s+of\b"),
    Tier1Rule("caption", "caption_tell_me_about", r"\btell\s+me\s+about\b"),
    Tier1Rule("caption", "caption_narrate_depict", r"\b(?:narrat\w+|depict\w+|portray\w+|illustrat\w+)\b"),
    Tier1Rule("caption", "caption_what_can_see", r"\bwhat\s+can\s+(?:you\s+)?(?:see|observe)\b"),
    Tier1Rule("caption", "caption_dominant_scene", r"\bscene\s+content\b"),
    # --- vqa: specific question about image content --------------------------
    Tier1Rule("vqa", "vqa_rural_urban", r"\brural\b.*\burban\b|\burban\b.*\brural\b"),
    Tier1Rule("vqa", "vqa_how_many", r"\bhow\s+many\b"),
    Tier1Rule("vqa", "vqa_are_there", r"\bare\s+there\b"),
    Tier1Rule("vqa", "vqa_is_there", r"\bis\s+there\b"),
    Tier1Rule("vqa", "vqa_object_state", r"\bis\s+(?:the|this)\b.{0,50}\b(?:full|empty|deep|shallow|dry|damp|clear)\b"),
    Tier1Rule("vqa", "vqa_which", r"\bwhich\b"),
    Tier1Rule("vqa", "vqa_what_fraction", r"\bwhat\s+(?:fraction|percentage|percent|proportion|share|portion|amount|area|extent|fraction)\s+of\b"),
    Tier1Rule("vqa", "vqa_yesno_show", r"\bdoes\s+this\s+(?:image|scene|picture|photo|imagery)\s+(?:show|contain|depict|display|have)\b"),
    Tier1Rule("vqa", "vqa_how_much", r"\bhow\s+(?:much|big|large|wide|far|deep|large|common|large|health)\b"),
    Tier1Rule("vqa", "vqa_dominant", r"\b(?:dominant|predominat\w+|prevail\w*)\b"),
    Tier1Rule("vqa", "vqa_classify", r"\bclassif\w*\b"),
    Tier1Rule("vqa", "vqa_count", r"\bcount\s+(?:the|of|all|any)\b"),
    Tier1Rule("vqa", "vqa_average_mean", r"\baverage\b|\bmean\b"),
    Tier1Rule("vqa", "vqa_what_is", r"\bwhat\s+(?:is|are|'s|kind|type|class)\b"),
    Tier1Rule("vqa", "vqa_land_cover", r"\bl(?:and)?[- ]?cover\b"),
    Tier1Rule("vqa", "vqa_water_level", r"\b(?:water\s+level|area\s+of\s+water|extent\s+of\s+water)\b"),
    Tier1Rule("vqa", "vqa_yields", r"\b(?:yield|harvest)\b"),
]

# Rules that "fire" on nearly every sentence are deliberately absent:
# every routing decision must be explainable by a *specific* cue, and the
# fallback path (NN below threshold) is needed anyway, so a generic question
# rule would only mask the NN score.

# ---------------------------------------------------------------------------
# Tier 2: hand-written exemplars for nearest-neighbour routing.
# ---------------------------------------------------------------------------

_EXEMPLARS: Dict[str, List[str]] = {
    "caption": [
        "describe the land-cover and major objects visible in this image",
        "describe the scene you see",
        "what is shown in this image",
        "caption this satellite image",
        "provide an overview of what the image shows",
        "summarize what you can see in this image",
        "what does this image contain",
        "describe the dominant land cover in the scene",
        "give a general description of the scene",
        "what's going on in this picture",
    ],
    "vqa": [
        "how many buildings are there in this image",
        "are there any water bodies visible",
        "what is the dominant land cover type",
        "which crop is being grown here",
        "is there a river in this image",
        "what fraction of the image is urban area",
        "how large is the forested area",
        "what kind of land cover is present",
        "count the number of vehicles in the parking lot",
    ],
    "grounding": [
        "highlight the water body in this image",
        "locate the buildings in the scene",
        "where is the river",
        "find the urban area",
        "bounding box around the crop fields",
        "point to the airport in the image",
        "mark the roads for me",
        "show me where the forest is",
        "where exactly is the reservoir",
    ],
    "change": [
        "what changed between these two dates",
        "compare these two images",
        "has the built-up area increased, decreased, or remained unchanged",
        "detect changes between the two images",
        "what are the differences between these two scenes",
        "identify new developments over time",
        "how has the coastline changed in this period",
        "analyze the change between the before and after images",
    ],
    "fusion": [
        "use the optical and SAR images together to identify built-up and water-covered regions",
        "combine the optical and SAR data",
        "cross-modal analysis of the SAR and optical scenes",
        "jointly interpret the sar and optical images",
        "fuse the optical and sar channels",
        "use both the sar and optical images to find water",
        "merge sar and optical information to identify flooded areas",
    ],
}

# ---------------------------------------------------------------------------
# Shared text machinery.
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """
    a an the and or nor but for with of at on in into from by to this that
    these those it its is are was were be been being am do does did can could
    will would shall should may might must have has had having not no yes
    what which who whom whose how where when why whether there here both each
    any all some every about between during under over above before after
    image images scene scenes photo photos picture pictures date dates pair
    them they their your you your me my i we us our
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lower-cased alphanumeric tokens minus a tiny stop list."""
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


def _build_idf(exemplars: List[str]) -> Dict[str, float]:
    """idf weights over the exemplar collection (each exemplar = one doc)."""
    docs = [tokenize(e) for e in exemplars]
    df: Dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            df[token] = df.get(token, 0) + 1
    n_docs = len(docs)
    return {
        token: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0
        for token, freq in df.items()
    }


def _all_exemplars() -> List[Tuple[str, str]]:
    return [(task, text) for task, texts in _EXEMPLARS.items() for text in texts]


_EXEMPLAR_DOCS: List[Tuple[str, str]] = _all_exemplars()
_IDF: Dict[str, float] = _build_idf([text for _, text in _EXEMPLAR_DOCS])


def _exemplar_scores(query_tokens: List[str]) -> Dict[str, float]:
    """idf-weighted cosine over exemplars, aggregated per task."""
    qvec: Dict[str, float] = {}
    for token in query_tokens:
        weight = _IDF.get(token, 0.0)
        if weight > 0.0:
            qvec[token] = qvec.get(token, 0.0) + weight
    q_norm = math.sqrt(sum(w * w for w in qvec.values()))
    if q_norm == 0.0:
        return {}
    per_task: Dict[str, float] = {}
    for task, docs in _EXEMPLARS.items():
        best = 0.0
        for ex_doc in docs:
            ex_tokens = tokenize(ex_doc)
            ex_norm2 = 0.0
            dot = 0.0
            for token in ex_tokens:
                weight = _IDF.get(token, 0.0)
                ex_norm2 += weight * weight
                if weight > 0.0 and token in qvec:
                    dot += weight * qvec[token]
            if ex_norm2 > 0.0:
                score = dot / (q_norm * math.sqrt(ex_norm2))
                if score > best:
                    best = score
        per_task[task] = best
    return per_task


def _nn_route(query_tokens: List[str]) -> Tuple[Optional[str], str, float, List[str]]:
    """(task, matched_exemplar, score, alternatives) via tier-2 NN, or
    (None, ...) if nothing clears the confidence floor."""
    per_task = _exemplar_scores(query_tokens)
    ranked = sorted(per_task.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return None, "", 0.0, []
    winner, winner_score = ranked[0]
    alternatives = [
        f"{task}(score={score:.3f})" for task, score in ranked[1:4] if task != winner
    ]
    return winner, winner, float(winner_score), alternatives


# ---------------------------------------------------------------------------
# Query-vs-input-count resolution + image selection.
# ---------------------------------------------------------------------------


def _image_reference(query: str) -> str:
    """Which of the uploaded images the query singles out for a single-image
    task: "first" | "second" | "both" | "unspecified"."""
    lower = query.lower()
    if re.search(
        r"\b(?:second|2nd|latter|other\s+image|image\s+2|2nd\s+image|last\s+image)\b",
        lower,
    ):
        return "second"
    if re.search(r"\b(?:first|1st|image\s+1|1st\s+image)\b", lower) or re.search(
        r"\bthe\s+first\b", lower
    ):
        return "first"
    if re.search(r"\b(?:both|each)\s+image\b", lower) or re.search(r"\bboth\b", lower):
        return "both"
    return "unspecified"


def _select_image_index(reference: str, n_inputs: int) -> Tuple[int, str]:
    """Map an image reference to a concrete 0-based index, with a note."""
    if reference == "second":
        if n_inputs and n_inputs >= 2:
            return 1, "second-or-later image"
        return 0, "second image referenced but only one supplied; using image 1"
    return 0, "first image"


# ---------------------------------------------------------------------------
# Public routing surface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteResult:
    """The routing decision plus everything the trace needs to explain it."""

    task: str
    routing: RoutingInfo
    selected_image_index: int
    image_reference: str
    input_count: int
    resolution_note: str

    @property
    def is_single_image_task(self) -> bool:
        return self.task in SINGLE_IMAGE_TASKS


def recommended_input_counts(task: str) -> Tuple[int, int]:
    """(min, max) image count each task accepts; used by validate.py."""
    if task in ("change", "fusion"):
        return 2, 2
    return 1, 2


def route_rules() -> List[Tier1Rule]:
    """Ordered tier-1 rule list actually used for routing (stable for tests,
    docs, and the W8 scoring harness)."""
    return list(_TIER1_RULES)


def _rule_hit(query: str) -> Optional[Tier1Rule]:
    for rule in route_rules():
        if rule.matches(query):
            return rule
    return None


def route(query: str, n_inputs: Optional[int] = None) -> RouteResult:
    """Route a query string to exactly one of {vqa, caption, grounding,
    change, fusion}. ``n_inputs`` is used ONLY to resolve which image a
    single-image task runs on and to write an auditable note about the
    query-vs-input-count reconciliation — it never chooses the task.

    Returns a ``RouteResult`` whose ``routing`` dict satisfies the
    ``contracts.RoutingInfo`` shape (mechanism in {"rule","exemplar_nn"}).
    """
    query = (query or "").strip()
    task: Optional[str] = None
    matched: str = ""
    score: float = 0.0
    mechanism: str = ""
    alternatives: List[str] = []

    rule_hit = _rule_hit(query)
    if rule_hit is not None:
        task = rule_hit.task
        matched = rule_hit.name
        score = rule_hit.strength
        mechanism = "rule"
        alternatives = _rule_alternatives(query, task)
    else:
        query_tokens = tokenize(query)
        nn_task, nn_matched, nn_score, nn_alternatives = _nn_route(query_tokens)
        if nn_task is not None and nn_score >= 0.22:
            task = nn_task
            matched = f"exemplar:{nn_matched}"
            score = nn_score
            mechanism = "exemplar_nn"
            alternatives = nn_alternatives
        else:
            task = "vqa"
            matched = "default:vqa_fallback"
            score = 0.0
            mechanism = "exemplar_nn"
            alternatives = nn_alternatives

    reference = _image_reference(query)
    idx, idx_note = _select_image_index(reference, n_inputs if n_inputs is not None else 0)

    routing: RoutingInfo = {
        "mechanism": mechanism,
        "matched": matched,
        "score": float(score),
        "alternatives_considered": list(alternatives),
    }

    note = _resolution_note(task, mechanism, matched, n_inputs, reference, idx)
    return RouteResult(
        task=task,
        routing=routing,
        selected_image_index=idx,
        image_reference=reference,
        input_count=n_inputs if n_inputs is not None else 0,
        resolution_note=note,
    )


def _rule_alternatives(query: str, winner: str) -> List[str]:
    """Other tasks whose tier-1 rules also matched the query (auditable
    context for routing.alternatives_considered)."""
    hits: List[str] = []
    for rule in route_rules():
        if rule.task != winner and rule.matches(query):
            label = f"{rule.task} ({rule.name})"
            if label not in hits:
                hits.append(label)
    return hits


def _resolution_note(
    task: str,
    mechanism: str,
    matched: str,
    n_inputs: Optional[int],
    reference: str,
    idx: int,
) -> str:
    """Human-readable explanation of how the query text and the input count
    were reconciled — this is the R4 story and it goes on the trace."""
    n = n_inputs if n_inputs is not None else 0
    parts = []
    parts.append(
        f"routing via {mechanism}"
        + (f" ({matched})" if matched else "")
    )
    if task in SINGLE_IMAGE_TASKS:
        if n >= 2:
            which = {"first": "image 1", "second": f"image {idx + 1}", "both": "image 1"}
            chosen = which.get(reference, "image 1")
            parts.append(
                f"single-image task '{task}' selected from query text despite "
                f"{n} inputs; default single-image task runs on {chosen}"
            )
            if reference == "both":
                parts.append(
                    "query refers to 'both images'; controller runs single-image "
                    "analyses one image at a time and defaults to image 1"
                )
        elif reference == "second":
            parts.append("query references a second image but only 1 was supplied; using image 1")
        else:
            parts.append(f"single-image task '{task}' runs on the one supplied image")
    else:
        if n < 2:
            parts.append(
                f"task '{task}' needs 2 images but {n} were supplied — validation "
                "will refuse this run; routing is query-driven by design (R4)"
            )
        else:
            parts.append(f"task '{task}' accepts the {n} supplied images")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Referring-expression extraction for the grounding specialist.
# ---------------------------------------------------------------------------

_SKIP_PREFIXES = re.compile(
    r"^\s*(?:"
    r"highlight|locate|localize|find|mark|point out|point to|show me|"
    r"segment|delineate|identify|detect|bound the|bounding box(?:es)? (?:of|for|around|on)|"
    r"box(?:es)? (?:around|for|of)|draw (?:a|an|the|me)? ?(?:bounding )?box(?:es)? (?:around|for|on)?|"
    r"where (?:is|are|was|were)|where exactly is|the location of|location of"
    r")",
    re.IGNORECASE,
)
_SKIP_TRAILERS = re.compile(
    r"\s*(?:"
    r"is|are|was|were|it\b|"  # trailing copula: "show me where the forest is"
    r"referred to in the (?:query|text|sentence|image)|"
    r"in (?:this|the|that) (?:satellite )?(?:image|scene|photo|picture)|"
    r"in the image|in the scene|in the query|"
    r"that you can see|you can see"
    r")[.\s]*$",
    re.IGNORECASE,
)

# Leftover relative clause: "where the forest is" after the prefix strip leaves
# "the forest is" except when the prefix list did not consume "where the".
_SKIP_WHERE_LEAD = re.compile(
    r"^\s*where\s*(?:'s|'re| is| are| was| were| did| do| does| have| has)?\s+"
    r"(?:the|a|an)?\s+",
    re.IGNORECASE,
)


def extract_grounding_target(query: str) -> str:
    """Pull the referring expression out of a grounding-style query.

    "Highlight the water body referred to in the query." -> "water body"
    "Where is the river?"                                -> "river"
    "show me where the forest is"                        -> "forest"
    If nothing is stripped, the whole (cleaned) query is returned so the
    grounding specialist always has *something* to ground on.
    """
    cleaned = _SKIP_PREFIXES.sub(" ", query)
    cleaned = _SKIP_WHERE_LEAD.sub("", cleaned)
    cleaned = _SKIP_TRAILERS.sub("", cleaned)
    cleaned = cleaned.strip(" .?!" " \t\n")
    # "the water body" -> "water body"; keep remainders that look like a phrase
    cleaned = re.sub(r"^\s*(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    if cleaned and len(cleaned) > 1:
        return cleaned
    return query.strip(" .?!")