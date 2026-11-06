"""Aggregate the corpus walk into the §2 calibration tables.

Counters for anything the generator needs as a rate; reservoirs for anything it needs as
a *sample* (identifier vocabularies, padding blocks, span lengths for the IQR bands).
Reservoir sampling keeps memory flat over an arbitrarily large corpus and gives a uniform
sample without a second pass.
"""

from __future__ import annotations

import ast
import collections
import warnings
from dataclasses import dataclass, field

import numpy as np

#: seed anchors per lexical domain. The corpus carries no domain labels, so domains are
#: operationalised: score each file by anchor density, take the densest files per domain,
#: harvest their identifiers. The seeds are the declared part and go into PROVENANCE.
#: Anchors are chosen to be *distinctive*, not merely topical. The first pass used
#: "level", "frame", "render", "order" and "sample", which fire on logging, dataframes,
#: templating and SQL across the whole corpus: "gaming" collected 33,679 files, four
#: times "finance", and the identifiers harvested were the same generic Python names in
#: every domain.
DOMAIN_SEEDS = {
    "finance": ("ledger", "invoice", "portfolio", "accrual", "creditor", "debit",
                "amortis", "payable", "brokerage", "settlement"),
    "biology": ("genome", "nucleotide", "protein", "phylogen", "assay", "organism",
                "chromosome", "enzyme", "taxon", "plasmid"),
    "gaming": ("sprite", "hitbox", "respawn", "gameplay", "playerstate", "tilemap",
               "projectile", "matchmaking", "powerup", "npc_"),
    "physics": ("velocity", "wavelength", "photon", "lattice", "quantum", "spectrom",
                "kinematic", "luminos", "scattering", "calorimeter"),
    "ecommerce": ("checkout", "sku", "storefront", "fulfilment", "fulfillment",
                  "shopping_cart", "coupon", "shipment", "merchant", "restock"),
}

#: an identifier must be this much more frequent in its own domain than elsewhere to
#: count as domain vocabulary; below it the name is generic Python (`AttributeError`,
#: `Column`, `CONF`) and cancels nothing the other domains do not also cancel
DISTINCTIVENESS = 3.0
MIN_DOMAIN_COUNT = 15
#: below this many corpus-derived identifiers a domain falls back to its declared nouns:
#: the corpus yields genuine domain vocabulary, but only where the domain has enough
#: files, and a four-word vocabulary cancels nothing
MIN_VOCABULARY = 40

#: hand-written domain nouns, used where the corpus yield is thin. Declared, recorded in
#: PROVENANCE, and never mixed silently: the output says which source each domain used.
DECLARED_NOUNS = {
    "finance": ("ledger_entries", "transactions", "invoices", "positions", "accruals",
                "settlements", "balances", "counterparties", "portfolios", "payables"),
    "biology": ("genome_reads", "samples", "sequences", "cultures", "assays",
                "chromosomes", "plasmids", "organisms", "enzymes", "taxa"),
    "gaming": ("player_scores", "sessions", "inventories", "matches", "spawns",
               "sprites", "hitboxes", "projectiles", "tilemaps", "powerups"),
    "physics": ("measurements", "trajectories", "detectors", "spectra", "lattices",
                "photons", "velocities", "wavelengths", "calorimeters", "kinematics"),
    "ecommerce": ("order_lines", "baskets", "shipments", "listings", "returns",
                  "checkouts", "coupons", "merchants", "storefronts", "restocks"),
}

#: an identifier must appear in this many *distinct files* of its domain. Without it,
#: ratio-ranking returns names concentrated in a single repository — `CThostFtdcOrderField`
#: for finance, `ALfloat` for physics — which are that repo's API, not the domain's
#: vocabulary.
MIN_DOMAIN_FILES = 8
RESERVOIR = 4000
MAX_FILE_CHARS = 1_000_000


class Reservoir:
    """Uniform sample of unbounded input, fixed memory (Vitter's algorithm R)."""

    def __init__(self, k: int, seed: int = 0):
        self.k, self.n, self.items = k, 0, []
        self.rng = np.random.default_rng(seed)

    def add(self, item) -> None:
        self.n += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = int(self.rng.integers(self.n))
            if j < self.k:
                self.items[j] = item


@dataclass
class Tables:
    files: int = 0
    parsed: int = 0
    skipped: int = 0
    oversize: int = 0
    chars: int = 0
    lines: int = 0
    occurrences: collections.Counter = field(default_factory=collections.Counter)
    files_with: collections.Counter = field(default_factory=collections.Counter)
    scope: dict = field(default_factory=lambda: collections.defaultdict(collections.Counter))
    depth: dict = field(default_factory=lambda: collections.defaultdict(collections.Counter))
    companions: dict = field(default_factory=lambda: collections.defaultdict(collections.Counter))
    axis: dict = field(default_factory=lambda: collections.defaultdict(collections.Counter))
    span_lines: dict = field(default_factory=dict)
    span_chars: dict = field(default_factory=dict)
    identifiers: dict = field(default_factory=lambda: collections.defaultdict(collections.Counter))
    identifier_df: dict = field(default_factory=lambda: collections.defaultdict(collections.Counter))
    literals: collections.Counter = field(default_factory=collections.Counter)
    int_values: collections.Counter = field(default_factory=collections.Counter)
    str_lengths: collections.Counter = field(default_factory=collections.Counter)
    padding: Reservoir = field(default_factory=lambda: Reservoir(RESERVOIR, 1))
    spans: dict = field(default_factory=dict)          # construct -> Reservoir of source
    domain_files: collections.Counter = field(default_factory=collections.Counter)

    def _res(self, store: dict, key: str, k: int = 600, seed: int = 2) -> Reservoir:
        if key not in store:
            store[key] = Reservoir(k, seed)
        return store[key]


def classify_domain(source: str) -> str | None:
    low = source.lower()
    scores = {dom: sum(low.count(s) for s in seeds)
              for dom, seeds in DOMAIN_SEEDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] < 3:
        return None
    # require the winner to be clearly ahead, or the file is simply generic code that
    # happens to mention one anchor
    runner_up = max((v for d, v in scores.items() if d != best), default=0)
    return best if scores[best] >= 2 * max(runner_up, 1) else None


def distinctive(identifiers: dict, top: int, df: dict | None = None) -> dict:
    """Per domain, the identifiers that are actually that domain's, by ratio.

    Ranking by raw frequency returns the same generic names in every domain, which is
    what the first calibration pass produced. The ratio of a name's count in its own
    domain to its count elsewhere is what separates `plasmid` from `AttributeError`.
    """
    totals = {d: max(sum(c.values()), 1) for d, c in identifiers.items()}
    out = {}
    for dom, counter in identifiers.items():
        scored = []
        files_here = (df or {}).get(dom, {})
        for name, n in counter.items():
            if n < MIN_DOMAIN_COUNT:
                continue
            if df is not None and files_here.get(name, 0) < MIN_DOMAIN_FILES:
                continue      # one repo is not a domain
            own = n / totals[dom]
            other = sum(identifiers[d].get(name, 0) / totals[d]
                        for d in identifiers if d != dom)
            ratio = own / (other + 1e-12)
            if ratio >= DISTINCTIVENESS:
                scored.append((name, n, ratio))
        scored.sort(key=lambda t: (-t[2], -t[1]))
        out[dom] = {name: {"count": n, "ratio": round(r, 2),
                           "files": files_here.get(name, 0)}
                    for name, n, r in scored[:top]}
    return out


def add_file(t: Tables, source: str, constructs: frozenset[str]) -> None:
    from .walk import line_offsets, span, walk

    t.files += 1
    if len(source) > MAX_FILE_CHARS:
        # a handful of generated modules are megabytes of literal data; they are not
        # representative of the population and they dominate the walk
        t.oversize += 1
        return
    try:
        # corpus files raise SyntaxWarning for bad escapes (`\d` outside a raw string);
        # they parse fine and the warning is noise at 300k files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        t.skipped += 1
        return
    t.parsed += 1
    lines = source.splitlines()
    offs = line_offsets(source)
    t.chars += len(source)
    t.lines += len(lines)

    domain = classify_domain(source)
    if domain:
        t.domain_files[domain] += 1
        here = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.isidentifier() \
                    and not node.id.startswith("_") and len(node.id) > 2:
                t.identifiers[domain][node.id] += 1
                here.add(node.id)
        for name in here:
            t.identifier_df[domain][name] += 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            v = node.value
            t.literals[type(v).__name__] += 1
            if isinstance(v, bool):
                pass
            elif isinstance(v, int) and 0 <= v < 1000:
                t.int_values[v] += 1
            elif isinstance(v, str) and len(v) < 200:
                t.str_lengths[len(v)] += 1

    seen_here = set()
    for occ in walk(tree, offs, constructs):
        c = occ.construct
        t.occurrences[c] += 1
        seen_here.add(c)
        t.scope[c][occ.scope] += 1
        t.depth[c][min(occ.depth, 8)] += 1
        for comp in occ.companions:
            t.companions[c][comp] += 1
        for axis, level in occ.axis:
            t.axis[c][f"{axis}:{level}"] += 1
        t._res(t.span_lines, c).add(occ.lines)
        t._res(t.span_chars, c).add(occ.chars)
    for c in seen_here:
        t.files_with[c] += 1

    _collect_padding(t, tree, lines)
    _collect_spans(t, tree, source, offs, constructs)


PAD_LINES = 3
PAD_CONSTRUCTS = (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.FunctionDef,
                  ast.ClassDef)


def _collect_padding(t: Tables, tree: ast.AST, lines: list[str]) -> None:
    """Corpus-typical code that *precedes* a construct — the grid's padding material.

    Taken only at module level and only where the block parses on its own, so padding
    never carries a fragment of a larger structure into a prompt.
    """
    for node in getattr(tree, "body", []):
        if not isinstance(node, PAD_CONSTRUCTS):
            continue
        lo = max(0, node.lineno - 1 - PAD_LINES)
        block = [ln for ln in lines[lo:node.lineno - 1] if ln.strip()
                 and not ln.startswith((" ", "\t"))]
        if not block:
            continue
        try:
            ast.parse("\n".join(block))
        except (SyntaxError, ValueError):
            continue
        t.padding.add(block)


SPANS_PER_FILE = 2


def _collect_spans(t: Tables, tree: ast.AST, source: str, offs: list[int],
                   constructs: frozenset[str]) -> None:
    """A sample of real construct source, for the tokeniser-based length bands.

    Capped per (file, construct): a single file offers hundreds of `Name` spans and they
    are not independent observations of the population, only of that file.
    """
    from .walk import span

    taken: collections.Counter = collections.Counter()
    for node in ast.walk(tree):
        name = type(node).__name__
        if name not in constructs or taken[name] >= SPANS_PER_FILE:
            continue
        res = t._res(t.spans, name, k=200, seed=3)
        if len(res.items) >= res.k and res.n > 5000:
            continue
        lo, hi = span(node, offs)
        if 0 < hi - lo < 4000:
            res.add(source[lo:hi])
            taken[name] += 1
