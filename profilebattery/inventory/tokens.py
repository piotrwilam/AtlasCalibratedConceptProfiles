"""Surface token per construct, and which constructs share one.

The B set needs "the construct's token present, the construct absent", so every
construct needs a declared surface token. Writing them out makes a fact visible that
the design's subtype list did not account for: **the surface token does not identify
the construct** for a large part of the inventory, and not only for symbols.

`for` belongs to For *and* to every comprehension. `if` belongs to If *and* to IfExp
*and* to a comprehension's condition. `import` belongs to Import and ImportFrom; `*`
to Starred and BinOp; `[` to Subscript and List; `{` to Dict, Set and two
comprehensions. Collisions are therefore *derived* from this table rather than
declared, and each one is a free other-role B subtype — a sibling construct sharing
the token is a far stronger control than the token in a comment, because the token is
doing live grammatical work in both.
"""

from __future__ import annotations

import builtins as _builtins
import collections

#: construct -> surface token. Keywords, then symbols; builtins add their own name.
NODE_TOKENS: dict[str, str] = {
    # compound statements
    "For": "for", "AsyncFor": "for", "While": "while", "If": "if", "With": "with",
    "AsyncWith": "with", "Try": "try", "TryStar": "except", "FunctionDef": "def",
    "AsyncFunctionDef": "def", "ClassDef": "class", "Match": "match",
    # simple statements
    "Assign": "=", "AugAssign": "+=", "AnnAssign": ":", "Return": "return",
    "Raise": "raise", "Assert": "assert", "Import": "import", "ImportFrom": "from",
    "Global": "global", "Nonlocal": "nonlocal", "Delete": "del", "Pass": "pass",
    "Break": "break", "Continue": "continue",
    # expressions
    "BinOp": "+", "Compare": "==", "BoolOp": "and", "UnaryOp": "not",
    "Subscript": "[", "Attribute": ".", "Call": "(", "IfExp": "if",
    "NamedExpr": ":=", "Lambda": "lambda", "Dict": "{", "Set": "{", "List": "[",
    "Tuple": ",", "Starred": "*", "Slice": ":", "Await": "await", "Yield": "yield",
    # comprehensions
    "ListComp": "for", "SetComp": "for", "DictComp": "for", "GeneratorExp": "for",
    # remaining 3.12 constructors
    "TypeAlias": "type", "YieldFrom": "yield", "JoinedStr": 'f"', "FormattedValue": "{",
    "MatchSequence": "[", "MatchMapping": "{", "MatchClass": "(", "MatchStar": "*",
    "MatchAs": "as", "MatchOr": "|", "ParamSpec": "**", "TypeVarTuple": "*",
}

#: Further surface tokens a construct carries. The *primary* token above is the one the
#: B set is built around ("token present, construct absent"); these are the rest of the
#: construct's surface, and they are what the keyword-completeness table needs — `elif`
#: and `else` are If's, `finally` is Try's, `in` and `is` are Compare's. Without them
#: seven of the 35 keywords look unmapped when they are simply not anyone's *primary*
#: token.
SECONDARY_TOKENS: dict[str, tuple[str, ...]] = {
    "If": ("elif", "else"),
    "IfExp": ("else",),
    "For": ("in", "else"), "AsyncFor": ("async", "in", "else"),
    "While": ("else",),
    "Try": ("except", "finally", "else"), "TryStar": ("except", "finally", "else"),
    "Compare": ("in", "is", "not"),
    "BoolOp": ("or",),
    "AsyncFunctionDef": ("async",), "AsyncWith": ("async", "as"),
    "With": ("as",),
    "Import": ("as",), "ImportFrom": ("import", "as"),
    "Raise": ("from",), "YieldFrom": ("from",),
    "MatchAs": ("as",), "Match": ("case",),
    "ListComp": ("in", "if", "async"), "SetComp": ("in", "if", "async"),
    "DictComp": ("in", "if", "async"), "GeneratorExp": ("in", "if", "async"),
    "Lambda": (), "FunctionDef": (), "ClassDef": (),
    "Await": (), "Yield": (),
}


def tokens_of(construct: str) -> tuple[str, ...]:
    """Every surface token of a construct, primary first."""
    primary = NODE_TOKENS.get(construct)
    rest = SECONDARY_TOKENS.get(construct, ())
    return ((primary,) if primary else ()) + tuple(t for t in rest if t != primary)


#: constructs with no distinctive surface token: any identifier is a `Name`, any literal
#: a `Constant`, any bare name in a type-parameter list a `TypeVar`. There is no token
#: whose presence signals them, so no B set can be built and the registered contrast is
#: "A against C only" (report addendum item 6).
NO_TOKEN = ("Name", "Constant", "TypeVar")

SITE_ADDITIONS = ("exit", "quit", "copyright", "credits", "license", "help")


def builtin_tokens() -> dict[str, str]:
    """A builtin's token is its own name (handout S1, read mechanically)."""
    return {n: n for n in dir(_builtins)
            if not n.startswith("_") and n not in SITE_ADDITIONS}


def all_tokens() -> dict[str, str]:
    return {**NODE_TOKENS, **builtin_tokens()}


def sharers() -> dict[str, tuple[str, ...]]:
    """token -> the constructs that share it, for tokens shared by more than one."""
    by_token = collections.defaultdict(list)
    for construct, token in all_tokens().items():
        by_token[token].append(construct)
    return {t: tuple(sorted(cs)) for t, cs in by_token.items() if len(cs) > 1}


def siblings(construct: str) -> tuple[str, ...]:
    """Other constructs sharing this construct's surface token."""
    tok = all_tokens().get(construct)
    if tok is None:
        return ()
    return tuple(c for c in sharers().get(tok, ()) if c != construct)
