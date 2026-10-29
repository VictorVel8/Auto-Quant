"""Every DSL operator as data. The checker, compiler, random generator and the LLM-facing
documentation are all derived from this table, so they cannot disagree.

Missing data is null, never a number: every operator returns null where an input is null, where
a window is incomplete, or where a result is not finite. Division returns null when the absolute
denominator is below 1e-12.

Conventions for the implementations: they receive polars expressions for float/bool arguments
(columns that earlier steps already computed) and plain numbers for window/const arguments, and
return an expression. Time-series operators run over a ticker (`tix`), session operators over
(`tix`, `sidx`), cross-sectional operators over a timestamp (`tsi`) among universe members.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

ArgKind = Literal["float", "bool", "window", "const"]
Group = Literal["ts", "session", "cs", "elementwise", "logic"]

WINDOWS = (1, 2, 3, 5, 8, 13, 26, 39, 65, 130)  # 1 only for Ref, Delta, Pct
WINDOW_ONE_OK = frozenset({"Ref", "Delta", "Pct"})
MIN_WINDOW = {"Skew": 3, "Corr": 3, "Cov": 3, "TsZScore": 3}  # default 2
EPS = 1e-12

T = ["tix"]
S = ["tix", "sidx"]


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    group: Group
    args: tuple[ArgKind, ...]
    returns: Literal["float", "bool"]
    commutative: bool
    impl: Callable[..., pl.Expr]
    doc: str  # one line, shown to the LLM
    choices: tuple[tuple[float, ...] | None, ...] = ()  # allowed literals for "const" args
    infix: str | None = None  # how the operator was written, for documentation only


def div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(b.abs() >= EPS).then(a / b)


def _member(x: pl.Expr) -> pl.Expr:
    """Cross-sectional operators see only universe members."""
    return pl.when(pl.col("is_member")).then(x)


# --- time series ------------------------------------------------------------------------------
def _ref(x: pl.Expr, n: int) -> pl.Expr:
    return x.shift(n).over(T)


def _delta(x: pl.Expr, n: int) -> pl.Expr:
    return x - x.shift(n).over(T)


def _pct(x: pl.Expr, n: int) -> pl.Expr:
    return div(x, x.shift(n).over(T)) - 1


def _roll(kind: str) -> Callable[[pl.Expr, int], pl.Expr]:
    def f(x: pl.Expr, n: int) -> pl.Expr:
        match kind:
            case "mean":
                r = x.rolling_mean(n, min_samples=n)
            case "sum":
                r = x.rolling_sum(n, min_samples=n)
            case "std":
                r = x.rolling_std(n, min_samples=n)
            case "min":
                r = x.rolling_min(n, min_samples=n)
            case "max":
                r = x.rolling_max(n, min_samples=n)
            case _:
                r = x.rolling_skew(n, bias=False, min_samples=n)
        return r.over(T)

    return f


def _ts_zscore(x: pl.Expr, n: int) -> pl.Expr:
    mean = x.rolling_mean(n, min_samples=n).over(T)
    return div(x - mean, x.rolling_std(n, min_samples=n).over(T))


def ts_rank_array(x: np.ndarray, tix: np.ndarray, n: int, chunk: int = 100_000) -> np.ndarray:
    """Percentile of x_t within its trailing window of n values (itself included), per ticker.

    (count below + half the count equal) / n, in (0, 1). Null where the window is incomplete or
    contains a gap. Windows are built from numpy sliding views, a chunk of rows at a time.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    size = len(x)
    out = np.full(size, np.nan)
    if size < n:
        return out
    starts = np.r_[0, np.flatnonzero(np.diff(tix) != 0) + 1]
    pos = np.arange(size) - np.repeat(starts, np.diff(np.r_[starts, size]))
    rows = np.flatnonzero((pos >= n - 1) & ~np.isnan(x))
    view = sliding_window_view(x, n)  # row i of the view is x[i : i + n]
    for lo in range(0, len(rows), chunk):
        r = rows[lo : lo + chunk]
        win = view[r - (n - 1)]
        cur = x[r][:, None]
        score = ((win < cur).sum(axis=1) + 0.5 * (win == cur).sum(axis=1)) / n
        out[r] = np.where(np.isnan(win).any(axis=1), np.nan, score)
    return out


def _ts_rank(x: pl.Expr, n: int) -> pl.Expr:
    def batch(s: pl.Series) -> pl.Series:
        arr = s.struct.field("x").cast(pl.Float64).to_numpy()
        tix = s.struct.field("tix").to_numpy()
        return pl.Series(ts_rank_array(arr, tix, n), dtype=pl.Float64)

    return pl.struct(x.alias("x"), pl.col("tix")).map_batches(batch, return_dtype=pl.Float64)


def _corr(x: pl.Expr, y: pl.Expr, n: int) -> pl.Expr:
    return pl.rolling_corr(x, y, window_size=n, min_samples=n).over(T)


def _cov(x: pl.Expr, y: pl.Expr, n: int) -> pl.Expr:
    return pl.rolling_cov(x, y, window_size=n, min_samples=n).over(T)


def _decay(x: pl.Expr, n: int) -> pl.Expr:
    """Linearly weighted trailing mean, newest value weighing n and the oldest 1.

    Computed from rolling sums, because polars' weighted rolling mean panics on nulls. With g the
    bar number, the weight of x_s in the window ending at t is n - t + s, so the weighted sum is
    (n - t) * sum(x) + sum(g * x).
    """
    g = pl.int_range(pl.len()).over(T).cast(pl.Float64)
    sx = x.rolling_sum(n, min_samples=n).over(T)
    sgx = (g * x).rolling_sum(n, min_samples=n).over(T)
    return ((n - g) * sx + sgx) / (n * (n + 1) / 2)


def _ema(x: pl.Expr, n: int) -> pl.Expr:
    return pl.when(x.is_not_null()).then(x.ewm_mean(span=n, adjust=False, min_samples=n).over(T))


def _slope(x: pl.Expr, n: int) -> pl.Expr:
    """OLS slope of x on bar number over the window, from rolling sums.

    slope = (sum(g * x) - mean(g) * sum(x)) / sum((g - mean(g))^2), with g the bar number.
    """
    g = pl.int_range(pl.len()).over(T).cast(pl.Float64)
    sgx = (g * x).rolling_sum(n, min_samples=n).over(T)
    sx = x.rolling_sum(n, min_samples=n).over(T)
    gbar = g - (n - 1) / 2  # mean bar number of the window ending at this row
    return (sgx - gbar * sx) / (n * (n * n - 1) / 12)


# --- session ----------------------------------------------------------------------------------
def _sess_sum(x: pl.Expr) -> pl.Expr:
    return pl.when(x.is_not_null()).then(x.cum_sum().over(S))


def _sess_mean(x: pl.Expr) -> pl.Expr:
    return pl.when(x.is_not_null()).then(x.cum_sum().over(S) / x.cum_count().over(S))


# --- cross section ----------------------------------------------------------------------------
def _rank(x: pl.Expr) -> pl.Expr:
    m = _member(x)
    return m.rank("average").over("tsi") / m.count().over("tsi")


def _zscore(x: pl.Expr) -> pl.Expr:
    m = _member(x)
    return div(m - m.mean().over("tsi"), m.std().over("tsi"))


def _demean(x: pl.Expr) -> pl.Expr:
    m = _member(x)
    return m - m.mean().over("tsi")


def _winsor(x: pl.Expr, q: float) -> pl.Expr:
    m = _member(x)
    lo = m.quantile(q, interpolation="linear").over("tsi")
    hi = m.quantile(1 - q, interpolation="linear").over("tsi")
    return pl.when(m < lo).then(lo).when(m > hi).then(hi).otherwise(m)


def _sector_neutral(x: pl.Expr) -> pl.Expr:
    m = _member(x)
    return m - m.mean().over(["tsi", "sector"])


# --- element-wise -----------------------------------------------------------------------------
def _slog(x: pl.Expr) -> pl.Expr:
    return x.sign() * (1 + x.abs()).log()


def _signed_power(x: pl.Expr, p: float) -> pl.Expr:
    return x.sign() * x.abs().pow(p)


def _both(a: pl.Expr, b: pl.Expr, value: pl.Expr) -> pl.Expr:
    return pl.when(a.is_not_null() & b.is_not_null()).then(value)


def _if(c: pl.Expr, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(c.is_null()).then(None).when(c).then(a).otherwise(b)


def _abs(x: pl.Expr) -> pl.Expr:
    return x.abs()


def _sign(x: pl.Expr) -> pl.Expr:
    return x.sign()


def _greater(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _both(a, b, pl.max_horizontal(a, b))


def _less(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _both(a, b, pl.min_horizontal(a, b))


def _clip(x: pl.Expr, lo: float, hi: float) -> pl.Expr:
    return x.clip(lo, hi)


def _and(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _both(a, b, a & b)


def _or(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _both(a, b, a | b)


def _not(a: pl.Expr) -> pl.Expr:
    return ~a


def _add(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a + b


def _sub(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a - b


def _mul(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a * b


def _neg(a: pl.Expr) -> pl.Expr:
    return -a


def _gt(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _both(a, b, a > b)


def _ge(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _both(a, b, a >= b)


def _lt(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _both(a, b, a < b)


def _le(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return _both(a, b, a <= b)


def _op(
    name: str,
    group: Group,
    args: tuple[ArgKind, ...],
    returns: Literal["float", "bool"],
    impl: Callable[..., pl.Expr],
    doc: str,
    *,
    commutative: bool = False,
    choices: tuple[tuple[float, ...] | None, ...] = (),
    infix: str | None = None,
) -> OperatorSpec:
    return OperatorSpec(name, group, args, returns, commutative, impl, doc, choices, infix)


_F, _B, _W, _C = "float", "bool", "window", "const"

_SPECS = (
    _op("Ref", "ts", (_F, _W), "float", _ref, "Value n bars ago."),
    _op("Delta", "ts", (_F, _W), "float", _delta, "x minus its value n bars ago."),
    _op("Pct", "ts", (_F, _W), "float", _pct, "Fractional change over n bars: x / Ref(x, n) - 1."),
    _op("Mean", "ts", (_F, _W), "float", _roll("mean"), "Trailing mean over n bars."),
    _op("Sum", "ts", (_F, _W), "float", _roll("sum"), "Trailing sum over n bars."),
    _op("Std", "ts", (_F, _W), "float", _roll("std"), "Trailing standard deviation over n bars."),
    _op("Min", "ts", (_F, _W), "float", _roll("min"), "Trailing minimum over n bars."),
    _op("Max", "ts", (_F, _W), "float", _roll("max"), "Trailing maximum over n bars."),
    _op("Skew", "ts", (_F, _W), "float", _roll("skew"), "Trailing skewness over n bars."),
    _op("TsRank", "ts", (_F, _W), "float", _ts_rank, "Percentile of x within its trailing n bars."),
    _op("TsZScore", "ts", (_F, _W), "float", _ts_zscore, "(x - trailing mean) / trailing std."),
    _op(
        "Corr",
        "ts",
        (_F, _F, _W),
        "float",
        _corr,
        "Trailing correlation of x and y.",
        commutative=True,
    ),
    _op(
        "Cov",
        "ts",
        (_F, _F, _W),
        "float",
        _cov,
        "Trailing covariance of x and y.",
        commutative=True,
    ),
    _op(
        "Decay",
        "ts",
        (_F, _W),
        "float",
        _decay,
        "Linearly weighted trailing mean, newest weighs most.",
    ),
    _op("EMA", "ts", (_F, _W), "float", _ema, "Exponential moving average with span n."),
    _op("Slope", "ts", (_F, _W), "float", _slope, "OLS slope of x against bar number over n bars."),
    _op("SessMean", "session", (_F,), "float", _sess_mean, "Mean of x so far in the session."),
    _op("SessSum", "session", (_F,), "float", _sess_sum, "Sum of x so far in the session."),
    _op(
        "Rank",
        "cs",
        (_F,),
        "float",
        _rank,
        "Cross-sectional percentile rank among universe members.",
    ),
    _op("ZScore", "cs", (_F,), "float", _zscore, "Cross-sectional z-score among members."),
    _op("Demean", "cs", (_F,), "float", _demean, "x minus its cross-sectional mean among members."),
    _op(
        "Winsor",
        "cs",
        (_F, _C),
        "float",
        _winsor,
        "Clip x to its cross-sectional q and 1-q quantiles; q is 0.01 or 0.05.",
        choices=(None, (0.01, 0.05)),
    ),
    _op(
        "SectorNeutral",
        "cs",
        (_F,),
        "float",
        _sector_neutral,
        "x minus its sector mean among members.",
    ),
    _op("Abs", "elementwise", (_F,), "float", _abs, "Absolute value."),
    _op("Sign", "elementwise", (_F,), "float", _sign, "Sign: -1, 0 or 1."),
    _op("SLog", "elementwise", (_F,), "float", _slog, "sign(x) * log(1 + abs(x))."),
    _op(
        "SignedPower",
        "elementwise",
        (_F, _C),
        "float",
        _signed_power,
        "sign(x) * abs(x)^p; p is 0.5 or 2.",
        choices=(None, (0.5, 2.0)),
    ),
    _op(
        "Greater",
        "elementwise",
        (_F, _F),
        "float",
        _greater,
        "Element-wise maximum.",
        commutative=True,
    ),
    _op("Less", "elementwise", (_F, _F), "float", _less, "Element-wise minimum.", commutative=True),
    _op(
        "Clip",
        "elementwise",
        (_F, _C, _C),
        "float",
        _clip,
        "Clip x to the constant bounds [lo, hi].",
    ),
    _op("If", "logic", (_B, _F, _F), "float", _if, "a where c is true, else b."),
    _op("And", "logic", (_B, _B), "bool", _and, "Logical and.", commutative=True),
    _op("Or", "logic", (_B, _B), "bool", _or, "Logical or.", commutative=True),
    _op("Not", "logic", (_B,), "bool", _not, "Logical not."),
    # Infix operators. They are written as + - * / > >= < <= and unary minus; the canonical form
    # and the compiler use these names.
    _op("Add", "elementwise", (_F, _F), "float", _add, "x + y.", commutative=True, infix="+"),
    _op("Sub", "elementwise", (_F, _F), "float", _sub, "x - y.", infix="-"),
    _op("Mul", "elementwise", (_F, _F), "float", _mul, "x * y.", commutative=True, infix="*"),
    _op(
        "Div", "elementwise", (_F, _F), "float", div, "x / y; null when abs(y) < 1e-12.", infix="/"
    ),
    _op("Neg", "elementwise", (_F,), "float", _neg, "-x.", infix="-"),
    _op("Gt", "elementwise", (_F, _F), "bool", _gt, "x > y.", infix=">"),
    _op("Ge", "elementwise", (_F, _F), "bool", _ge, "x >= y.", infix=">="),
    _op("Lt", "elementwise", (_F, _F), "bool", _lt, "x < y.", infix="<"),
    _op("Le", "elementwise", (_F, _F), "bool", _le, "x <= y.", infix="<="),
)

REGISTRY: dict[str, OperatorSpec] = {s.name: s for s in _SPECS}

FIELD_DOCS = {
    "open": "Bar open, forward-adjusted.",
    "high": "Bar high, forward-adjusted.",
    "low": "Bar low, forward-adjusted.",
    "close": "Bar close, forward-adjusted.",
    "volume": "Shares traded in the bar.",
    "trades": "Number of trades in the bar.",
    "vwap": "Volume-weighted typical price of the bar.",
    "ret": "Close-to-close return versus the previous bar (includes the overnight gap at bar 0).",
    "dayopen": "Open of the session's first bar (known from bar 0 onwards).",
    "prevclose": "Close of the previous session's last bar.",
    "bar": "Bar of the day: 0 is 09:30-10:00, 12 is 15:30-16:00 (known in advance).",
}
FIELDS = tuple(FIELD_DOCS)
# DSL field -> panel column, where they differ (bar is stored as an integer in the panel)
FIELD_COLUMN = {"bar": "barnum"}


def render_docs() -> str:
    """The operator and field reference shown to the LLM, generated from the registry."""
    lines = ["Fields:", *(f"  ${n}: {d}" for n, d in FIELD_DOCS.items()), "", "Operators:"]
    for s in REGISTRY.values():
        if s.infix is not None:
            continue
        sig = ", ".join({"float": "x", "bool": "c", "window": "n", "const": "k"}[a] for a in s.args)
        lines.append(f"  {s.name}({sig}) -> {s.returns}: {s.doc}")
    lines += [
        "",
        f"Windows n: {', '.join(str(w) for w in WINDOWS)} (1 only for Ref, Delta, Pct).",
        "Infix: + - * / and comparisons > >= < <=; unary minus. Comparisons give bool.",
        "Missing data is null and propagates. Limits: depth 6, 30 nodes, 3 numeric constants.",
    ]
    return "\n".join(lines)
