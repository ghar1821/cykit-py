"""Axis scales for cytometry plots (linear, log, asinh, logicle/biex).

A :class:`Scale` maps *data* values to monotonically increasing *display*
coordinates and back, and knows where to put axis ticks. The viewer works
entirely in display coordinates, so switching between an arcsinh axis and a
logicle ("biex") axis is just swapping the scale object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "AsinhScale",
    "LinearScale",
    "LogScale",
    "LogicleScale",
    "PretransformedScale",
    "Scale",
    "get_scale",
]

_LN10 = np.log(10.0)


#: How much of the span to add beyond each end of an axis, by default. A
#: population sitting against the edge of the data is the normal case in
#: cytometry -- a negative peak at zero, a saturated one at the top of scale --
#: and a tight frame cuts it in half. Matches matplotlib's own axis margin.
AXIS_MARGIN = 0.05


def pad_range(lo: float, hi: float, frac: float = AXIS_MARGIN) -> tuple[float, float]:
    """Widen a range so points on the limit are not clipped by the frame.

    Parameters
    ----------
    lo, hi
        The range to pad. A degenerate or non-finite range becomes a unit-wide
        one around ``lo``, because an axis has to have some extent.
    frac
        Fraction of the span to add at each end; :data:`AXIS_MARGIN` by default.

    Returns
    -------
    tuple of float
        The padded ``(lo, hi)``.
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = lo - 0.5, lo + 0.5
    pad = frac * (hi - lo)
    return (lo - pad, hi + pad)


@dataclass
class Ticks:
    """Where an axis's ticks go, in *display* coordinates.

    Parameters
    ----------
    major
        Positions of the labelled ticks, ascending.
    labels
        One label per entry of ``major``, e.g. ``"0"``, ``"10³"``, ``"-10²"``.
    minor
        Positions of the unlabelled intermediate ticks.
    """

    major: np.ndarray
    labels: list[str]
    minor: np.ndarray


#: Roughly how many labelled ticks a linear axis aims for. The step is rounded
#: to 1/2/5 x 10^k, so the count that comes out is between about four and nine.
_LINEAR_TICKS = 6


def _nice_step(span: float, target: int) -> tuple[float, int]:
    """A round step dividing ``span`` into about ``target`` intervals.

    Parameters
    ----------
    span
        Width of the range to be ticked; must be positive and finite.
    target
        Number of intervals to aim for.

    Returns
    -------
    tuple
        ``(step, divisions)`` -- the step, one of 1, 2 or 5 times a power of
        ten, and how many sub-steps of it a minor tick should split it into so
        that the minor ticks also land on round numbers.
    """
    raw = span / max(target, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for mult, divisions in ((1.0, 5), (2.0, 4), (5.0, 5), (10.0, 5)):
        if raw <= magnitude * mult * (1 + 1e-9):
            return magnitude * mult, divisions
    return magnitude * 10.0, 5  # pragma: no cover - unreachable, raw < 10*magnitude


def _multiples_within(step: float, lo: float, hi: float) -> np.ndarray:
    """Every multiple of ``step`` inside ``[lo, hi]``, inclusive.

    Built from integer multiples rather than :func:`numpy.arange` so that a
    tick cannot drift off a round number, and so that none can fall outside
    the range and be drawn beyond the edge of the plot.

    Parameters
    ----------
    step
        Spacing between ticks; must be positive.
    lo, hi
        The range to fill, in the same units as ``step``.

    Returns
    -------
    ndarray
        The multiples, ascending; empty when none fit.
    """
    tol = abs(step) * 1e-9
    first = int(np.ceil((lo - tol) / step))
    last = int(np.floor((hi + tol) / step))
    if last < first:
        return np.asarray([], dtype=float)
    return np.arange(first, last + 1, dtype=float) * step


def _linear_label(value: float, step: float) -> str:
    """Format a linear tick with just enough decimals for ``step``.

    Plain decimal throughout: an instrument range of 262144 reads better as
    ``250000`` than as ``2.5e+05``, and ``%g`` would give the latter.
    """
    decimals = max(0, int(-np.floor(np.log10(step)))) if step > 0 else 0
    value = round(float(value), decimals + 2)
    if value == 0.0:
        return "0"  # never "-0", which is what a rounded-down negative gives
    return f"{value:.{decimals}f}"


def _decade_label(exponent: int, negative: bool = False) -> str:
    sup = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{'-' if negative else ''}10{sup}"


class Scale:
    """An invertible, monotonically increasing map from data to display units.

    Subclasses implement :meth:`forward`, :meth:`inverse` and :meth:`ticks`.
    ``name`` identifies the scale in the viewer and in stored parameters.
    """

    name = "scale"

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Map data values to display coordinates.

        Parameters
        ----------
        x
            Data values, any shape.

        Returns
        -------
        ndarray
            Display coordinates, same shape as ``x``.
        """
        raise NotImplementedError

    def inverse(self, s: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Map display coordinates back to data values.

        Parameters
        ----------
        s
            Display coordinates, any shape.

        Returns
        -------
        ndarray
            Data values, same shape as ``s``.
        """
        raise NotImplementedError

    def __call__(self, x):
        """Alias for :meth:`forward`, coercing ``x`` to a float array."""
        return self.forward(np.asarray(x, dtype=float))

    def ticks(self, lo: float, hi: float) -> Ticks:  # pragma: no cover - abstract
        """Choose axis ticks for a display range.

        Parameters
        ----------
        lo, hi
            The visible range, in display coordinates.

        Returns
        -------
        Ticks
            Positions and labels, all within ``[lo, hi]``.
        """
        raise NotImplementedError

    def to_raw(self, s: np.ndarray) -> np.ndarray:
        """Display coordinates to the units the tick labels are written in.

        Usually the same as :meth:`inverse`. It differs for
        :class:`PretransformedScale`, whose ``inverse`` is the identity -- the
        values are already transformed -- while its ticks are labelled in the
        raw units they came from. Anything that puts an axis position in front
        of a person, or takes one from them, wants this rather than
        :meth:`inverse`.

        Parameters
        ----------
        s
            Display coordinates, any shape.

        Returns
        -------
        ndarray
            The same positions in tick-label units.
        """
        return self.inverse(s)

    def from_raw(self, x: np.ndarray) -> np.ndarray:
        """Tick-label units back to display coordinates. Inverts :meth:`to_raw`.

        Parameters
        ----------
        x
            Positions in tick-label units, any shape.

        Returns
        -------
        ndarray
            Display coordinates, same shape as ``x``.
        """
        return self.forward(x)

    def limits(
        self, x: np.ndarray, quantiles: tuple[float, float] = (0.0, 1.0)
    ) -> tuple[float, float]:
        """Display-coordinate limits covering ``x``, padded slightly.

        Parameters
        ----------
        x
            Data values to span. Non-finite entries are ignored.
        quantiles
            Lower and upper quantiles of ``x`` to bound the range by, before
            padding. The default spans the full extent; something like
            ``(0.001, 0.999)`` keeps a handful of extreme events from
            flattening the plot.

        Returns
        -------
        tuple of float
            ``(lo, hi)`` in display coordinates, widened by :data:`AXIS_MARGIN`
            at each end.
            Degenerate input (all equal, or empty) still yields ``hi > lo``.
        """
        s = self.forward(np.asarray(x, dtype=float))
        s = s[np.isfinite(s)]
        if s.size == 0:
            return (0.0, 1.0)
        lo = float(np.quantile(s, quantiles[0])) if quantiles[0] > 0 else float(s.min())
        hi = float(np.quantile(s, quantiles[1])) if quantiles[1] < 1 else float(s.max())
        return pad_range(lo, hi)


# --------------------------------------------------------------------------
# linear / log
# --------------------------------------------------------------------------
@dataclass
class LinearScale(Scale):
    """The identity: display coordinates are the data values themselves."""

    name: str = "linear"

    def forward(self, x):
        """Data values to display coordinates; the identity."""
        return np.asarray(x, dtype=float)

    def inverse(self, s):
        """Display coordinates to data values; the identity."""
        return np.asarray(s, dtype=float)

    def ticks(self, lo, hi):
        """Round ticks at a 1/2/5 x 10^k step, about six of them, all within the range."""
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return Ticks(np.array([float(lo)]), [_linear_label(lo, 1.0)], np.array([]))
        step, divisions = _nice_step(span, _LINEAR_TICKS)
        major = _multiples_within(step, lo, hi)
        minor = _multiples_within(step / divisions, lo, hi)
        # Drop the sub-steps that land on a labelled tick, so the frame does
        # not draw a short tick underneath every long one.
        if minor.size:
            on_major = np.isclose(minor[:, None], major[None, :], rtol=0.0, atol=step * 1e-6)
            minor = minor[~on_major.any(axis=1)] if major.size else minor
        return Ticks(major, [_linear_label(v, step) for v in major], minor)


@dataclass
class LogScale(Scale):
    """log10 scale; non-positive values are clamped to ``floor``."""

    floor: float = 1.0
    name: str = "log"

    def forward(self, x):
        """``log10(x)``, with values at or below ``floor`` clamped to it."""
        x = np.asarray(x, dtype=float)
        return np.log10(np.maximum(x, self.floor))

    def inverse(self, s):
        """``10 ** s``."""
        return np.power(10.0, np.asarray(s, dtype=float))

    def ticks(self, lo, hi):
        """A labelled tick per decade, with unlabelled 2..9 x 10^k between."""
        lo_e, hi_e = int(np.floor(lo)), int(np.ceil(hi))
        major, labels, minor = [], [], []
        for e in range(lo_e, hi_e + 1):
            if lo <= e <= hi:
                major.append(float(e))
                labels.append(_decade_label(e))
            for m in range(2, 10):
                v = e + np.log10(m)
                if lo <= v <= hi:
                    minor.append(v)
        return Ticks(np.asarray(major), labels, np.asarray(minor))


# --------------------------------------------------------------------------
# arcsinh
# --------------------------------------------------------------------------
@dataclass
class AsinhScale(Scale):
    """``asinh(x / cofactor)``: linear near zero, logarithmic far from it.

    Parameters
    ----------
    cofactor
        Width of the linear region. Roughly, values well inside ``+/-
        cofactor`` are scaled linearly and those outside logarithmically.
        Typical values: 5 for mass cytometry, 150 for flow, ~3000 for spectral.
    name
        Identifier for the scale; leave as the default.
    """

    cofactor: float = 5.0
    name: str = "asinh"

    def forward(self, x):
        """``asinh(x / cofactor)``."""
        return np.arcsinh(np.asarray(x, dtype=float) / self.cofactor)

    def inverse(self, s):
        """``sinh(s) * cofactor``."""
        return np.sinh(np.asarray(s, dtype=float)) * self.cofactor

    def ticks(self, lo, hi):
        """Decade ticks in data units, mirrored either side of zero."""
        d_lo, d_hi = self.inverse(np.array([lo, hi]))
        return _biex_ticks(self.forward, d_lo, d_hi, lo, hi)


# --------------------------------------------------------------------------
# logicle / biexponential
# --------------------------------------------------------------------------
def _solve_d(b: float, w: float) -> float:
    """Root of ``2*log(d/b) + w*(b+d)`` on ``(0, b)`` (Moore & Parks 2012)."""
    if w == 0:
        return b
    lo, hi = 1e-300, b
    f = lambda d: 2.0 * (np.log(d) - np.log(b)) + w * (b + d)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 2 * np.finfo(float).eps * b:
            break
    return 0.5 * (lo + hi)


@dataclass
class LogicleScale(Scale):
    """The logicle (a.k.a. biexponential / "biex") display scale.

    Parameters follow Moore & Parks (2012). Display coordinates run from 0 to
    1 across the full scale.

    Parameters
    ----------
    T
        Top of scale, i.e. the largest data value the axis should reach.
    W
        Number of decades of linearisation around zero. Larger ``W`` gives
        more room to negative / near-zero events.
    M
        Total number of decades the display covers.
    A
        Additional decades of negative data to show below zero.
    lut_size
        Number of samples in the lookup table used for the forward direction,
        which has no closed form. Larger is finer but slower to build.
    name
        Identifier for the scale; leave as the default.
    """

    T: float = 262144.0
    W: float = 0.5
    M: float = 4.5
    A: float = 0.0
    lut_size: int = 1 << 17
    name: str = "logicle"
    _lut: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.M <= 0:
            raise ValueError("M must be positive")
        if self.W < 0 or 2 * self.W > self.M:
            raise ValueError("W must satisfy 0 <= 2W <= M")
        M, A, W, T = float(self.M), float(self.A), float(self.W), float(self.T)
        self.w = W / (M + A)
        self.x2 = A / (M + A)
        self.x1 = self.x2 + self.w
        self.x0 = self.x2 + 2 * self.w
        self.b = (M + A) * _LN10
        self.d = _solve_d(self.b, self.w)
        c_a = np.exp(self.x0 * (self.b + self.d))
        mf_a = np.exp(self.b * self.x1) - c_a / np.exp(self.d * self.x1)
        self.a = T / ((np.exp(self.b) - mf_a) - c_a / np.exp(self.d))
        self.c = c_a * self.a
        self.f = mf_a * self.a

    # -- core biexponential -------------------------------------------------
    def inverse(self, s):
        """Display coordinate (0..1) -> data value."""
        s = np.asarray(s, dtype=float)
        neg = s < self.x1
        s_r = np.where(neg, 2 * self.x1 - s, s)
        val = self.a * np.exp(self.b * s_r) - self.c * np.exp(-self.d * s_r) - self.f
        return np.where(neg, -val, val)

    def _lookup(self):
        """Monotone lookup table for the forward (data -> display) direction.

        The biexponential has no closed-form inverse, so we tabulate it. The
        table is antisymmetric about ``x1``, so spanning ``[2*x1 - 1, 1]``
        covers exactly ``[-T, T]``; the extra 5% keeps axis padding in range.
        """
        if self._lut is None:
            s = np.linspace(2 * self.x1 - 1.05, 1.05, int(self.lut_size))
            self._lut = (self.inverse(s), s)
        return self._lut

    def forward(self, x):
        """Data value -> display coordinate (0..1)."""
        data_grid, s_grid = self._lookup()
        x = np.asarray(x, dtype=float)
        return np.interp(x, data_grid, s_grid, left=s_grid[0], right=s_grid[-1])

    def ticks(self, lo, hi):
        """Decade ticks in data units, mirrored either side of zero."""
        d_lo, d_hi = self.inverse(np.array([lo, hi]))
        return _biex_ticks(self.forward, float(d_lo), float(d_hi), lo, hi)

    # -- parameter estimation ----------------------------------------------
    @classmethod
    def from_data(
        cls,
        x: np.ndarray,
        *,
        T: float | None = None,
        M: float = 4.5,
        A: float = 0.0,
        quantile: float = 0.05,
        **kwargs,
    ) -> LogicleScale:
        """Pick ``T`` and ``W`` from data, the way FlowJo and flowCore do.

        Parameters
        ----------
        x
            Values for one channel. Non-finite entries are ignored; an empty
            array falls back to the class defaults.
        T
            Top of scale. Defaults to the maximum of ``x``.
        M
            Total number of decades the scale covers.
        A
            Additional decades of negative data shown below zero.
        quantile
            Quantile of the *negative* values taken as the spread of the
            negative population. ``W`` is then set so the linear region just
            covers it, and clamped to ``[0, M/2]``. Channels with fewer than
            ten negative events get a narrow ``W = 0.1``.
        **kwargs
            Passed to the constructor, e.g. ``lut_size``.

        Returns
        -------
        LogicleScale
            A scale fitted to ``x``.
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return cls(T=T or 262144.0, W=0.5, M=M, A=A, **kwargs)
        if T is None:
            T = float(np.max(x))
            T = 262144.0 if not np.isfinite(T) or T <= 0 else max(T, 10.0)
        neg = x[x < 0]
        if neg.size < 10:
            W = 0.1
        else:
            r = float(np.quantile(neg, quantile))
            if r >= 0 or not np.isfinite(r):
                W = 0.1
            else:
                W = (M - np.log10(T / abs(r))) / 2.0
        W = float(np.clip(W, 0.0, M / 2.0 - 1e-6))
        return cls(T=float(T), W=W, M=M, A=A, **kwargs)


@dataclass
class PretransformedScale(Scale):
    """For values that have *already* been transformed, e.g. an asinh layer.

    ``forward``/``inverse`` are the identity — the stored values are the
    display coordinates. ``inner`` is only consulted to work out where the
    decade ticks of the *original* units land, so an arcsinh-transformed
    channel still gets an axis labelled 0, 10², 10³, ... in raw units.

    Parameters
    ----------
    inner
        The transform that was already applied to the values. Consulted only
        for tick placement, never to change the data.
    name
        Identifier for the scale; leave as the default.
    """

    inner: Scale
    name: str = "pretransformed"

    def forward(self, x):
        """The identity: the stored values are already display coordinates."""
        return np.asarray(x, dtype=float)

    def inverse(self, s):
        """The identity. Use ``self.inner.inverse`` to reach raw units."""
        return np.asarray(s, dtype=float)

    def ticks(self, lo, hi):
        """Decade ticks of ``inner``'s *raw* units, mirrored around zero."""
        d_lo, d_hi = self.inner.inverse(np.array([lo, hi]))
        return _biex_ticks(self.inner.forward, float(d_lo), float(d_hi), lo, hi)

    def to_raw(self, s):
        """Display coordinates to the original units, through ``inner``."""
        return self.inner.inverse(np.asarray(s, dtype=float))

    def from_raw(self, x):
        """The original units back to display coordinates, through ``inner``."""
        return self.inner.forward(np.asarray(x, dtype=float))


def _biex_ticks(forward, d_lo: float, d_hi: float, lo: float, hi: float) -> Ticks:
    """Decade ticks (and their negative mirrors) for a biexponential axis."""
    major_d: list[float] = [0.0]
    labels: list[str] = ["0"]
    minor_d: list[float] = []
    top = max(abs(d_lo), abs(d_hi), 10.0)
    max_e = int(np.ceil(np.log10(top)))
    for e in range(max_e + 1):
        for sign in (1, -1):
            v = sign * 10.0**e
            if d_lo <= v <= d_hi:
                major_d.append(v)
                labels.append(_decade_label(e, negative=sign < 0))
            for m in range(2, 10):
                mv = sign * m * 10.0 ** (e - 1)
                if d_lo <= mv <= d_hi:
                    minor_d.append(mv)
    major = np.asarray(forward(np.asarray(major_d)))
    minor = np.asarray(forward(np.asarray(minor_d))) if minor_d else np.asarray([])
    keep = (major >= lo) & (major <= hi)
    order = np.argsort(major[keep])
    major = major[keep][order]
    labels = [labels[i] for i in np.flatnonzero(keep)[order]]
    if minor.size:
        minor = np.sort(minor[(minor >= lo) & (minor <= hi)])
        minor = _thin(minor, 0.006 * (hi - lo))
    major, labels = _thin_labelled(major, labels, 0.03 * (hi - lo))
    return Ticks(major, labels, minor)


def _thin(positions: np.ndarray, min_gap: float) -> np.ndarray:
    """Drop ticks that would sit on top of their neighbour."""
    kept: list[float] = []
    for p in positions:
        if not kept or p - kept[-1] >= min_gap:
            kept.append(float(p))
    return np.asarray(kept)


def _thin_labelled(
    positions: np.ndarray, labels: list[str], min_gap: float
) -> tuple[np.ndarray, list[str]]:
    """Thin labelled ticks outward from zero, which is always kept.

    Decades crowd together around the linear region of a biex axis, so without
    this the labels overprint each other into an unreadable smear.
    """
    if positions.size == 0:
        return positions, labels
    zero = (
        int(np.argmin(np.abs(positions - positions[np.array(labels) == "0"][0])))
        if "0" in labels
        else 0
    )
    keep = [zero]
    last = positions[zero]
    for i in range(zero + 1, positions.size):
        if positions[i] - last >= min_gap:
            keep.append(i)
            last = positions[i]
    last = positions[zero]
    for i in range(zero - 1, -1, -1):
        if last - positions[i] >= min_gap:
            keep.append(i)
            last = positions[i]
    keep.sort()
    return positions[keep], [labels[i] for i in keep]


def get_scale(kind: str, x: np.ndarray | None = None, *, cofactor: float = 5.0, **kwargs) -> Scale:
    """Build a scale by name, fitting it to ``x`` where that helps.

    Parameters
    ----------
    kind
        One of ``"linear"``/``"lin"``, ``"log"``, ``"asinh"``/``"arcsinh"``, or
        ``"logicle"``/``"biex"``/``"biexponential"``.
    x
        Data to fit to. Only ``"logicle"`` uses it, via
        :meth:`LogicleScale.from_data`; without it you get the class defaults.
    cofactor
        Linear width for ``"asinh"``; ignored by the other kinds.
    **kwargs
        Passed to the chosen scale's constructor (or to ``from_data`` for
        logicle), e.g. ``floor`` for log or ``M`` for logicle.

    Returns
    -------
    Scale
        The requested scale.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the names above.
    """
    kind = kind.lower()
    if kind in ("linear", "lin"):
        return LinearScale()
    if kind == "log":
        return LogScale(**kwargs)
    if kind in ("asinh", "arcsinh"):
        return AsinhScale(cofactor=cofactor)
    if kind in ("logicle", "biex", "biexponential"):
        if x is None:
            return LogicleScale(**kwargs)
        return LogicleScale.from_data(x, **kwargs)
    raise ValueError(f"unknown scale {kind!r}")
