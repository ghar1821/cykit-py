"""Choosing arcsinh cofactors by eye, in a napari window.

Unlike :mod:`cykit.viewer`, this module *does* transform -- but only for
display, in memory, and never back onto the AnnData. It is the interactive
counterpart of ``plot_biaxial(..., cofactor=...)``: you are handed numbers
rather than a layer, and applying them is your own explicit
:func:`~cykit.asinh_transform` call.

Three panels share the canvas: the two channels against each other, and a
distribution of each with its own slider. The sliders are log-spaced across a
range **you** supply, because there is no range that suits both mass cytometry
(~5) and spectral flow (~3000), and that judgement is not one the tool can make
for you.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager

import anndata as ad
import numpy as np

from ._util import layer_matrix, subsample_indices
from .density import Axes2D, density_curve, density_image
from .scales import AXIS_MARGIN, AsinhScale, PretransformedScale, pad_range
from .transforms import find_channel_name, get_fluor_channels

__all__ = [
    "CofactorWindow",
    "Cofactors",
    "current_transform_window",
    "open_napari_transform",
]

#: Raw columns kept in memory at once. Each is 4 MB at a million events, so a
#: handful is cheap and all 48 channels of a spectral panel is the whole matrix
#: again -- which is exactly what the cache exists to avoid.
CACHE_SIZE = 8

#: Display coordinate of the "knee": the raw value equal to the cofactor, where
#: the arcsinh stops being linear. ``asinh(c / c)`` is this whatever ``c`` is,
#: so the line never moves and the data slides under it.
KNEE = float(np.arcsinh(1.0))

#: Events plotted by default. The picture is drawn from this many; the numbers
#: in the status line are always computed from every event.
MAX_EVENTS = 200_000

#: Fewer negative events than this and there is nothing for a cofactor to sit
#: on.
MIN_NEGATIVES = 50

#: Gutter between panels, as a fraction of the panel side. Wide enough for the
#: axis labels of the panel to its left.
GUTTER = 0.45

#: Hint on the box that puts one cofactor on the whole panel.
_ALL_TOOLTIP = "Type a cofactor and press Enter to give every channel that value."

#: Widest axis margin the slider offers, as a percentage of the span. Every
#: point of it is canvas spent on empty space, so the travel stops where the
#: data would be squeezed into the middle third of the panel.
MAX_MARGIN_PERCENT = 50

#: Hint on the box beside each cofactor slider.
_BOX_TOOLTIP = (
    "Type an exact cofactor and press Enter. The slider follows, to the nearest "
    "step it has; the value kept is the one you typed."
)

#: Width of that box, in pixels. Wide enough for a six-figure cofactor and no
#: wider, so the slider keeps the rest of the row.
_BOX_WIDTH = 78

#: Steps the cofactor slider has across ``cofactor_range``, log-spaced. Over a
#: two-decade range that is about half a per cent per step, which is finer than
#: the judgement being made -- the box beside it is there for an exact number.
SLIDER_STEPS = 1000

#: Hint on that slider.
_MARGIN_TOOLTIP = (
    "How far past the data the axis runs, as a proportion of the span at each "
    "end. The span is in decades here, so a little goes a long way."
)

_CURVE = "#4c78a8"
_BAND = "#f58518"


class Cofactors(dict):
    """Cofactors chosen in a window, and the record of which ones you moved.

    A plain ``dict`` of var_name to cofactor, so it goes straight into
    :func:`~cykit.asinh_transform`. Its ``repr`` is a Python dict literal, one
    channel per line, so the cell output is the thing you paste into a notebook
    -- which is where a decision like this belongs, rather than in the memory of
    a kernel you will restart.

    Parameters
    ----------
    values
        Initial mapping of var_name to cofactor.
    layer
        Layer the cofactors were chosen against, kept so a notebook can say what
        they apply to.
    adjusted
        Channels that have been moved off their seed. Everything else is still
        sitting where the window put it.
    """

    def __init__(self, values=None, *, layer: str = "", adjusted: Sequence[str] = ()):
        super().__init__(values or {})
        self.layer = str(layer)
        self.adjusted: set[str] = set(adjusted)

    @property
    def n_adjusted(self) -> int:
        """How many channels have been moved off their seed."""
        return len(self.adjusted)

    @property
    def untouched(self) -> list[str]:
        """Channels still sitting on the value the window seeded them with."""
        return [k for k in self if k not in self.adjusted]

    def to_source(self, name: str = "COFACTORS") -> str:
        """The dict as an assignment, ready to paste into a notebook.

        Parameters
        ----------
        name
            Variable name to assign to.

        Returns
        -------
        str
            ``"COFACTORS = {...}"``, one channel per line.
        """
        return f"{name} = {self!r}"

    def __repr__(self) -> str:
        if not self:
            return "{}"
        rows = "\n".join(f"    {k!r}: {v:.6g}," for k, v in self.items())
        return "{\n" + rows + "\n}"


class _ChannelData:
    """One channel's cofactor-independent facts, computed once and kept."""

    __slots__ = ("buf", "median_neg", "n", "n_neg", "q05_neg", "q95_neg", "q_hi", "q_lo", "raw")

    def __init__(self, raw: np.ndarray, full: np.ndarray):
        self.raw = raw
        self.buf = np.empty_like(raw)
        self.n = int(full.size)
        # Every number reported comes from every event, even when the picture
        # is drawn from a sample of them.
        self.q_lo, self.q_hi = (float(v) for v in np.quantile(full, [0.001, 0.999]))
        neg = full[full < 0]
        self.n_neg = int(neg.size)
        if self.n_neg:
            self.q05_neg, self.q95_neg, self.median_neg = (
                float(v) for v in np.quantile(neg, [0.05, 0.95, 0.5])
            )
        else:
            self.q05_neg = self.q95_neg = self.median_neg = 0.0

    def display(self, cofactor: float) -> np.ndarray:
        """``asinh(raw / cofactor)``, written into this channel's own buffer.

        Parameters
        ----------
        cofactor
            Width of the linear region.

        Returns
        -------
        ndarray
            The buffer, reused between calls -- valid until the next one.
        """
        np.divide(self.raw, cofactor, out=self.buf)
        np.arcsinh(self.buf, out=self.buf)
        return self.buf

    def limits(self, cofactor: float, margin: float = AXIS_MARGIN) -> tuple[float, float]:
        """Axis range, from the raw quantiles rather than the transformed ones.

        Parameters
        ----------
        cofactor
            Width of the linear region.
        margin
            Proportion of the span to add beyond each end, as
            :func:`~cykit.scales.pad_range` takes it.

        Returns
        -------
        tuple
            ``(lo, hi)`` in display coordinates, padded as ``Scale.limits`` pads.

        Notes
        -----
        arcsinh is monotone, so the event at a given quantile is the same event
        whatever the cofactor: transforming the two stored quantiles gives
        exactly what re-percentiling a million transformed values would, and
        keeps the axis pinned to fixed raw values instead of shifting under the
        population as the slider moves.
        """
        lo, hi = np.arcsinh(np.array([self.q_lo, self.q_hi]) / cofactor)
        return pad_range(float(lo), float(hi), margin)


#: The log-slider class, built on first use because magicgui is a GUI import.
_LOG_SLIDER: type | None = None


def _log_slider_class() -> type:
    """``FloatLogSlider`` with its handle where the value actually is.

    magicgui converts a value to a slider position with ``int(pos)``, which
    truncates rather than rounds. Two things follow, both visible:

    * The handle settles a step below the value it is showing. Every redraw
      writes the cofactor back to the widget, so a drag to position 500 lands
      on 499, and the handle and the number beside it disagree from then on.
    * The top of the range is unreachable. The largest cofactor maps to the
      last position, truncates to the one before it, and the handle stops a
      step short of the right-hand end however far you drag.

    Rounding fixes both, and halves the worst-case error into the bargain: the
    position is the nearest one to the value rather than the next one down.
    """
    global _LOG_SLIDER
    if _LOG_SLIDER is None:
        from magicgui.widgets import FloatLogSlider

        class _RoundedLogSlider(FloatLogSlider):  # type: ignore[misc, valid-type]
            def _position_from_value(self, value: float) -> int:
                base = np.log(self.base)
                offset = np.log(self.min) / base
                pos = (np.log(value) / base - offset) / self._scale + self._min_pos
                return int(np.clip(round(float(pos)), self._min_pos, self._max_pos))

        _LOG_SLIDER = _RoundedLogSlider
    return _LOG_SLIDER


def _resolve_margin(margin: float) -> float:
    """Validate an axis margin, so a percentage typed as one is not 100x too wide."""
    value = float(margin)
    if not np.isfinite(value) or not 0.0 <= value <= MAX_MARGIN_PERCENT / 100.0:
        raise ValueError(
            f"margin must be a proportion between 0 and {MAX_MARGIN_PERCENT / 100.0:g}, "
            f"not {margin!r}"
        )
    return value


def _resolve_range(cofactor_range) -> tuple[float, float]:
    """The slider range, checked. See :func:`open_napari_transform`."""
    from .transforms import DEFAULT_COFACTOR

    try:
        lo, hi = (float(v) for v in cofactor_range)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cofactor_range must be a (lo, hi) pair, got {cofactor_range!r}. "
            f"Typical cofactors: {DEFAULT_COFACTOR}"
        ) from exc
    if not (np.isfinite([lo, hi]).all() and 0 < lo < hi):
        raise ValueError(
            f"cofactor_range must be finite with 0 < lo < hi, got ({lo}, {hi}). "
            f"Typical cofactors: {DEFAULT_COFACTOR}"
        )
    return lo, hi


def _resolve_names(adata: ad.AnnData, channels: Sequence[str] | None) -> list[str]:
    names = get_fluor_channels(adata) if channels is None else list(channels)
    return [str(adata.var_names[find_channel_name(adata, n)]) for n in names]


def _seed(adata, names, cofactor, lo, hi) -> dict[str, float]:
    """One cofactor per channel before anything has been touched.

    The geometric midpoint of the range, unless a seed was given. The midpoint
    makes no claim about the data: it is fixed entirely by an argument the
    caller had to supply, and it starts the slider mid-travel.
    """
    midpoint = float(np.sqrt(lo * hi))
    out: dict[str, float] = {}
    for name in names:
        if cofactor is None:
            out[name] = midpoint
            continue
        if not isinstance(cofactor, Mapping):
            out[name] = float(cofactor)
            continue
        j = find_channel_name(adata, name)
        chan = str(adata.var["channel"].iloc[j]) if "channel" in adata.var else name
        mark = str(adata.var["marker"].iloc[j]) if "marker" in adata.var else name
        for key in (name, mark, chan):
            if key in cofactor:
                out[name] = float(cofactor[key])
                break
        else:
            out[name] = float(cofactor.get("default", midpoint))

    outside = {k: v for k, v in out.items() if not lo <= v <= hi}
    if outside:
        listed = ", ".join(f"{k}={v:g}" for k, v in sorted(outside.items())[:6])
        raise ValueError(
            f"{len(outside)} seed cofactor(s) fall outside cofactor_range "
            f"({lo:g}, {hi:g}): {listed}. Widen the range, or drop them from the seed -- "
            "a slider that silently clamped them would misreport what you asked for."
        )
    return out


def _default_y(adata: ad.AnnData, names: Sequence[str]) -> str:
    """A scatter channel to plot against: a fixed frame the population moves in.

    Scatter is never arcsinh-transformed, so only one axis is live at a time and
    a change in the picture is attributable to the slider you moved.
    """
    var_names = list(adata.var_names)
    detectors = (
        [str(v) for v in adata.var["channel"]] if "channel" in adata.var else list(var_names)
    )
    for want in ("SSC-A", "FSC-A"):
        for name, detector in zip(var_names, detectors):
            if want in (detector, name):
                return str(name)
    if "kind" in adata.var:
        scatter = [n for n, k in zip(var_names, adata.var["kind"]) if k == "scatter"]
        if scatter:
            return str(scatter[0])
    # Mass cytometry has no scatter at all; fall back to another channel.
    return str(names[1] if len(names) > 1 else names[0])


class CofactorWindow:
    """A napari window for choosing arcsinh cofactors by eye.

    Three panels: the two channels against each other, and a distribution of
    each with its own log-spaced slider. Everything is transformed for display
    only -- nothing is written to ``adata``, and applying the answer is your own
    :func:`~cykit.asinh_transform` call.

    Every channel holds a cofactor from the moment the window opens, so moving
    between channels cannot lose a decision. Channels you have moved are marked
    in the channel list, which is what makes a forty-channel panel tractable:
    settle the bulk value on two or three, type it into **set every channel to**
    to put it on the whole panel, then go hunting for the handful that are
    wrong. ``n`` and ``p`` step between channels in file order.

    Parameters
    ----------
    adata
        What to read. One events x channels AnnData, a path, or several of
        either, exactly as :func:`~cykit.open_napari` accepts. Never modified.
    layer
        Matrix to read the **untransformed** values from, by name. Note the
        asymmetry with :func:`~cykit.open_napari`, which is pointed at a layer
        that has already been transformed: this window does the arcsinh itself,
        so it needs the values the transform would be applied to. Usually
        ``"comp"``, since cofactors are best chosen on compensated data.
    x, y
        Channels on the two axes to begin with. ``x`` defaults to the first
        tunable channel; ``y`` to a scatter channel, whose slider stays greyed
        out because scatter is never transformed.
    cofactor_range
        ``(lo, hi)`` the sliders span, in raw data units. Required, and
        ``0 < lo < hi``: no one range suits both mass cytometry and spectral
        flow, so it is your prior on where the answer lives. The slider is
        log-spaced across it, so narrowing it is how you get finer control.
    channels
        Channels to tune. Defaults to :func:`~cykit.get_fluor_channels`.
    cofactor
        Seed for the sliders: one value for every channel, or a mapping read the
        way :func:`~cykit.asinh_transform` reads one, including its
        ``"default"`` key. ``None`` seeds the geometric midpoint of
        ``cofactor_range``. A seed outside the range is refused rather than
        clamped.
    ticks
        How the tuned axes are labelled, one of
        :data:`~cykit.viewer.TICK_CHOICES`, and changeable in the window
        afterwards. ``"untransformed"``, the default, puts the ticks at round
        numbers of the raw units the slider is measured in, so the knee line
        and the cofactor read against the same scale. ``"transformed"`` labels
        the arcsinh values the panel is actually plotting. Labelling only: it
        moves nothing.
    margin
        How far past the data the axes run, as a proportion of the span added
        at each end; :data:`~cykit.scales.AXIS_MARGIN` by default, and
        adjustable in the window with the **axis margin %** slider. The span is
        in display coordinates -- decades, on an arcsinh axis -- so a tenth of
        it moves the raw value at the end of the axis by a good deal more than
        a tenth. Capped at :data:`MAX_MARGIN_PERCENT`.
    max_events
        Events plotted. ``None`` plots every one. The statistics in the status
        line always come from every event, whatever this is set to.
    seed
        Seed for that draw, so the same events are plotted each time.
    bins
        Resolution of each panel, per axis.
    smooth
        Gaussian smoothing of the density, in bins. The curves get twice this.
    colormap
        Colormap for the density panel.
    background
        Canvas colour behind the plots. Any napari colour.
    viewer
        An existing ``napari.Viewer`` to build into. One is created when omitted.
    title
        Window title, used only when creating a viewer.
    """

    def __init__(
        self,
        adata,
        layer: str,
        x: str | None = None,
        y: str | None = None,
        *,
        cofactor_range,
        channels: Sequence[str] | None = None,
        cofactor=None,
        ticks: str = "untransformed",
        margin: float = AXIS_MARGIN,
        max_events: int | None = MAX_EVENTS,
        seed: int = 0,
        bins: int = 512,
        smooth: float = 1.0,
        colormap: str = "turbo",
        background: str | None = None,
        viewer=None,
        title: str = "cykit cofactors",
    ):
        """Build the panels and the control panel, then draw. See the class docstring."""
        import napari

        from .viewer import DEFAULT_BACKGROUND, Panel, _hide_overlays, _is_light, _resolve_ticks

        self.cofactor_range = _resolve_range(cofactor_range)
        lo, hi = self.cofactor_range

        adata = _as_one(adata)
        self.adata = adata
        self.layer = str(layer)
        layer_matrix(adata, self.layer)  # fail here, not on the first redraw
        self.bins = int(bins)
        self.smooth = float(smooth)
        self._updating = False
        self._drawing = False
        self._cache: OrderedDict[str, _ChannelData] = OrderedDict()

        self.channels = _resolve_names(adata, channels)
        if not self.channels:
            raise ValueError("no channels to tune")
        self.cofactors = Cofactors(_seed(adata, self.channels, cofactor, lo, hi), layer=self.layer)
        self._ticks = _resolve_ticks(ticks)
        self.margin = _resolve_margin(margin)
        self.x = str(adata.var_names[find_channel_name(adata, x)]) if x else self.channels[0]
        self.y = (
            str(adata.var_names[find_channel_name(adata, y)])
            if y
            else _default_y(adata, self.channels)
        )

        rows = subsample_indices(adata.n_obs, max_events, rng=np.random.default_rng(seed))
        self._rows = rows
        self.n_plotted = int(adata.n_obs if rows is None else rows.size)
        self.n_total = int(adata.n_obs)

        background = DEFAULT_BACKGROUND if background is None else background
        self.background = str(background)
        light = _is_light(self.background)
        self._foreground = "#1e1e21" if light else "#f0f1f2"
        self._grid_colour = "#9a9a9a" if light else "#5a5a5a"
        self._knee_colour = "#c0392b" if light else "#ff7b6b"

        self.viewer = viewer if viewer is not None else napari.Viewer(title=title)
        _hide_overlays(self.viewer)

        b, g = self.bins, self.bins * GUTTER
        self.p_density = Panel(
            self.viewer,
            1,
            kind="density",
            layer=self.layer,
            x=self.x,
            y=self.y,
            row=(b + g) / 2.0,
            col=0.0,
            colormap=colormap,
        )
        self.p_x = Panel(
            self.viewer,
            2,
            kind="histogram",
            layer=self.layer,
            x=self.x,
            row=0.0,
            col=b + g,
            colormap=colormap,
        )
        self.p_y = Panel(
            self.viewer,
            3,
            kind="histogram",
            layer=self.layer,
            x=self.y,
            row=b + g,
            col=b + g,
            colormap=colormap,
        )
        for panel in self._panels:
            panel.image.translate = (panel.row, panel.col)
        for panel in (self.p_x, self.p_y):
            panel.image.colormap = _fill_colormap(_CURVE)
            panel.image.blending = "translucent_no_depth"
        self._init_overlays()
        self._init_widget()
        self.viewer.window.add_dock_widget(self.widget, area="right", name="cofactors")

        # scipy.ndimage is imported lazily inside density_image, which makes the
        # first call several times slower. Pay for it now rather than on the
        # first drag of a slider.
        density_image(np.zeros(1), np.zeros(1), Axes2D(0.0, 1.0, 0.0, 1.0, bins=8))
        self._set_background()
        self._bind()
        self.refresh()
        self.viewer.reset_view()

    # ------------------------------------------------------------------ setup
    @property
    def _panels(self):
        return (self.p_density, self.p_x, self.p_y)

    def _init_overlays(self) -> None:
        """The frame, the labels, and the two guides that read a cofactor."""
        from .viewer import _TEXT_STYLE, Y_LABEL_ROTATION

        # Drawn under the frame: a band is a backdrop, not an annotation.
        # Full opacity, with the transparency carried in the colours: the band
        # wants to be a wash, but the knee line is the thing you actually read a
        # cofactor against and has to stay crisp against it.
        self.guides = self.viewer.add_shapes(name="guides", opacity=1.0)
        self.guides.editable = False
        # Vectors rather than Shapes: the frame and its ticks are ~150 separate
        # segments, and rebuilding them as shapes measured 19 ms against 2 ms
        # here -- on a layer that is rewritten on every slider tick.
        self.grid = self.viewer.add_vectors(
            np.zeros((0, 2, 2)),
            name="axes",
            edge_width=1,
            edge_color=self._grid_colour,
            vector_style="line",
            opacity=0.6,
        )
        self.labels = self.viewer.add_points(
            np.empty((0, 2)),
            name="axis labels",
            size=1,
            face_color="transparent",
            border_color="transparent",
            text={**_TEXT_STYLE, "string": [], "color": self._foreground},
        )
        self.labels.editable = False
        self.ylabel = self.viewer.add_points(
            np.empty((0, 2)),
            name="y axis labels",
            size=1,
            face_color="transparent",
            border_color="transparent",
            text={
                **_TEXT_STYLE,
                "string": [],
                "color": self._foreground,
                "rotation": Y_LABEL_ROTATION,
            },
        )
        self.ylabel.editable = False
        self.viewer.layers.selection = set()

    def _set_background(self) -> None:
        try:
            self.viewer.canvas.background_color_override = self.background
        except AttributeError:  # pragma: no cover - older napari
            canvas = getattr(getattr(self.viewer.window, "_qt_viewer", None), "canvas", None)
            if canvas is not None:
                canvas.bgcolor = self.background

    def _cofactor_row(self, label: str, live: bool):
        """A log slider with a box beside it holding the same cofactor.

        The slider's own readout is hidden because magicgui fills it with the
        *slider position* -- 869 for a cofactor of 3000 -- which is an internal
        coordinate no one wants to read, let alone type into.

        The box is where an exact number goes. The slider quantises to a
        thousandth of its travel, about a quarter of a per cent here, so typing
        3000 and letting the slider round it would store 2992. The typed value
        goes straight to :meth:`set_cofactor` instead, and the slider is only
        moved to the nearest step it can reach -- quietly, so it cannot write
        its own rounding back.

        Returns
        -------
        tuple
            ``(slider, box, row)`` -- the row is what goes in the panel.
        """
        from magicgui.widgets import Container, LineEdit

        lo, hi = self.cofactor_range
        slider = _log_slider_class()(
            label="", min=lo, max=hi, base=10, max_pos=SLIDER_STEPS, tracking=live
        )
        slider._widget._mgui_set_readout_visibility(False)
        box = LineEdit(label="", tooltip=_BOX_TOOLTIP)
        box.max_width = _BOX_WIDTH
        row = Container(widgets=[slider, box], layout="horizontal", label=label, labels=False)
        row.margins = (0, 0, 0, 0)
        return slider, box, row

    def _init_widget(self) -> None:
        from magicgui.widgets import (
            CheckBox,
            ComboBox,
            Container,
            Label,
            LineEdit,
            PushButton,
            Slider,
        )

        from .viewer import TICK_CHOICES

        live = self.n_plotted <= MAX_EVENTS

        self.w_x = ComboBox(label="x", choices=self._channel_choices(), value=self.x)
        self.w_y = ComboBox(label="y", choices=[str(n) for n in self.adata.var_names], value=self.y)

        self.w_cx, self.w_cx_box, cx_row = self._cofactor_row("x cofactor", live)
        self.w_cy, self.w_cy_box, cy_row = self._cofactor_row("y cofactor", live)
        self._cy_row = cy_row
        self.w_ticks = ComboBox(label="axis ticks", choices=TICK_CHOICES, value=self._ticks)
        self.w_margin = Slider(
            label="axis margin %",
            min=0,
            max=MAX_MARGIN_PERCENT,
            value=round(self.margin * 100),
            tracking=live,
            tooltip=_MARGIN_TOOLTIP,
        )
        self.w_all = LineEdit(label="set every channel to", tooltip=_ALL_TOOLTIP)
        self.w_copy = PushButton(text="copy as a Python dict")
        self.w_live = CheckBox(label="redraw while dragging", value=live)

        self.w_head = Label(value="")
        self.w_status = Label(value="")

        channel = Container(
            widgets=[self.w_x, self.w_y, self.w_ticks, self.w_margin], label="channels"
        )
        cofactor = Container(
            widgets=[
                cx_row,
                cy_row,
                self.w_all,
                self.w_live,
            ],
            label="cofactor",
        )
        self.widget = Container(
            widgets=[channel, cofactor, self.w_head, self.w_status, self.w_copy]
        )

        self.w_x.changed.connect(self._x_changed)
        self.w_y.changed.connect(self._y_changed)
        self.w_cx.changed.connect(lambda e: self._slider_moved(self.x, self.w_cx.value))
        self.w_cy.changed.connect(lambda e: self._slider_moved(self.y, self.w_cy.value))
        self.w_cx_box.native.editingFinished.connect(lambda: self._box_entered(self.w_cx_box, "x"))
        self.w_cy_box.native.editingFinished.connect(lambda: self._box_entered(self.w_cy_box, "y"))
        # editingFinished, not the widget's own ``changed``: magicgui wires a
        # LineEdit to textChanged, which would fire -- and redraw -- on every
        # keystroke, so "3000" would pass through 3, 30 and 300 on its way.
        self.w_ticks.changed.connect(self._ticks_changed)
        self.w_margin.changed.connect(self._margin_changed)
        self.w_all.native.editingFinished.connect(self._all_entered)
        self.w_copy.changed.connect(self._copy)
        self.w_live.changed.connect(self._set_tracking)

        self.viewer.bind_key("n", lambda v: self.step(1), overwrite=True)
        self.viewer.bind_key("p", lambda v: self.step(-1), overwrite=True)

    # -------------------------------------------------------------- selection
    @property
    def _y_tuned(self) -> bool:
        """Whether the y channel is one this window transforms."""
        return self.y in self.cofactors

    def _channel_choices(self):
        """``(label, var_name)`` per channel, the label carrying its state."""
        out = []
        for name in self.channels:
            mark = "●" if name in self.cofactors.adjusted else "○"
            out.append((f"{mark} {name}   {self.cofactors[name]:,.0f}", name))
        return out

    @contextmanager
    def _quiet(self):
        """Set widget values without their ``changed`` callbacks firing back."""
        self._updating = True
        try:
            yield
        finally:
            self._updating = False

    def _bind(self) -> None:
        """Point the widgets at the current state, without firing their callbacks."""
        lo, _ = self.cofactor_range
        with self._quiet():
            # Reassigning choices resets a ComboBox to its first entry, so the
            # value has to be put back after the marks are redrawn.
            keep = self.x
            self.w_x.choices = self._channel_choices()
            self.w_x.value = keep
            self.w_y.value = self.y
            # The slider rounds to the nearest step it has; the box carries the
            # value that is actually stored, which is the one that gets used.
            self.w_cx.value = self.cofactors[self.x]
            self.w_cx_box.value = f"{self.cofactors[self.x]:g}"
            self.w_cy.value = self.cofactors.get(self.y, lo)
            self.w_cy_box.value = f"{self.cofactors[self.y]:g}" if self._y_tuned else ""
            self.w_cy.enabled = self.w_cy_box.enabled = self._y_tuned
            self._cy_row.label = "y cofactor" if self._y_tuned else "y (not transformed)"
            self.w_margin.value = round(self.margin * 100)

    def set_cofactor(self, channel: str, value: float) -> None:
        """Set one channel's cofactor and redraw. The single mutation point.

        Parameters
        ----------
        channel
            Channel to set, by var_name. Must be one this window is tuning.
        value
            The cofactor, clamped into ``cofactor_range``.
        """
        if channel not in self.cofactors:
            raise KeyError(f"{channel!r} is not one of the channels being tuned")
        lo, hi = self.cofactor_range
        self.cofactors[channel] = float(min(max(float(value), lo), hi))
        self.cofactors.adjusted.add(channel)
        self._bind()
        self.refresh()

    def set_channels(self, x: str | None = None, y: str | None = None) -> None:
        """Change which channels the panels show.

        Parameters
        ----------
        x
            Channel to tune, by any alias :func:`~cykit.find_channel_name` accepts.
        y
            Channel to plot it against.
        """
        if x is not None:
            self.x = str(self.adata.var_names[find_channel_name(self.adata, x)])
        if y is not None:
            self.y = str(self.adata.var_names[find_channel_name(self.adata, y)])
        self._bind()
        self.refresh()

    def step(self, delta: int = 1) -> None:
        """Move to another channel in file order, wrapping at either end.

        Parameters
        ----------
        delta
            How far to move; negative goes back.
        """
        if self.x in self.channels:
            index = self.channels.index(self.x)
        else:  # pragma: no cover - x is always one of them
            index = 0
        self.set_channels(x=self.channels[(index + int(delta)) % len(self.channels)])

    def set_margin(self, margin: float) -> None:
        """How far past the data the axes run, and redraw.

        Parameters
        ----------
        margin
            Proportion of the span to add beyond each end, clamped to
            ``0 .. MAX_MARGIN_PERCENT / 100``. The span is in display
            coordinates, so on these arcsinh axes it is measured in decades and
            a small proportion moves the raw value at the end of the axis by a
            good deal more.
        """
        self.margin = _resolve_margin(margin)
        self._bind()
        self.refresh()

    def set_all(self, value: float) -> None:
        """Give every channel this cofactor, and clear the per-channel marks.

        The action that makes a forty-channel panel tractable: settle a bulk
        value on two or three channels, put it on the whole panel, then go
        hunting for the handful that are wrong. It overwrites every channel,
        including ones you have already moved, so afterwards nothing is
        individually tuned and every channel reads ``○`` again -- which is the
        truth, since they all now hold the same number. There is no undo.

        Parameters
        ----------
        value
            Cofactor to write, clamped into ``cofactor_range``.
        """
        lo, hi = self.cofactor_range
        value = float(min(max(float(value), lo), hi))
        for name in self.cofactors:
            self.cofactors[name] = value
        self.cofactors.adjusted.clear()
        self._bind()
        self.refresh()

    # -------------------------------------------------------------- callbacks
    def _slider_moved(self, channel, value) -> None:
        if self._updating or channel not in self.cofactors:
            return
        self.set_cofactor(channel, value)

    def _x_changed(self, *_) -> None:
        if not self._updating:
            self.set_channels(x=str(self.w_x.value))

    def _y_changed(self, *_) -> None:
        if not self._updating:
            self.set_channels(y=str(self.w_y.value))

    def _margin_changed(self, *_) -> None:
        if not self._updating:
            self.set_margin(int(self.w_margin.value) / 100.0)

    def _ticks_changed(self, *_) -> None:
        if not self._updating:
            self._ticks = str(self.w_ticks.value)
            self.refresh()

    def _box_entered(self, box, axis: str) -> None:
        """A cofactor typed beside a slider. Exact: the slider does not round it."""
        channel = self.x if axis == "x" else self.y
        if channel not in self.cofactors:
            return
        text = str(box.value).strip()
        if not text:
            self._bind()
            return
        try:
            value = float(text.replace(",", "").replace("_", ""))
        except ValueError:
            self.w_status.value = f"{text!r} is not a number"
            self._bind()
            return
        lo, hi = self.cofactor_range
        self.set_cofactor(channel, value)
        if not lo <= value <= hi:
            self.w_status.value = (
                f"{value:g} is outside {lo:g}-{hi:g} - clamped to {self.cofactors[channel]:g}"
            )

    def _all_entered(self, *_) -> None:
        text = str(self.w_all.value).strip()
        if not text:
            return
        try:
            value = float(text.replace(",", "").replace("_", ""))
        except ValueError:
            self.w_status.value = f"{text!r} is not a number"
            return
        lo, hi = self.cofactor_range
        self.set_all(value)
        # Echo back what was actually applied, so the box never shows a number
        # the panel is not on.
        applied = self.cofactors[self.x]
        with self._quiet():
            self.w_all.value = f"{applied:g}"
        if not lo <= value <= hi:
            self.w_status.value = f"{value:g} is outside {lo:g}–{hi:g} — clamped to {applied:g}"

    def _copy(self, *_) -> None:
        from qtpy.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.cofactors.to_source())
        self.w_status.value = "copied — paste it into your notebook"

    def _set_tracking(self, *_) -> None:
        for slider in (self.w_cx, self.w_cy):
            slider.tracking = bool(self.w_live.value)

    # ------------------------------------------------------------------- data
    def _channel_data(self, channel: str) -> _ChannelData:
        """This channel's raw values and its cofactor-independent statistics.

        Read once and kept, because none of it changes when a slider moves:
        re-slicing a million-row matrix on every tick is the one cost that would
        make the window feel dead.
        """
        hit = self._cache.get(channel)
        if hit is not None:
            self._cache.move_to_end(channel)
            return hit

        j = find_channel_name(self.adata, channel)
        column = np.asarray(layer_matrix(self.adata, self.layer)[:, j]).ravel()
        full = np.ascontiguousarray(column, dtype=np.float32)
        full = full[np.isfinite(full)]
        raw = full if self._rows is None else np.ascontiguousarray(full[self._rows])
        data = _ChannelData(raw, full)

        self._cache[channel] = data
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return data

    # -------------------------------------------------------------- rendering
    def refresh(self, *_) -> None:
        """Redraw all three panels, the frame and the guides."""
        if self._updating or self._drawing:
            # Qt will deliver the next slider value anyway, so a dropped frame
            # costs nothing and a queued one costs everything.
            return
        self._drawing = True
        try:
            self._redraw()
        finally:
            self._drawing = False

    def _tuned_scale(self, cofactor: float):
        """How to label a tuned axis. Tick placement only; it moves nothing.

        The panel always plots ``asinh(raw / cofactor)``, so ``"transformed"``
        is the plain identity -- the stored numbers are already the display
        coordinates. ``"untransformed"`` hands the same coordinates to
        :class:`~cykit.scales.PretransformedScale`, which works out where the
        decades of the original units land and labels those instead.
        """
        from .scales import LinearScale

        if self.w_ticks.value == "transformed":
            return LinearScale()
        return PretransformedScale(AsinhScale(cofactor=cofactor))

    def _redraw(self) -> None:
        from .scales import LinearScale

        # The panels carry their own channel names, and that is what the frame
        # labels its axes with -- so they have to be re-pointed here, or the
        # label under a panel goes on naming the channel it opened on.
        self.p_density.x, self.p_density.y = self.x, self.y
        self.p_x.x = self.x
        self.p_y.x = self.y

        dx = self._channel_data(self.x)
        cx = float(self.cofactors[self.x])
        xv = dx.display(cx)
        x_lo, x_hi = dx.limits(cx, self.margin)
        x_scale = self._tuned_scale(cx)

        dy = self._channel_data(self.y)
        if self.y == self.x:
            yv, y_lo, y_hi, y_scale = xv, x_lo, x_hi, x_scale
        elif self._y_tuned:
            cy = float(self.cofactors[self.y])
            yv = dy.display(cy)
            y_lo, y_hi = dy.limits(cy, self.margin)
            y_scale = self._tuned_scale(cy)
        else:
            # Scatter is never transformed, so it is plotted as stored and its
            # axis stays put however far the x slider travels.
            yv = dy.raw
            y_lo, y_hi = pad_range(dy.q_lo, dy.q_hi, self.margin)
            y_scale = LinearScale()

        self.p_density.axes = Axes2D(x_lo, x_hi, y_lo, y_hi, bins=self.bins)
        self.p_density.x_scale, self.p_density.y_scale = x_scale, y_scale
        image = density_image(xv, yv, self.p_density.axes, smooth=self.smooth, log=True)
        self.p_density.image.data = image
        self.p_density.image.contrast_limits = (0.0, float(image.max()) or 1.0)
        self.p_density.image.visible = True
        self.p_density.curves.visible = False

        for panel, values, scale, lo, hi in (
            (self.p_x, xv, x_scale, x_lo, x_hi),
            (self.p_y, yv, y_scale, y_lo, y_hi),
        ):
            panel.axes = Axes2D(lo, hi, 0.0, 1.0, bins=self.bins)
            panel.x_scale, panel.y_scale = scale, LinearScale()
            panel.image.visible = True
            panel.curves.visible = True
            self._draw_curve(panel, values, lo, hi)

        self._draw_guides(cx)
        self._draw_frames()
        self._status()

    def _draw_curve(self, panel, values, lo, hi) -> None:
        """One distribution, scaled so its tallest point is 100 per cent of mode.

        The outline is a shape, but the area under it is an **image**. Filling it
        with a polygon means triangulating a ``bins``-vertex shape on every
        slider tick, which measured 49 ms of a 50 ms redraw and does not shrink
        with the event count -- it would have set the floor on how live the
        window could feel, whatever the kernels underneath did. A mask costs
        0.7 ms and the panel already owns an image layer to put it in.
        """
        from .viewer import _MODE_TOP

        curve = density_curve(values, lo, hi, self.bins, smooth=max(self.smooth, 0.5) * 2)
        peak = float(curve.max())
        curve = curve / peak * _MODE_TOP if peak > 0 else curve
        scale = (self.bins - 1) / (_MODE_TOP * 1.05)
        heights = (self.bins - 1.0) - curve * scale

        rows = np.clip(np.round(heights).astype(np.intp), 0, self.bins - 1)
        under = np.arange(self.bins)[:, None] >= rows[None, :]
        panel.image.data = under.astype(np.float32)
        panel.image.contrast_limits = (0.0, 1.0)

        columns = np.arange(self.bins, dtype=float) - 0.5 + panel.col
        line = np.column_stack([panel.row + heights, columns])
        panel.curves.data = []
        panel.curves.add([line], shape_type=["path"], edge_color=[_CURVE], edge_width=2)
        panel.peak = _MODE_TOP

    def _draw_guides(self, cofactor: float) -> None:
        """The knee line and the spread of the negative population.

        Both read a cofactor at a glance. The knee sits at ``asinh(1)`` whatever
        the cofactor is, so it never moves and the data slides under it: the
        negative population belongs comfortably to its left, the positives to
        its right.
        """
        shapes, kinds, faces, edges = [], [], [], []
        b = self.bins
        for panel, channel in ((self.p_x, self.x), (self.p_y, self.y)):
            ax = panel.axes
            if ax is None or (channel not in self.cofactors and panel is self.p_y):
                continue
            c = float(self.cofactors[channel])
            top, bottom = panel.row - 0.5, panel.row + b - 0.5
            if ax.x_lo <= KNEE <= ax.x_hi:
                col = panel.col + ax.to_pixels(np.array([KNEE]), np.array([ax.y_lo]))[0, 1]
                shapes.append(np.array([[top, col], [bottom, col]]))
                kinds.append("path")
                faces.append("#00000000")
                edges.append(self._knee_colour)

            data = self._cache.get(channel)
            if data is None or data.n_neg < MIN_NEGATIVES:
                continue
            band = np.arcsinh(np.array([data.q05_neg, data.q95_neg]) / c)
            band = np.clip(band, ax.x_lo, ax.x_hi)
            c0, c1 = panel.col + ax.to_pixels(band, np.full(2, ax.y_lo))[:, 1]
            if c1 - c0 >= 1.0:
                shapes.append(np.array([[top, c0], [top, c1], [bottom, c1], [bottom, c0]]))
                kinds.append("polygon")
                faces.append(_BAND + "2e")
                edges.append("#00000000")

        self.guides.data = []
        if shapes:
            self.guides.add(
                shapes, shape_type=kinds, face_color=faces, edge_color=edges, edge_width=2
            )
        self.guides.visible = bool(shapes)

    def _draw_frames(self) -> None:
        """A labelled box around each panel."""
        from .viewer import _TEXT_STYLE, Y_LABEL_ROTATION, _panel_frame

        lines: list[np.ndarray] = []
        coords: list[list[float]] = []
        text: list[str] = []
        y_coords: list[list[float]] = []
        y_text: list[str] = []
        tick = 0.02 * self.bins

        for panel, y_label, title, mode in (
            (self.p_density, self.y, f"{self.x} × {self.y}", False),
            (self.p_x, "", f"{self.x}   c = {self.cofactors[self.x]:,.0f}", True),
            (self.p_y, "", self._y_title(), True),
        ):
            if panel.axes is None:  # pragma: no cover - always drawn first
                continue
            pl, pc, pt, yl = _panel_frame(
                panel.axes,
                x_scale=panel.x_scale,
                y_scale=panel.y_scale,
                bins=self.bins,
                row=panel.row,
                col=panel.col,
                x_label=str(panel.x),
                y_label=str(y_label),
                title=title,
                mode_axis=mode,
            )
            lines += pl
            coords += pc
            text += pt
            y_coords.append([panel.row + self.bins / 2.0, panel.col - 12.0 * tick])
            y_text.append(yl)

        if lines:
            ends = np.asarray(lines, dtype=float)
            self.grid.data = np.stack([ends[:, 0], ends[:, 1] - ends[:, 0]], axis=1)
        else:  # pragma: no cover - there is always a frame
            self.grid.data = np.zeros((0, 2, 2))
        self.labels.data = np.asarray(coords, dtype=float) if coords else np.empty((0, 2))
        self.labels.text = {**_TEXT_STYLE, "string": text, "color": self._foreground}
        self.ylabel.data = np.asarray(y_coords, dtype=float) if y_coords else np.empty((0, 2))
        self.ylabel.text = {
            **_TEXT_STYLE,
            "string": y_text,
            "color": self._foreground,
            "rotation": Y_LABEL_ROTATION,
        }

    def _y_title(self) -> str:
        if self._y_tuned:
            return f"{self.y}   c = {self.cofactors[self.y]:,.0f}"
        return f"{self.y}   not transformed"

    # ------------------------------------------------------------------ status
    def statistics(self, channel: str | None = None) -> dict[str, float]:
        """What the current cofactor is doing to a channel, in numbers.

        Computed from **every** event, not from the subsample the picture is
        drawn from, so thinning the plot never changes what is reported.

        Parameters
        ----------
        channel
            Channel to describe. Defaults to the one being tuned.

        Returns
        -------
        dict
            ``frac_negative`` of events below zero; ``n_negative``; ``spread``,
            the width of the negative population in display units, which wants
            to be around 1; and ``separation`` between the negative median and
            the bright tail, which wants to be more than about 2.
        """
        channel = self.x if channel is None else channel
        data = self._channel_data(channel)
        c = float(self.cofactors.get(channel, 1.0))
        spread = separation = float("nan")
        if data.n_neg >= MIN_NEGATIVES:
            lo, hi = np.arcsinh(np.array([data.q05_neg, data.q95_neg]) / c)
            spread = float(hi - lo)
            separation = float(np.arcsinh(data.q_hi / c) - np.arcsinh(data.median_neg / c))
        return {
            "frac_negative": data.n_neg / data.n if data.n else 0.0,
            "n_negative": float(data.n_neg),
            "spread": spread,
            "separation": separation,
        }

    def _status(self) -> None:
        data = self._channel_data(self.x)
        stats = self.statistics()
        mark = "● adjusted" if self.x in self.cofactors.adjusted else "○ on the seed"
        self.w_head.value = (
            f"{self.x}   c = {self.cofactors[self.x]:,.0f}   {mark}"
            f"   —   {self.cofactors.n_adjusted} / {len(self.channels)} adjusted"
        )
        if data.n_neg < MIN_NEGATIVES:
            detail = (
                f"only {data.n_neg:,} negative events — "
                "this channel gives a cofactor nothing to sit on"
            )
        else:
            detail = (
                f"{stats['frac_negative']:.1%} negative   "
                f"spread {stats['spread']:.2f}   separation {stats['separation']:.1f}"
            )
        drawn = (
            f"{self.n_plotted:,} of {self.n_total:,} events"
            if self._rows is not None
            else f"{self.n_total:,} events"
        )
        live = "live" if self.w_live.value else "updates on release"
        self.w_status.value = f"{detail}   ·   {drawn} · {live}"


#: The window this session last opened, so a non-blocking call does not let it
#: be collected and :func:`current_transform_window` has something to hand back.
_CURRENT_TRANSFORM: CofactorWindow | None = None


def current_transform_window() -> CofactorWindow | None:
    """The window :func:`open_napari_transform` last opened.

    Returns
    -------
    CofactorWindow or None
        ``None`` before one has been opened in this session.
    """
    return _CURRENT_TRANSFORM


def _fill_colormap(colour: str):
    """A flat wash of one colour, transparent where the histogram is empty."""
    from napari.utils.colormaps import Colormap

    rgb = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return Colormap(colors=[[*rgb, 0.0], [*rgb, 0.28]], name="cykit_fill")


def _as_one(adata):
    from .viewer import as_one_anndata

    return as_one_anndata(adata)


def open_napari_transform(
    adata,
    layer: str,
    x: str | None = None,
    y: str | None = None,
    *,
    cofactor_range,
    channels: Sequence[str] | None = None,
    cofactor=None,
    ticks: str = "untransformed",
    margin: float = AXIS_MARGIN,
    max_events: int | None = MAX_EVENTS,
    seed: int = 0,
    bins: int = 512,
    smooth: float = 1.0,
    colormap: str = "turbo",
    background: str | None = None,
    block: bool | None = None,
    verbose: bool = False,
) -> Cofactors:
    """Choose arcsinh cofactors by eye, and hand the numbers back.

    Opens a window showing what a cofactor *does*: the two channels against each
    other, and a distribution of each with its own log-spaced slider. Nothing is
    written to ``adata`` -- applying the answer is still your own
    :func:`~cykit.asinh_transform` call, which is what keeps a transform
    something you ran rather than something that happened.

    ``layer`` is the **untransformed** matrix, unlike
    :func:`~cykit.open_napari`, which is pointed at a layer already transformed.

    Parameters
    ----------
    adata
        What to read. One events x channels AnnData, a path to an FCS file, an
        ``.h5ad`` or a directory, or several of either as a list or a
        ``{name: object}`` mapping. Never modified.
    layer
        Layer to read the untransformed values from, by name. Required, so a
        call always says which matrix it measured. Usually ``"comp"``.
    x, y
        Channels to start on. ``x`` defaults to the first tunable channel, ``y``
        to a scatter channel, whose slider stays greyed out because scatter is
        never transformed -- so one axis moves at a time and a change in the
        picture is attributable to the slider you moved.
    cofactor_range
        ``(lo, hi)`` the sliders span, in raw units, with ``0 < lo < hi``.
        Required: it is your prior on where the answer lives, and there is no
        range that suits both mass cytometry and spectral flow. The slider is
        log-spaced across it, so narrowing it is how you get finer control.
    channels
        Channels to tune. Defaults to :func:`~cykit.get_fluor_channels`.
    cofactor
        Seed for the sliders: one value for every channel, or a mapping read the
        way :func:`~cykit.asinh_transform` reads one. ``None`` seeds the
        geometric midpoint of ``cofactor_range``, which makes no claim about the
        data. A seed outside the range is refused rather than clamped.
    ticks
        How the tuned axes are labelled to begin with, one of
        :data:`~cykit.viewer.TICK_CHOICES`. ``"untransformed"``, the default,
        labels them in the raw units the cofactor itself is measured in;
        ``"transformed"`` labels the arcsinh values being plotted. The **axis
        ticks** box changes it in the window.
    margin
        How far past the data the axes run, as a proportion of the span added
        at each end; :data:`~cykit.scales.AXIS_MARGIN` by default, and
        adjustable in the window with the **axis margin %** slider. The span is
        in display coordinates -- decades, on an arcsinh axis -- so a tenth of
        it moves the raw value at the end of the axis by a good deal more than
        a tenth. Capped at :data:`MAX_MARGIN_PERCENT`.
    max_events
        Events plotted. ``None`` plots every one. The reported statistics always
        come from every event.
    seed
        Seed for that draw, so the same events are plotted each time.
    bins
        Resolution of each panel, per axis.
    smooth
        Gaussian smoothing of the density, in bins.
    colormap
        Colormap for the density panel.
    background
        Canvas colour behind the plots. Any napari colour.
    block
        Wait for the window to close before returning. The default detects the
        context: ``True`` from a script, ``False`` under IPython, where a Qt loop
        is already running and blocking would wedge the kernel.

        A non-blocking call returns **before you have moved anything**. The dict
        it returns is the live one -- already seeded for every channel, and
        filling in as you drag -- so re-evaluating it in a later cell shows where
        you have got to.
    verbose
        Print the cofactors, as a dict literal, when the window closes.

    Returns
    -------
    Cofactors
        var_name to cofactor, ready to pass to
        :func:`~cykit.asinh_transform`. Its ``repr`` is a dict literal you can
        paste into a notebook, which is also what the window's **copy as a
        Python dict** button puts on the clipboard.
    """
    global _CURRENT_TRANSFORM
    import napari

    _CURRENT_TRANSFORM = CofactorWindow(
        adata,
        layer,
        x,
        y,
        cofactor_range=cofactor_range,
        channels=channels,
        cofactor=cofactor,
        ticks=ticks,
        margin=margin,
        max_events=max_events,
        seed=seed,
        bins=bins,
        smooth=smooth,
        colormap=colormap,
        background=background,
    )
    if block is None:
        try:
            get_ipython  # type: ignore[name-defined]  # noqa: B018
            block = False
        except NameError:
            block = True
    if block:
        napari.run()
    if verbose:
        print(_CURRENT_TRANSFORM.cofactors.to_source())
    return _CURRENT_TRANSFORM.cofactors
