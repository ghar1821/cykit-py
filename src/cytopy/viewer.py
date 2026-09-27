"""napari front-end: 2-channel density plots with polygon gating.

The viewer never transforms data. It plots the values of the layer you point
it at, exactly as they are stored, so what you see is the result of the
transform *you* ran (:func:`cytopy.asinh_transform`,
:func:`cytopy.logicle_transform`, ...). The only thing it infers is how to
*label* the axes: when a layer records the transform that produced it, the
ticks are drawn as decades of the original units, which is what makes an
arcsinh or logicle layer read as a biexponential plot.

That is the contract of :class:`CytoViewer`, and it is the reason there is a
separate window for *choosing* a cofactor: :mod:`cytopy.cofactors` arcsinhs what
it draws, but only for display, in memory, and it hands back numbers rather than
a layer -- the same standing ``plot_biaxial(..., cofactor=...)`` has.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .density import Axes2D, density_curve, density_image
from .gating import add_gate, gate_children, gate_record, recompute_gates, shapes_mask
from .plotting import axis_limits, axis_scale
from .scales import LinearScale, Scale
from .transforms import find_channel_name

__all__ = [
    "CytoViewer",
    "Panel",
    "as_one_anndata",
    "current_viewer",
    "faded_colormap",
    "open_napari",
]

#: How the axes are labelled. ``"untransformed"`` puts the ticks at round
#: numbers of the channel's original units -- what the detector measured --
#: which is what makes an arcsinh layer read as biexponential. ``"transformed"``
#: labels the numbers actually stored in the layer. Neither ever changes the
#: data or where a point lands; only what is written beside the ticks.
TICK_CHOICES = ["untransformed", "transformed"]


def _resolve_ticks(value: str) -> str:
    """Validate an axis-tick mode, so a typo is not silently a different axis."""
    text = str(value)
    if text not in TICK_CHOICES:
        raise ValueError(f"ticks must be one of {TICK_CHOICES}, not {value!r}")
    return text


#: How far past the data the axis-range sliders can be dragged, as a fraction
#: of the data span. The slider has to reach beyond the data or it cannot frame
#: an outlier with any room around it, and it has to stop somewhere or the
#: whole travel is spent on empty canvas.
_SLIDER_OVERSHOOT = 1.0

#: Steps along each axis-range slider. Fine enough that a drag lands where you
#: meant it to on a 0..1 display axis.
_SLIDER_STEPS = 1000

#: Narrowest an axis may be dragged, as a fraction of a slider's travel.
#: Dragging "min" past "max" pushes the other end along rather than leaving an
#: inverted range, which reads as "fit the data" and would throw the view away
#: in the middle of a drag.
_MIN_AXIS_SPAN = 0.01
COLORMAPS = ["turbo", "viridis", "magma", "inferno", "gray", "plasma"]

#: Canvas background. White by default: a density plot reads as a flow plot on
#: white, and napari's own canvas is a dark blue-grey that most cytometry
#: software does not use. Any napari colour works -- a name, or "#rrggbb".
BACKGROUNDS = ["white", "black", "#262930", "#f5f5f5"]
DEFAULT_BACKGROUND = "white"

#: How much of the colormap fades out at the bottom, so that empty bins show
#: the background instead of the colormap's darkest colour.
_FADE = 0.02

_TEXT_STYLE = {"size": 12, "anchor": "center"}

_PANEL = "panel"
_YLABEL = "y axis label"
_GRID = "axes"
_LABELS = "axis labels"
_GATES = "gates"
_APPLIED = "applied gates"

#: What the canvas draws. ``"density"`` is the two-channel plot;
#: ``"histogram"`` is one smoothed distribution of the x channel per sample.
PLOT_KINDS = ["density", "histogram"]

#: Top of a histogram's y axis. Curves are scaled to per cent of mode, so the
#: tallest point of each one is 100.
_MODE_TOP = 100.0

#: Opacity of the area under a histogram curve, as a two-digit hex suffix.
#: Light enough that curves stay legible where they overlap.
CURVE_FILL_ALPHA = "2e"

#: Rotation of the y-axis label, in degrees, so it reads up the side.
Y_LABEL_ROTATION = 270

#: Curve colours for the histogram, one per sample, reused beyond eight.
CURVE_COLOURS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#9d755d",
    "#eeca3b",
    "#72b7b2",
]


#: How the comparison panel is shown against the main one.
class Panel:
    """One plot on the canvas: what it shows, where it sits, and its layers.

    A panel owns the image (or the curves) it draws into, so it appears in
    napari's own layer list and can be hidden or reordered there. Its frame and
    labels are drawn from :attr:`row` and :attr:`col`, so the box travels with
    the plot when it is moved.

    Parameters
    ----------
    viewer
        The napari viewer to add the layers to.
    index
        Number used to name the layers; unique within a window.
    kind
        ``"density"`` for the two-channel plot, ``"histogram"`` for one
        smoothed distribution of ``x`` per sample.
    layer
        Matrix to plot: ``"X"`` or a key of ``adata.layers``.
    x, y
        Channels on each axis. ``y`` is unused by a histogram.
    samples
        Samples to show. Empty means all of them: a density pools whatever is
        picked, a histogram draws a curve for each.
    parent
        Gate to restrict to, or ``"<none>"``.
    row, col
        Position of the panel's top-left corner, in bin units.
    colormap
        Colormap for the density image.

    Attributes
    ----------
    x_lim, y_lim
        Manual axis ranges in display coordinates, or ``None`` to fit the data.
        Set from the window's min/max fields; see
        :meth:`CytoViewer.set_limits`.
    """

    def __init__(
        self,
        viewer,
        index: int,
        *,
        kind: str = "density",
        layer: str = "X",
        x: str = "",
        y: str = "",
        samples: Sequence[str] = (),
        parent: str = "<none>",
        row: float = 0.0,
        col: float = 0.0,
        colormap: str = "turbo",
    ):
        self.index = int(index)
        self.kind = kind
        self.layer = layer
        self.x = x
        self.y = y
        self.samples = tuple(samples)
        self.parent = parent
        # Where the panel's top-left corner sits, in bin units. Zero for a
        # window showing one plot; the cofactor window stacks three.
        self.row = float(row)
        self.col = float(col)

        self.axes: Axes2D | None = None
        self.x_scale: Scale | None = None
        self.y_scale: Scale | None = None
        # Manual axis ranges, in display coordinates. `None` means the panel
        # works them out from the data, which is what it does until someone
        # types into the min/max fields.
        self.x_lim: tuple[float, float] | None = None
        self.y_lim: tuple[float, float] | None = None
        self.peak = 0.0
        self.legend: list[tuple[float, float, str]] = []

        self.image = viewer.add_image(
            np.zeros((2, 2), dtype=np.float32),
            name=f"{_PANEL} {index}",
            colormap=faded_colormap(colormap),
            interpolation2d="nearest",
            blending="translucent_no_depth",
        )
        self.curves = viewer.add_shapes(
            name=f"{_PANEL} {index} curves", shape_type="path", edge_width=2, visible=False
        )
        self.curves.editable = False

    # ------------------------------------------------------------------ state
    @property
    def histogram(self) -> bool:
        """Whether this panel draws distributions rather than a density."""
        return self.kind == "histogram"

    @property
    def title(self) -> str:
        """Heading above the plot: which sample, inside which gate."""
        sample = ", ".join(self.samples) if self.samples else "all samples"
        parent = "ungated" if self.parent == "<none>" else self.parent
        return f"{sample} — {parent}"

    def settings(self) -> dict:
        """The panel's settings, as keyword arguments for a new one."""
        return {
            "kind": self.kind,
            "layer": self.layer,
            "x": self.x,
            "y": self.y,
            "samples": self.samples,
            "parent": self.parent,
        }


class CytoViewer:
    """A napari window showing a 2-channel density plot of an AnnData.

    Parameters
    ----------
    adata
        What to plot. One events x channels AnnData, e.g. from
        :func:`cytopy.read_fcs`; a path to an FCS file, an ``.h5ad`` or a
        directory; or several of either as a list or as a
        ``{name: object}`` mapping. Several are concatenated on the channels
        they share and become entries in the **sample** selector.
    names
        Names for the samples, overriding whatever they carry. One per object.
    layer
        Which matrix to plot: ``None`` for ``adata.X``, otherwise a key of
        ``adata.layers`` (``"comp"``, ``"asinh"``, ``"logicle"``, ...). Its
        values are plotted as-is.
    x, y
        Channels to put on each axis initially. Each may be a ``var_name``, a
        ``$PnN`` detector (``"FITC-A"``) or a ``$PnS`` marker (``"CD3"``).
        Default to the first two channels in the file.
    ticks
        How the axes are labelled, one of :data:`TICK_CHOICES`, and changeable
        in the window afterwards. ``"untransformed"``, the default, labels them
        in the channel's original units when the layer records which transform
        produced it, and falls back to the stored values when it does not.
        ``"transformed"`` always labels the stored values. This is labelling
        only: neither setting moves a point or changes the data.
    bins
        Resolution of the density image, per axis.
    robust
        Clip the axis range to the 0.1-99.9th percentile, so a handful of
        extreme events (common after compensation) cannot flatten the plot.
        Events outside the range are not drawn and fall outside any gate --
        which is why this is off by default: on a large panel 0.1% a side is
        thousands of events, and the rare ones are usually the point. The
        "clip outliers" checkbox turns it on.
    colormap
        Initial colormap for the density image; one of :data:`COLORMAPS`.
    smooth
        Gaussian smoothing of the density, in bins. ``0`` shows raw counts.
    background
        Canvas colour behind the plot, ``"white"`` by default. Any napari
        colour: a name, or ``"#rrggbb"``. Empty bins are transparent, so this
        is what shows through them, and the axis text, ticks and gate outlines
        switch between light and dark to stay legible against it.
    viewer
        An existing ``napari.Viewer`` to build into. A new one is created when
        omitted; pass one to embed the plot in a window you already have.
    title
        Window title, used only when creating a viewer.

    Notes
    -----
    The plot is a 2-D histogram drawn as a napari image layer, so canvas
    coordinates are histogram pixels. :attr:`axes` converts between those and
    the values being plotted.

    Each plot is a :class:`Panel`, owning the napari layer it draws into. That
    layer is rewritten on every redraw, so duplicating it in napari gives a
    frozen snapshot rather than a second plot -- and once the axes move it is a
    snapshot in a coordinate frame that no longer applies. Add a panel instead:
    it keeps its own settings, shares the intensity scale with the others, and
    redraws with them.
    """

    def __init__(
        self,
        adata,
        *,
        names: Sequence[str] | None = None,
        layer: str | None = None,
        x: str | None = None,
        y: str | None = None,
        ticks: str = "untransformed",
        bins: int = 512,
        colormap: str = "turbo",
        smooth: float = 1.0,
        robust: bool = False,
        background: str = DEFAULT_BACKGROUND,
        viewer=None,
        title: str = "cytopy",
    ):
        """Build the layers and the control panel, then draw. See the class docstring."""
        import napari

        adata = as_one_anndata(adata, names=names)
        self.adata = adata
        self.bins = int(bins)
        self.smooth = float(smooth)
        self.panel: Panel | None = None
        self._robust = bool(robust)
        self._derive_colours(background)
        self._updating = False
        # Data extent, and the slider travel derived from it, per channel: the
        # sliders have to hold still between redraws. See :meth:`_travel`.
        self._extents: dict[tuple, tuple[float, float]] = {}
        self._travel_cache: dict[tuple, tuple[float, float]] = {}

        channels = list(adata.var_names)
        if not channels:
            raise ValueError("adata has no channels")
        # Accept a marker ("CD3") or a detector ("FITC-A") as well as a var_name.
        x = channels[find_channel_name(adata, x)] if x else channels[0]
        y = channels[find_channel_name(adata, y)] if y else channels[min(1, len(channels) - 1)]

        self.viewer = viewer if viewer is not None else napari.Viewer(title=title)
        _hide_overlays(self.viewer)

        self._init_layers(colormap)
        self._init_widget(layer, x, y, ticks, colormap, robust)
        self.panel = Panel(
            self.viewer, 1, x=x, y=y, layer="X" if layer is None else layer, colormap=colormap
        )
        self._bind()
        self._raise_overlays()
        self.set_background(self.background)
        self.refresh()
        self.viewer.reset_view()

    # ------------------------------------------------------------------ setup
    def _init_layers(self, colormap: str) -> None:
        """Build the layers every panel shares; panels bring their own."""
        self.grid = self.viewer.add_shapes(
            name=_GRID, shape_type="line", edge_color=self._grid_colour, edge_width=1, opacity=0.6
        )
        self.grid.editable = False
        self.labels = self.viewer.add_points(
            np.empty((0, 2)),
            name=_LABELS,
            size=1,
            face_color="transparent",
            border_color="transparent",
            text={**_TEXT_STYLE, "string": [], "color": self._foreground},
        )
        self.labels.editable = False
        # Its own layer because napari rotates text per layer, not per string,
        # and the y axis reads down the side.
        self.ylabel = self.viewer.add_points(
            np.empty((0, 2)),
            name=_YLABEL,
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
        # Gates that have been applied, drawn where they were drawn and labelled
        # with what they caught. Separate from the editable layer so that the
        # next gate is only the shape drawn for it, not the union of every gate
        # ever applied.
        self.applied = self.viewer.add_shapes(
            name=_APPLIED,
            face_color="transparent",
            edge_color=self._gate_colour,
            edge_width=2,
            opacity=0.9,
        )
        self.applied.editable = False
        self.gates = self.viewer.add_shapes(
            name=_GATES, face_color="#ffcc0022", edge_color="#ffcc00", edge_width=2
        )
        self.viewer.layers.selection = {self.gates}

    def _raise_overlays(self) -> None:
        """Keep the frame, the labels and the gates above every panel.

        Panels are added after the shared layers and napari draws later layers
        on top, so without this a new panel buries the gates you are drawing --
        and a gate you cannot see is a gate you cannot adjust.
        """
        for layer in (
            self.grid,
            self.labels,
            self.ylabel,
            self.applied,
            self.gates,
        ):
            if layer in self.viewer.layers:
                self.viewer.layers.move(self.viewer.layers.index(layer), len(self.viewer.layers))

    # ------------------------------------------------------------------ panels
    @property
    def axes(self) -> Axes2D | None:
        """Axes of the active panel, which is what gates are drawn against."""
        return self.panel.axes

    @property
    def density(self):
        """The active panel's image layer."""
        return self.panel.image

    @property
    def curves(self):
        """The active panel's curve layer."""
        return self.panel.curves

    @property
    def x_scale(self) -> Scale | None:
        """Axis scale of the active panel's x channel."""
        return self.panel.x_scale

    @property
    def y_scale(self) -> Scale | None:
        """Axis scale of the active panel's y channel."""
        return self.panel.y_scale

    def _init_widget(self, layer, x, y, ticks, colormap, robust) -> None:
        from magicgui.widgets import (
            CheckBox,
            ComboBox,
            Container,
            FloatSlider,
            FloatSpinBox,
            Label,
            LineEdit,
            PushButton,
            Select,
            SpinBox,
        )

        names = list(self.adata.var_names)
        layer_choices = ["X"] + [k for k in self.adata.layers if k is not None]
        layer_value = "X" if layer is None else layer
        if layer_value not in layer_choices:
            raise KeyError(f"layer {layer!r} not in {layer_choices}")

        # --- per panel --------------------------------------------------
        self.w_plot = ComboBox(label="plot", choices=PLOT_KINDS, value=PLOT_KINDS[0])
        self.w_layer = ComboBox(
            label="data", choices=lambda w: self._layer_choices(), value=layer_value
        )
        self.w_x = ComboBox(label="x channel", choices=names, value=x)
        self.w_y = ComboBox(label="y channel", choices=names, value=y)
        self.w_swap = PushButton(text="swap x / y")
        self.w_ticks = ComboBox(
            label="axis ticks", choices=TICK_CHOICES, value=_resolve_ticks(ticks)
        )
        self.w_transform = Label(value="")
        # Axis range, as one slider per end -- four of them. Two handles on a
        # single track cannot be told apart once they meet, and the end you
        # wanted is then the one you cannot grab; a slider each also means the
        # whole travel is available to each end.
        #
        # They travel in display coordinates, not the raw units the ticks are
        # labelled in: a logicle axis is 0..1 across, so the travel is spread
        # evenly over the plot, where in raw units nearly all of it would be
        # spent inside the top decade. Each label carries its own raw
        # equivalent, so the numbers beside the ticks are still there to read.
        #
        # tracking=False so the value lands when the handle is let go rather
        # than on every pixel of the drag, which would recompute the density
        # hundreds of times on the way across.
        # The bounds here are placeholders: every redraw sets them, and the
        # values, from where the axis actually is. See `_set_axis_sliders`.
        end = {
            "min": 0.0,
            "max": 1.0,
            "tracking": False,
            "tooltip": "Drag and let go; the plot redraws on release.",
        }
        self.w_xmin = FloatSlider(label="x min", value=0.0, **end)
        self.w_xmax = FloatSlider(label="x max", value=1.0, **end)
        self.w_ymin = FloatSlider(label="y min", value=0.0, **end)
        self.w_ymax = FloatSlider(label="y max", value=1.0, **end)
        self.w_clip = CheckBox(
            label="clip outliers",
            value=bool(robust),
            tooltip=(
                "Fit the axes to the 0.1-99.9th percentile instead of the whole "
                "data. Events outside the range are not drawn and fall outside "
                "any gate."
            ),
        )
        self.w_autoscale = PushButton(text="fit axes to data")
        samples = (
            [str(v) for v in self.adata.obs["sample"].astype(str).unique()]
            if "sample" in self.adata.obs
            else []
        )
        # One list, however many are picked. None picked means all of them, so
        # the plot is never accidentally empty.
        self.w_samples = Select(label="samples", choices=samples, value=[])
        self.w_parent = ComboBox(
            label="parent gate", choices=lambda w: self._gate_choices(), value="<none>"
        )

        # --- global -----------------------------------------------------
        self.w_bins = SpinBox(label="bins", value=self.bins, min=64, max=2048, step=64)
        self.w_smooth = FloatSpinBox(label="smoothing", value=self.smooth, min=0, max=10, step=0.5)
        self.w_log = CheckBox(label="log counts", value=True)
        self.w_cmap = ComboBox(label="colormap", choices=COLORMAPS, value=colormap)
        self.w_background = ComboBox(
            label="background",
            choices=sorted({*BACKGROUNDS, self.background}),
            value=self.background,
        )
        self.w_lock = CheckBox(label="same axes across samples", value=len(samples) > 1)

        self.w_gate_name = LineEdit(label="gate name", value="gate1")
        self.w_gate_apply = PushButton(text="apply gate from shapes")
        self.w_gate_clear = PushButton(text="clear shapes")
        self.w_gate_pick = ComboBox(
            label="edit gate", choices=lambda w: self._gate_choices(), value="<none>"
        )
        self.w_gate_load = PushButton(text="load gate onto canvas")
        self.w_gate_delete = PushButton(text="delete gate")
        self.w_status = Label(value="")

        plot = Container(
            widgets=[
                self.w_plot,
                self.w_layer,
                self.w_x,
                self.w_y,
                self.w_swap,
                self.w_samples,
                self.w_parent,
                self.w_ticks,
                self.w_transform,
            ],
            label="plot",
        )
        limits = Container(
            widgets=[
                self.w_xmin,
                self.w_xmax,
                self.w_ymin,
                self.w_ymax,
                self.w_clip,
                self.w_autoscale,
            ],
            label="axis range",
        )
        style = Container(
            widgets=[
                self.w_bins,
                self.w_smooth,
                self.w_log,
                self.w_lock,
                self.w_cmap,
                self.w_background,
            ],
            label="display",
        )
        gate = Container(
            widgets=[
                self.w_gate_name,
                self.w_gate_apply,
                self.w_gate_clear,
                self.w_gate_pick,
                self.w_gate_load,
                self.w_gate_delete,
            ],
            label="gating",
        )
        sections = [plot, limits, style, gate]
        self.widget = Container(widgets=[*sections, self.w_status])

        # Per-panel settings write to the active panel, then redraw.
        for w in (
            self.w_plot,
            self.w_layer,
            self.w_x,
            self.w_y,
            self.w_samples,
            self.w_parent,
        ):
            w.changed.connect(self._panel_changed)
        for w in (
            self.w_ticks,
            self.w_bins,
            self.w_smooth,
            self.w_log,
            self.w_lock,
        ):
            w.changed.connect(self._on_change)
        self.w_xmin.changed.connect(lambda e: self._limit_slider_changed("x", "min"))
        self.w_xmax.changed.connect(lambda e: self._limit_slider_changed("x", "max"))
        self.w_ymin.changed.connect(lambda e: self._limit_slider_changed("y", "min"))
        self.w_ymax.changed.connect(lambda e: self._limit_slider_changed("y", "max"))
        self.w_clip.changed.connect(self._clip_changed)
        self.w_autoscale.changed.connect(self._autoscale_clicked)
        self.w_cmap.changed.connect(self._on_change)
        self.w_background.changed.connect(lambda e: self.set_background(self.w_background.value))
        self.w_swap.changed.connect(self._swap)
        self.w_gate_apply.changed.connect(self._apply_gate)
        self.w_gate_clear.changed.connect(lambda e: setattr(self.gates, "data", []))
        self.w_gate_load.changed.connect(self._load_gate_clicked)
        self.w_gate_delete.changed.connect(self._delete_gate_clicked)

        self.viewer.window.add_dock_widget(self.widget, area="right", name="cytopy")

    # ------------------------------------------------------------- selection
    @contextmanager
    def _quiet(self):
        """Set widget values without their ``changed`` callbacks firing back."""
        self._updating = True
        try:
            yield
        finally:
            self._updating = False

    def _bind(self) -> None:
        """Point the per-panel widgets at the active panel."""
        panel = self.panel
        with self._quiet():
            self.w_plot.value = panel.kind
            # A transform can add a layer while the window is open, so the
            # list has to be re-read before the value is set against it.
            self.w_layer.reset_choices()
            self.w_layer.value = panel.layer
            self.w_x.value = panel.x
            self.w_y.value = panel.y
            self.w_samples.value = [s for s in panel.samples if s in self.w_samples.choices]
            self.w_parent.value = panel.parent

    def set_plot(self, **settings) -> None:
        """Change what the plot shows, and bring the widgets along.

        Settings live on the plot, not on the widgets: the widgets are a view
        of it. Setting a widget while its callback is suppressed -- which is
        how a redraw is avoided mid-change -- would move the view and leave the
        plot behind, so everything that changes the plot goes through here.

        Parameters
        ----------
        **settings
            Any of ``kind``, ``layer``, ``x``, ``y``, ``sample``, ``parent``.

        Raises
        ------
        AttributeError
            If a name is not one of the plot's settings.
        """
        allowed = set(self.panel.settings())
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise AttributeError(f"not a plot setting: {unknown}; choose from {sorted(allowed)}")
        for key, value in settings.items():
            setattr(
                self.panel,
                key,
                tuple(str(v) for v in value) if key == "samples" else str(value),
            )
        self._bind()
        self.refresh()

    def set_limits(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
    ) -> None:
        """Pin the active panel's axis range, or hand it back to the data.

        Values are in the units the ticks are labelled in -- raw ones on a
        transformed axis, so ``x=(0, 200000)`` means what it says on an FSC
        axis and ``x=(-100, 10000)`` means what it says on an arcsinh one.

        Parameters
        ----------
        x, y
            ``(min, max)`` for that axis, or ``None`` to leave it alone.
            Pass ``(0, 0)`` -- any empty or inverted range -- to put the axis
            back on automatic. A histogram's vertical axis is per cent of mode
            and ignores ``y``.

        Examples
        --------
        >>> cv.set_limits(x=(0, 262144), y=(0, 262144))  # doctest: +SKIP
        >>> cv.set_limits(x=(0, 0))  # back to fitting the data  # doctest: +SKIP
        """
        panel = self.panel
        if x is not None:
            panel.x_lim = _display_range(panel.x_scale, x)
        if y is not None:
            panel.y_lim = _display_range(panel.y_scale, y)
        self.refresh()

    def _axis_sliders(self, axis: str):
        """The ``(min, max)`` pair of sliders for ``"x"`` or ``"y"``."""
        if axis == "x":
            return self.w_xmin, self.w_xmax
        return self.w_ymin, self.w_ymax

    def _limit_slider_changed(self, axis: str, end: str) -> None:
        """Copy one end of one axis onto the active panel, and redraw.

        The sliders are already in display coordinates, so this does not go
        through :meth:`set_limits`, which takes raw units.

        Parameters
        ----------
        axis
            ``"x"`` or ``"y"``.
        end
            ``"min"`` or ``"max"`` -- which slider was moved. The other end is
            the one that gives way when the two are dragged into each other.

        Notes
        -----
        Only the end that moved is read off its slider. The other comes from
        the axis itself, because a slider holds a rounded copy of it -- a
        thousand steps across its travel -- and reading that back would shift
        the end nobody touched by a step every time its neighbour moved.
        """
        if self._updating:
            return
        panel = self.panel
        if axis == "y" and panel.histogram:
            # Per cent of mode, not a channel: nothing for a range to mean.
            return
        lo_w, hi_w = self._axis_sliders(axis)
        lo, hi = self._axis_span(axis)
        if end == "min":
            lo = float(lo_w.value)
        else:
            hi = float(hi_w.value)
        floor = _MIN_AXIS_SPAN * (float(lo_w.max) - float(lo_w.min))
        if hi - lo < floor:
            # Dragged past the other end: push it along rather than leave an
            # inverted range, which reads as "fit the data".
            if end == "min":
                hi = min(float(hi_w.max), lo + floor)
                lo = hi - floor
            else:
                lo = max(float(lo_w.min), hi - floor)
                hi = lo + floor
        setattr(panel, f"{axis}_lim", (lo, hi))
        self.refresh()

    def _axis_span(self, axis: str) -> tuple[float, float]:
        """Where one axis of the active panel currently sits, in display units."""
        axes = self.panel.axes
        if axes is None:
            lo_w, hi_w = self._axis_sliders(axis)
            return (float(lo_w.value), float(hi_w.value))
        if axis == "x":
            return (float(axes.x_lo), float(axes.x_hi))
        return (float(axes.y_lo), float(axes.y_hi))

    def _clip_changed(self, *_) -> None:
        """Turn the percentile clip on or off and refit."""
        if self._updating:
            return
        self._robust = bool(self.w_clip.value)
        self._autoscale_clicked()

    def _autoscale_clicked(self, *_) -> None:
        """Drop both manual ranges and refit to the data."""
        self.panel.x_lim = None
        self.panel.y_lim = None
        # Fitting the axes refits the sliders: the travel a pinned range had
        # widened is no longer anything the data asked for.
        self._forget_travel()
        self.refresh()

    def _forget_travel(self) -> None:
        """Drop the cached slider travel, so the next redraw re-measures it.

        Anything that moves the data under a channel -- a gate applied or
        deleted, a refit -- has to go through here, because the cache is keyed
        by name and those keep their names.
        """
        self._extents.clear()
        self._travel_cache.clear()

    def _show_limits(self) -> None:
        """Write the range actually on screen onto the sliders.

        They are a readout as much as an input: after a channel change or a
        refit they have to show where the axes ended up, or the next drag sends
        the plot somewhere nobody asked for.
        """
        panel = self.panel
        axes = panel.axes
        if axes is None:
            return
        with self._quiet():
            self._set_axis_sliders("x", panel.x_scale, (axes.x_lo, axes.x_hi))
            if panel.histogram:
                top = float(_MODE_TOP)
                self._set_axis_sliders("y", None, (0.0, top), extent=(0.0, top))
            else:
                self._set_axis_sliders("y", panel.y_scale, (axes.y_lo, axes.y_hi))

    def _set_axis_sliders(
        self,
        axis: str,
        scale: Scale | None,
        span: tuple[float, float],
        extent: tuple[float, float] | None = None,
    ) -> None:
        """Point one axis's pair of sliders at ``span``, bounds and labels too.

        The travel is measured against the *data*, not against wherever the
        axis currently sits. Re-deriving it from the view -- which is what this
        used to do -- moved the handles back to the middle of the track after
        every drag, so the same gesture meant something different each time and
        a zoom looked like it had not taken.

        Parameters
        ----------
        axis
            ``"x"`` or ``"y"``.
        scale
            The axis's scale, used only to put raw units in the labels.
            ``None`` leaves them without.
        span
            ``(lo, hi)`` in display coordinates -- where the axis actually is.
        extent
            Where the data lies, in display coordinates, if it is already
            known. Looked up from the channel when omitted.
        """
        lo, hi = float(span[0]), float(span[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = 0.0, 1.0
        floor, ceiling = self._travel(axis, (lo, hi), extent)
        lo_w, hi_w = self._axis_sliders(axis)
        for widget in (lo_w, hi_w):
            # Bounds, then step, then bounds again -- and the bounds before the
            # value, or the value is clamped to the bounds still in force from
            # the previous channel.
            #
            # The second write is not redundant. magicgui keeps a float slider
            # as an integer one scaled by a precision factor, and coarsens that
            # factor to fit a wide range into an int (`FloatSlider.
            # _update_precision`, magicgui 0.10). Setting a step finer than the
            # factor makes it ten times finer again -- but it rescales the
            # bounds only on the min/max path, not on the step one, so the
            # bounds are left reading a tenth of what was asked for. A wide
            # channel such as Time or FSC coarsens the factor; the next logicle
            # channel's step, a thousandth of a much smaller range, then trips
            # it, and the slider refuses the very value the axis is sitting at.
            widget.min, widget.max = floor, ceiling
            widget.step = (ceiling - floor) / _SLIDER_STEPS
            widget.min, widget.max = floor, ceiling
        # Clamped, so that a widget that still disagrees about its own range
        # draws a handle in the wrong place rather than stopping the redraw.
        lo_w.value = float(np.clip(lo, lo_w.min, lo_w.max))
        hi_w.value = float(np.clip(hi, hi_w.min, hi_w.max))
        raw_lo, raw_hi = _raw_range(scale, (lo, hi))
        lo_w.label = f"{axis} min" if scale is None else f"{axis} min  ({_short(raw_lo)})"
        hi_w.label = f"{axis} max" if scale is None else f"{axis} max  ({_short(raw_hi)})"

    def _travel(
        self,
        axis: str,
        span: tuple[float, float],
        extent: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """How far one axis's sliders may be dragged. Cached, and only widens.

        The travel has to come back the same on every redraw or the handles
        creep across the track while the axis stands still -- and a handle that
        moves when its neighbour is dragged looks like the two are tied
        together. So it is worked out once per channel and kept: pinning the
        view somewhere the data is not can only ever widen it, and it goes back
        to the data when the channel, the layer or the selection changes.

        Parameters
        ----------
        axis
            ``"x"`` or ``"y"``.
        span
            ``(lo, hi)`` the axis is showing, which the travel must cover.
        extent
            Where the data lies, if it is already known. Measured from the
            channel when omitted.

        Returns
        -------
        tuple of float
            ``(floor, ceiling)`` in display coordinates.
        """
        panel = self.panel
        channel = getattr(panel, axis)
        locked = bool(self.w_lock.value) and bool(panel.samples)
        key = (
            axis,
            channel,
            panel.layer,
            panel.histogram,
            bool(self._robust),
            () if locked else panel.samples,
            panel.parent,
        )
        if extent is None:
            extent = self._extents.get(key)
        if extent is None:
            mask = self.selection_mask(sample=not locked)
            extent = (
                self._limits(channel, self._column(channel, mask, panel.layer), panel.layer)
                if mask.any()
                else (0.0, 1.0)
            )
        bounds = _slider_bounds(extent, span)
        was = self._travel_cache.get(key)
        if was is not None:
            bounds = (min(bounds[0], was[0]), max(bounds[1], was[1]))
        self._extents[key] = extent
        self._travel_cache[key] = bounds
        return bounds

    def _panel_changed(self, *_) -> None:
        """Copy the per-panel widgets onto the active panel."""
        if self._updating:
            return
        panel = self.panel
        previous_x, previous_y, previous_layer = panel.x, panel.y, panel.layer
        panel.kind = str(self.w_plot.value)
        panel.layer = str(self.w_layer.value)
        panel.x = str(self.w_x.value)
        panel.y = str(self.w_y.value)
        panel.samples = tuple(str(v) for v in self.w_samples.value)
        panel.parent = str(self.w_parent.value)
        # A range typed for one channel means nothing on another, and a layer
        # change moves the units under it too.
        if (panel.x, panel.layer) != (previous_x, previous_layer):
            panel.x_lim = None
        if (panel.y, panel.layer) != (previous_y, previous_layer):
            panel.y_lim = None
        self.refresh()

    # ------------------------------------------------------------ appearance
    def _derive_colours(self, colour: str) -> str:
        """Store the background and the decoration colours that suit it.

        Returns the gate colour, which only ``set_background`` needs.
        """
        from napari.utils.colormaps.standardize_color import transform_color

        transform_color(colour)  # fail here, not three layers later
        self.background = str(colour)
        light = _is_light(colour)
        self._foreground = "#1e1e21" if light else "#f0f1f2"
        self._grid_colour = "#9a9a9a" if light else "#5a5a5a"
        # Applied gates need to read against the canvas too. A pale blue is
        # invisible on white at label size, which is why this follows the
        # background rather than being fixed.
        self._gate_colour = "#0b5fa5" if light else "#7fc7ff"
        return "#0072b2" if light else "#ffcc00"

    def set_background(self, colour: str) -> None:
        """Set the canvas colour, and re-colour the decorations to suit it.

        Parameters
        ----------
        colour
            Any napari colour: a name such as ``"white"``, or ``"#rrggbb"``.
            The axis text, tick lines and gate outlines flip between light and
            dark so they stay legible against it.

        Raises
        ------
        ValueError
            If napari does not recognise the colour.
        """
        gate_edge = self._derive_colours(colour)
        try:
            self.viewer.canvas.background_color_override = self.background
        except AttributeError:  # pragma: no cover - older napari
            canvas = getattr(getattr(self.viewer.window, "_qt_viewer", None), "canvas", None)
            if canvas is not None:
                canvas.bgcolor = self.background

        # Both: `edge_color` recolours the shapes that exist, `current_*` the
        # ones drawn next -- and the grid rebuilds its lines on every redraw.
        self.grid.edge_color = self._grid_colour
        self.grid.current_edge_color = self._grid_colour
        if len(self.applied.data):
            self.applied.edge_color = self._gate_colour
        self.applied.current_edge_color = self._gate_colour
        self.gates.edge_color = gate_edge
        self.gates.face_color = gate_edge + "22"
        self.gates.current_edge_color = gate_edge
        self.gates.current_face_color = gate_edge + "22"
        if self.axes is not None:
            self._draw_axes()

        # ------------------------------------------------------------------- data

    def _layer_choices(self) -> list[str]:
        """``X`` plus every real layer; a transform can add one while the window is open."""
        return ["X"] + [k for k in self.adata.layers if k is not None]

    def _gate_choices(self) -> list[str]:
        bools = [
            c
            for c in self.adata.obs.columns
            if self.adata.obs[c].dtype == bool or self.adata.obs[c].dtype == "boolean"
        ]
        return ["<none>"] + bools

    # ------------------------------------------------------------------ axes
    def _limits(self, channel: str, values: np.ndarray, layer: str) -> tuple[float, float]:
        """Axis range for ``channel``.

        The rule lives in :func:`~cytopy.axis_limits`, so the window and the
        static figures cannot come to disagree about an axis: an untransformed
        channel spans its detector's full ``$PnR``, and everything else spans
        the data, widened by :data:`~cytopy.scales.AXIS_MARGIN` at each end.

        The "clip outliers" checkbox trades that for the 0.1-99.9th percentile,
        which keeps a single extreme event -- and compensation makes those --
        from stretching the axis until everything else is a dot in the corner.
        The cost is that the clipped events are not drawn and fall outside
        every gate, so it is off unless asked for.
        """
        return axis_limits(self.adata, channel, layer, values, robust=self._robust)

    # ------------------------------------------------------------------- data
    def _matrix(self, layer: str):
        return self.adata.X if layer == "X" else self.adata.layers[layer]

    def selection_mask(self, *, sample: bool = True, parent: bool = True) -> np.ndarray:
        """Events the plot is showing: its sample, narrowed by its parent gate.

        Parameters
        ----------
        sample
            Apply the chosen sample. ``False`` keeps every sample, which is
            how the axis limits stay put as you switch between them.
        parent
            Apply the parent gate. ``False`` gives the sample alone, which is
            the scope a *new* gate is allowed to change: re-gating a sample
            has to be able to turn an event off as well as on.

        Returns
        -------
        ndarray
            Boolean mask over all events.
        """
        panel = self.panel
        mask = np.ones(self.adata.n_obs, dtype=bool)
        if sample and panel.samples and "sample" in self.adata.obs:
            labels = self.adata.obs["sample"].astype(str).to_numpy()
            mask &= np.isin(labels, list(panel.samples))
        if parent and panel.parent != "<none>" and panel.parent in self.adata.obs:
            mask &= self.adata.obs[panel.parent].to_numpy(dtype=bool)
        return mask

    def _column(self, channel: str, mask: np.ndarray, layer: str | None = None) -> np.ndarray:
        """The stored values for ``channel``. Never transformed here."""
        layer = self.panel.layer if layer is None else layer
        j = find_channel_name(self.adata, channel)
        col = np.asarray(self._matrix(layer)[:, j], dtype=np.float64).ravel()
        return col[mask]

    def _axis_scale(self, channel: str, layer: str | None = None) -> Scale:
        """How to label the axis for ``channel``. Tick placement only.

        The rule itself lives in :func:`~cytopy.axis_scale`, so the window and
        the static figures cannot come to disagree about an axis. What is the
        window's own is the **axis ticks** setting, which overrides it.
        """
        layer = self.panel.layer if layer is None else layer
        if self.w_ticks.value == "transformed" or not channel:
            return LinearScale()
        return axis_scale(self.adata, channel, layer)

    def _describe_transform(self) -> str:
        layer = self.panel.layer
        info = self.adata.uns.get("cytopy", {})
        if self.w_ticks.value == "transformed":
            return "ticks: transformed values, as stored"
        if layer != "X" and layer == info.get("asinh_layer"):
            return "ticks: untransformed units (arcsinh)"
        if layer != "X" and layer == info.get("logicle_layer"):
            return "ticks: untransformed units (logicle)"
        return "ticks: transformed values (layer records no transform)"

    @property
    def histogram(self) -> bool:
        """Whether the active panel draws distributions rather than a density."""
        return self.panel.histogram

    def histogram_groups(self) -> list[tuple[str, np.ndarray]]:
        """``(name, mask)`` per curve, within the parent gate.

        Which samples get a curve is the panel's own choice; picking none means
        all of them, so the plot is never accidentally empty.


        Returns
        -------
        list of tuple
            Empty when the panel's selection holds no events.
        """
        panel = self.panel
        mask = self.selection_mask(sample=False)
        if not mask.any() or "sample" not in self.adata.obs:
            return [("all", mask)] if mask.any() else []
        labels = self.adata.obs["sample"].astype(str).to_numpy()
        wanted = list(panel.samples) or list(pd.unique(labels[mask]))
        out = []
        for name in wanted:
            group = mask & (labels == name)
            if group.any():
                out.append((str(name), group))
        return out

    def _draw_curves(self, panel: Panel, mask: np.ndarray) -> float:
        """One smoothed distribution of the panel's x channel per sample."""
        ax = panel.axes
        groups = self.histogram_groups()
        curves, colours, legend = [], [], []
        for index, (name, group) in enumerate(groups):
            values = self._column(panel.x, group, panel.layer)
            curves.append(
                density_curve(
                    values,
                    ax.x_lo,
                    ax.x_hi,
                    self.bins,
                    smooth=max(float(self.w_smooth.value), 0.5) * 2,
                )
            )
            colours.append(CURVE_COLOURS[index % len(CURVE_COLOURS)])
            legend.append(f"{name} ({int(group.sum()):,})")

        if not curves:
            panel.curves.data = []
            panel.legend = []
            return 0.0

        # Per cent of mode: each curve scaled so its tallest point is 100.
        # The flow convention, and the only one that lets a rare population be
        # compared with a common one at a glance.
        stacked = np.vstack(curves)
        peaks = stacked.max(axis=1, keepdims=True)
        stacked = stacked / np.where(peaks > 0, peaks, 1.0) * 100.0
        scale = (self.bins - 1) / (_MODE_TOP * 1.05)
        columns = np.arange(self.bins, dtype=float) - 0.5
        floor = self.bins - 1.0

        # Fills first so the lines sit on top of them, and every fill stays
        # translucent enough to read through where curves overlap.
        shapes, kinds, faces, edges = [], [], [], []
        for row, colour in zip(stacked, colours):
            line = np.column_stack([floor - row * scale, columns])
            # The closing edge sits just below the floor. Level with it, every
            # point of a curve that touches zero would lie on the boundary and
            # the polygon would have no area to triangulate.
            base = floor + 1.0
            shapes.append(np.vstack([line, [[base, columns[-1]], [base, columns[0]]]]))
            kinds.append("polygon")
            faces.append(colour + CURVE_FILL_ALPHA)
            edges.append("#00000000")
        for row, colour in zip(stacked, colours):
            shapes.append(np.column_stack([floor - row * scale, columns]))
            kinds.append("path")
            faces.append("#00000000")
            edges.append(colour)

        step = 0.045 * self.bins
        for index, colour in enumerate(colours):
            y = 0.04 * self.bins + index * step
            shapes.append(np.array([[y, 0.70 * self.bins], [y, 0.76 * self.bins]]))
            kinds.append("path")
            faces.append("#00000000")
            edges.append(colour)

        panel.curves.data = []
        panel.curves.add(shapes, shape_type=kinds, face_color=faces, edge_color=edges, edge_width=2)
        panel.legend = [
            (0.04 * self.bins + i * step, 0.78 * self.bins, text) for i, text in enumerate(legend)
        ]
        return _MODE_TOP

    def _status(self) -> str:
        panel = self.panel
        n = int(self.selection_mask().sum())
        if panel.histogram:
            return f"{n:,} events   ({panel.x})   {len(self.histogram_groups())} curve(s)"
        smoothed = " smoothed" if self.w_smooth.value else ""
        peak = f"   peak {self.peak_density():,.0f}{smoothed} events/bin"
        return f"{n:,} events   ({panel.x} × {panel.y}){peak}"

    # -------------------------------------------------------------- rendering
    def refresh(self, *_) -> None:
        """Redraw every panel, and the decorations that go round them."""
        if self._updating:
            return
        self.bins = int(self.w_bins.value)
        self._sync_enabled()
        self.w_transform.value = self._describe_transform()

        drawn = self._shapes_in_data()
        self._draw_panel(self.panel)
        self._restore_shapes(drawn)

        top = self.panel.peak
        if not self.panel.histogram:
            self.panel.image.contrast_limits = (0.0, top or 1.0)

        self._draw_applied_gates()
        self._draw_axes()
        self._show_limits()
        self.w_status.value = self._status()

    def _draw_panel(self, panel: Panel) -> None:
        """Compute and draw one panel's density or curves."""
        mask = self.selection_mask()
        panel.x_scale = self._axis_scale(panel.x, panel.layer)
        panel.y_scale = self._axis_scale(panel.y, panel.layer)
        panel.legend = []
        panel.peak = 0.0

        if not mask.any():
            panel.image.data = np.zeros((self.bins, self.bins), dtype=np.float32)
            panel.curves.data = []
            panel.axes = panel.axes or Axes2D(0.0, 1.0, 0.0, 1.0, bins=self.bins)
            return

        xv = self._column(panel.x, mask, panel.layer)
        scope = mask
        if self.w_lock.value and panel.samples:
            scope = self.selection_mask(sample=False)
        x_lo, x_hi = panel.x_lim or self._limits(
            panel.x, self._column(panel.x, scope, panel.layer), panel.layer
        )
        if panel.histogram:
            # The vertical axis is per cent of mode, not a channel, so there is
            # nothing for a manual range to mean.
            y_lo, y_hi = 0.0, 1.0
        else:
            y_lo, y_hi = panel.y_lim or self._limits(
                panel.y, self._column(panel.y, scope, panel.layer), panel.layer
            )
        panel.axes = Axes2D(x_lo, x_hi, y_lo, y_hi, bins=self.bins)

        if panel.histogram:
            panel.image.visible = False
            panel.curves.visible = True
            panel.peak = self._draw_curves(panel, mask)
            return

        panel.curves.visible = False
        panel.image.visible = True
        image = density_image(
            xv,
            self._column(panel.y, mask, panel.layer),
            panel.axes,
            smooth=float(self.w_smooth.value),
            log=bool(self.w_log.value),
        )
        panel.image.data = image
        panel.image.colormap = faded_colormap(self.w_cmap.value)
        panel.peak = float(image.max())

    def _draw_applied_gates(self) -> None:
        """Outline every gate belonging to a panel's plane, and note its label.

        Applying a gate used to make it vanish, which left no way to see what
        had been drawn or to go back to it. It stays here instead, on a layer
        of its own so that it cannot be mistaken for -- or merged into -- the
        shape being drawn next. The label goes on the text layer rather than on
        the shapes, alongside the axis labels that are known to draw.
        """
        gates = self.adata.uns.get("cytopy", {}).get("gates", {})
        shapes, kinds = [], []
        self._gate_labels: list[tuple[float, float, str]] = []
        panel = self.panel
        if panel.axes is not None and not panel.histogram:
            for name in list(gates):
                record = gate_record(self.adata, name)
                if record.x != panel.x or record.y != panel.y:
                    continue
                if record.layer != panel.layer:
                    continue
                if not record.has_outline():
                    continue
                # Of its own parent, the way a cytometrist reads a hierarchy --
                # not of whatever the panel happens to be showing, which would
                # make the same gate read differently from panel to panel.
                n = int(self.adata.obs[name].sum()) if name in self.adata.obs else 0
                above = record.parent
                total = (
                    int(self.adata.obs[above].sum())
                    if above and above in self.adata.obs
                    else self.adata.n_obs
                ) or 1

                corner = None
                for pixels, kind in self._to_canvas_shapes(record.vertices, record.shape_types):
                    shapes.append(pixels)
                    kinds.append(kind)
                    top_left = (float(pixels[:, 0].min()), float(pixels[:, 1].min()))
                    corner = top_left if corner is None else min(corner, top_left)
                if corner is not None:
                    self._gate_labels.append(
                        (
                            corner[0] - 0.025 * self.bins,
                            corner[1],
                            f"{name}  {100 * n / total:.1f}%",
                        )
                    )

        self.applied.data = []
        if shapes:
            self.applied.add(shapes, shape_type=kinds, edge_color=self._gate_colour, edge_width=2)
        self.applied.visible = bool(shapes)

    def peak_density(self) -> float:
        """Events in the densest bin of the plot, smoothing included.

        The absolute number behind the colour bar's percentages. It depends on
        the bin count as much as on the data, which is why it is reported once
        rather than used to label the bar.

        Returns
        -------
        float
            ``0.0`` before anything has been drawn.
        """
        top = float(self.density.contrast_limits[1])
        return float(np.expm1(top)) if self.w_log.value else top

    def _sync_enabled(self) -> None:
        """Grey out the settings the active panel has no use for."""
        histogram = self.panel.histogram
        for widget in (self.w_y, self.w_swap, self.w_log, self.w_cmap):
            widget.enabled = not histogram
        # A histogram's vertical axis is a fixed per-cent-of-mode scale.
        for widget in (self.w_ymin, self.w_ymax):
            widget.enabled = not histogram

    def _draw_axes(self) -> None:
        """Draw a labelled frame around every panel, and the colour bar's ticks."""
        b = self.bins
        tick = 0.02 * b
        lines: list[np.ndarray] = []
        coords: list[list[float]] = []
        text: list[str] = []
        y_label = ""
        panel = self.panel
        ax = panel.axes
        if ax is not None:
            lines, coords, text, y_label = _panel_frame(
                ax,
                x_scale=panel.x_scale,
                y_scale=panel.y_scale,
                bins=b,
                row=panel.row,
                col=panel.col,
                x_label=str(panel.x),
                y_label=str(panel.y),
                title=panel.title,
                mode_axis=panel.histogram,
            )
            if panel.histogram:
                for row, col, label in panel.legend:
                    coords.append([row, col])
                    text.append(label)

        colours = [self._foreground] * len(text)
        for row, col, label in getattr(self, "_gate_labels", []):
            coords.append([row, col])
            text.append(label)
            colours.append(self._gate_colour)

        self.grid.data = []
        if lines:
            self.grid.add_lines(lines)
        # Assign the strings directly rather than through a feature column: an
        # encoding is briefly out of step with the new point count and napari
        # silently falls back to blank labels.
        self.labels.data = np.asarray(coords, dtype=float) if coords else np.empty((0, 2))
        self.labels.text = {**_TEXT_STYLE, "string": text, "color": colours or self._foreground}
        self.ylabel.data = np.array([[b / 2.0, -12.0 * tick]], dtype=float)
        self.ylabel.text = {
            **_TEXT_STYLE,
            "string": [y_label],
            "color": self._foreground,
            "rotation": Y_LABEL_ROTATION,
        }

    # --------------------------------------------------------------- callbacks
    def _on_change(self, *_):
        if self._updating:
            return
        self.refresh()

    def _swap(self, *_):
        self.set_plot(x=self.panel.y, y=self.panel.x)

    # ------------------------------------------------------------------ gating
    def to_display(self, shape) -> np.ndarray:
        """Canvas coordinates of a shape, in the values the panel is plotting.

        Parameters
        ----------
        shape
            ``(n, 2)`` vertices in canvas coordinates.
        panel
            Which panel's axes to read them against; the active one by default.

        Returns
        -------
        ndarray
            ``(n, 2)`` of ``(x, y)`` data values.
        """
        verts = np.asarray(shape, dtype=float)
        return self.panel.axes.to_display(verts)

    def to_canvas(self, x, y) -> np.ndarray:
        """Where data values land on the canvas.

        Parameters
        ----------
        x, y
            Data values.
        panel
            Which panel; the active one by default.

        Returns
        -------
        ndarray
            ``(n, 2)`` of ``(row, col)`` canvas coordinates.
        """
        return self.panel.axes.to_pixels(np.asarray(x), np.asarray(y))

    def current_gate_mask(self) -> np.ndarray:
        """Mask over *all* events of those inside the shapes drawn on the canvas.

        Worked out in data coordinates rather than pixels, so a rescale of
        the axes cannot change which events a drawn shape holds.

        On a histogram there is no second channel to be inside of, so a shape
        gates the interval it spans on the x axis -- drag a box over a peak and
        you get the events under it, which is how a threshold is set.

        Returns
        -------
        ndarray
            Boolean mask over all events, all ``False`` when nothing is drawn.
        """
        out = np.zeros(self.adata.n_obs, dtype=bool)
        if self.axes is None or not len(self.gates.data):
            return out
        sel = self.selection_mask()
        x = self._column(self.w_x.value, sel)
        shapes = [self.to_display(shape) for shape in self.gates.data]

        if self.panel.histogram:
            inside = np.zeros(x.shape[0], dtype=bool)
            for verts in shapes:
                inside |= (x >= verts[:, 0].min()) & (x <= verts[:, 0].max())
        else:
            points = np.column_stack([x, self._column(self.w_y.value, sel)])
            inside = shapes_mask(points, shapes, [str(k) for k in self.gates.shape_type])

        out[np.flatnonzero(sel)] = inside
        return out

    def _shapes_in_data(self):
        """The drawn shapes in data coordinates, before the axes move under them."""
        if self.axes is None or not len(self.gates.data):
            return None
        return (
            [self.to_display(v) for v in self.gates.data],
            [str(k) for k in self.gates.shape_type],
        )

    def _to_canvas_shapes(self, shapes, kinds):
        """Outlines in data coordinates, as canvas pixels ready for a Shapes layer."""
        out = []
        for verts, kind in zip(shapes, kinds):
            verts = np.asarray(verts, dtype=float)
            out.append((self.to_canvas(verts[:, 0], verts[:, 1]), str(kind)))
        return out

    def _put_on_gates_layer(self, shapes, kinds) -> None:
        """Replace whatever is on the editable gates layer with these outlines."""
        self.gates.data = []
        for pixels, kind in self._to_canvas_shapes(shapes, kinds):
            self.gates.add(pixels, shape_type=kind)

    def _restore_shapes(self, drawn) -> None:
        """Redraw shapes so they keep covering the same data after a rescale.

        A shape lives on the canvas in pixels, but it means a region of the
        data. Changing channel, sample, parent gate or bin count moves the
        axes underneath it -- and moving the panel moves the whole box -- so
        without this the shape would silently come to mean something else.
        """
        if drawn is None or self.axes is None:
            return
        self._put_on_gates_layer(*drawn)

    def off_axis_count(self) -> int:
        """Events in the current selection that fall outside the plotted axes.

        Robust limits clip the axes to the 0.1-99.9th percentile, so a few
        events are drawn nowhere. They cannot be gated either, which is worth
        knowing when a gate around everything visible comes back smaller than
        the population it was drawn on.

        Returns
        -------
        int
            Number of selected events outside the axis range, ``0`` when the
            plot has not been drawn yet.
        """
        if self.axes is None:
            return 0
        sel = self.selection_mask()
        x = self._column(self.w_x.value, sel)
        y = self._column(self.w_y.value, sel)
        inside = (
            (x >= self.axes.x_lo)
            & (x <= self.axes.x_hi)
            & (y >= self.axes.y_lo)
            & (y <= self.axes.y_hi)
        )
        return int((~inside).sum())

    def apply_gate(self, name: str, *, parent: str | None = None) -> np.ndarray:
        """Turn the shapes on the canvas into a named gate.

        What the *apply gate from shapes* button does, callable directly.

        Parameters
        ----------
        name
            ``obs`` column to write. An existing gate of the same name is
            overwritten.
        parent
            Gate to nest inside, so the new one only ever contains events the
            parent already held. ``None`` gates the current selection.

        Returns
        -------
        ndarray
            Boolean mask over all events.

        Raises
        ------
        ValueError
            If nothing has been drawn on the canvas.
        """
        if not len(self.gates.data):
            raise ValueError("draw a shape on the canvas first")
        with self._quiet():
            self.w_gate_name.value = name
            self.w_parent.value = parent or "<none>"
        self.refresh()
        self._apply_gate()
        return self.adata.obs[name].to_numpy(dtype=bool)

    def load_gate(self, name: str) -> None:
        """Put a gate back on the canvas, in the plane it was drawn in, to adjust.

        The active panel switches to the gate's plot kind, channels, layer and
        parent, the outline reappears on the **gates** layer, and the name box
        is filled in -- so adjusting it and hitting *apply* replaces the gate
        rather than adding another. Its children are recomputed when you do.
        A gate recorded before this switched panel kind back to density
        defaults to density too, since nothing was saved to tell them apart.

        Parameters
        ----------
        name
            The gate to load.

        Raises
        ------
        KeyError
            If the gate was never recorded, or recorded no outline: one
            applied from a mask rather than drawn cannot be put back.
        """
        record = gate_record(self.adata, name)
        if not record.vertices:
            raise KeyError(f"gate {name!r} recorded no outline, so there is nothing to adjust")

        panel = self.panel
        panel.x = record.x or panel.x
        panel.y = record.y or panel.y
        panel.layer = record.layer
        # Its own parent, not itself: a gate cannot be nested inside itself.
        panel.parent = record.parent or "<none>"
        panel.kind = record.kind
        self._bind()
        self.refresh()

        self._put_on_gates_layer(record.vertices, record.shape_types)
        with self._quiet():
            self.w_gate_name.value = name
            self.w_gate_pick.value = name
        self.w_status.value = f"{name} loaded: adjust it, then apply to replace it"

    def delete_gate(self, name: str) -> list[str]:
        """Remove a gate, and any gates nested inside it.

        Parameters
        ----------
        name
            The gate to remove.

        Returns
        -------
        list of str
            Everything removed, the gate itself first.

        Raises
        ------
        KeyError
            If the gate was never recorded.
        """
        gate_record(self.adata, name)  # raises, listing what is recorded
        gates = self.adata.uns.get("cytopy", {}).get("gates", {})
        gone = [name]
        for child in gate_children(self.adata, name):
            gone += self.delete_gate(child)
        gates.pop(name, None)
        self.adata.obs.drop(columns=[name], inplace=True, errors="ignore")
        self._forget_travel()
        if self.panel.parent == name:
            self.panel.parent = "<none>"
        self._refresh_gate_choices()
        self._bind()
        self.refresh()
        return gone

    def _refresh_gate_choices(self) -> None:
        with self._quiet():
            self.w_parent.reset_choices()
            self.w_gate_pick.reset_choices()

    def _load_gate_clicked(self, *_) -> None:
        name = str(self.w_gate_pick.value)
        if name == "<none>":
            self.w_status.value = "pick a gate to edit first"
            return
        try:
            self.load_gate(name)
        except KeyError as exc:
            self.w_status.value = str(exc).strip("'")

    def _delete_gate_clicked(self, *_) -> None:
        name = str(self.w_gate_pick.value)
        if name == "<none>":
            self.w_status.value = "pick a gate to delete first"
            return
        gone = self.delete_gate(name)
        self.w_status.value = f"deleted {', '.join(gone)}"

    def _apply_gate(self, *_):
        name = (self.w_gate_name.value or "").strip()
        if not name:
            self.w_status.value = "give the gate a name first"
            return
        if not len(self.gates.data):
            self.w_status.value = "draw a shape on the canvas first"
            return
        parent = None if self.w_parent.value == "<none>" else self.w_parent.value
        if parent == name:
            self.w_status.value = "a gate cannot be its own parent"
            return
        replacing = name in self.adata.uns.get("cytopy", {}).get("gates", {})
        mask = add_gate(
            self.adata,
            name,
            self.current_gate_mask(),
            parent=parent,
            # Only the samples on screen: gating one file does not undo the
            # gate drawn on another under the same name.
            within=self.selection_mask(parent=False),
            meta={
                "kind": self.panel.kind,
                "x": str(self.w_x.value),
                "y": str(self.w_y.value),
                "layer": str(self.w_layer.value),
                # In data coordinates, so the gate can be redrawn in a figure
                # long after the canvas it was drawn on is gone -- and put back
                # on the canvas to adjust.
                "vertices": [self.to_display(shape).tolist() for shape in self.gates.data],
                "shape_types": [str(k) for k in self.gates.shape_type],
            },
        )
        self._forget_travel()
        n = int(mask.sum())
        denom = int(self.selection_mask().sum()) or 1
        note = ""
        # Children were worked out against the old outline, so they are stale
        # until they are recomputed against the new one.
        updated = recompute_gates(self.adata, name) if replacing else []
        if updated:
            note = f"; recomputed {', '.join(updated)}"
        off = self.off_axis_count()
        if off:
            note += f"; {off:,} outside the axes, not gated"
        # The shape moves to the applied layer, where it stays visible and
        # labelled. Clearing the editable one is what keeps the next gate to
        # the shape drawn for it, rather than the union of everything applied.
        self.gates.data = []
        self._draw_applied_gates()
        self._draw_axes()
        verb = "replaced" if replacing else name
        self.w_status.value = f"{verb}: {n:,} events ({100 * n / denom:.2f}% of parent){note}"
        self._refresh_gate_choices()


def as_one_anndata(data, *, names: Sequence[str] | None = None) -> ad.AnnData:
    """Gather whatever was passed into a single AnnData with a ``sample`` column.

    The viewer plots one matrix; several files become several values of
    ``obs['sample']``, which the **sample** selector then switches between.

    Parameters
    ----------
    data
        An ``AnnData``; a path to an FCS file, an ``.h5ad`` or a directory of
        FCS files; a sequence of any of those; or a mapping from the name you
        want in the selector to any of those.
    names
        Names for the samples, overriding whatever they carry. Must be one per
        object.

    Returns
    -------
    AnnData
        A single object. One input is returned untouched; several are
        concatenated on the channels they share, with duplicate sample names
        made unique so two files never merge into one entry.

    Raises
    ------
    ValueError
        If nothing was passed, or ``names`` is the wrong length.
    TypeError
        If an entry is neither an AnnData nor a path.
    """
    from .io import concat_samples

    if isinstance(data, ad.AnnData):
        parts, labels = [data], list(names) if names else [None]
    elif isinstance(data, Mapping):
        parts, labels = list(data.values()), [str(k) for k in data]
        if names:
            labels = list(names)
    elif isinstance(data, str | os.PathLike):
        parts, labels = [data], list(names) if names else [None]
    else:
        parts = list(data)
        labels = list(names) if names else [None] * len(parts)

    if not parts:
        raise ValueError("nothing to view")
    if len(labels) != len(parts):
        raise ValueError(f"{len(parts)} objects but {len(labels)} names")

    loaded = [_load(part) for part in parts]
    if len(loaded) == 1 and labels[0] is None:
        return loaded[0]

    seen: dict[str, int] = {}
    for adata, label in zip(loaded, labels):
        name = label if label is not None else _sample_name(adata)
        # Two files called the same thing must not collapse into one entry.
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        adata.obs["sample"] = pd.Categorical([name] * adata.n_obs)
    return concat_samples(loaded) if len(loaded) > 1 else loaded[0]


def _load(part):
    """An AnnData, read from disk if a path was given."""
    if isinstance(part, ad.AnnData):
        return part
    if not isinstance(part, str | os.PathLike):
        raise TypeError(f"expected an AnnData or a path, got {type(part).__name__}")
    from .io import read_fcs, read_fcs_dir

    path = Path(part)
    if path.is_dir():
        return read_fcs_dir(path)
    if path.suffix == ".h5ad":
        return ad.read_h5ad(path)
    return read_fcs(path)


def _sample_name(adata: ad.AnnData) -> str:
    """Whatever this object already calls itself."""
    if "sample" in adata.obs and adata.n_obs:
        return str(adata.obs["sample"].astype(str).iloc[0])
    return "sample"


def _is_light(colour) -> bool:
    """Whether text should be drawn dark on this background."""
    from napari.utils.colormaps.standardize_color import transform_color

    red, green, blue = transform_color(colour)[0][:3]
    return float(0.2126 * red + 0.7152 * green + 0.0722 * blue) > 0.5


def _slider_bounds(extent: tuple[float, float], span: tuple[float, float]) -> tuple[float, float]:
    """How far an axis's sliders may travel, in display coordinates.

    Far enough to cover the data and wherever the axis has been pinned, plus
    :data:`_SLIDER_OVERSHOOT` spans either side so an outlier can be framed
    with room around it -- and no further, or the whole travel is spent on
    empty canvas.

    Parameters
    ----------
    extent
        ``(lo, hi)`` of the data on this axis.
    span
        ``(lo, hi)`` the axis is showing, which may be outside the data.
    """
    lo, hi = float(extent[0]), float(extent[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = float(span[0]), float(span[1])
    lo, hi = min(lo, float(span[0])), max(hi, float(span[1]))
    room = _SLIDER_OVERSHOOT * (hi - lo) or 0.5
    return (lo - room, hi + room)


def _short(value: float) -> str:
    """A raw axis bound, short enough to sit in a widget label."""
    if not np.isfinite(value):
        return "?"
    if abs(value) >= 1e5:
        return f"{value:.3g}"
    if abs(value) >= 10:
        return f"{value:,.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _raw_range(scale: Scale | None, span: tuple[float, float]) -> tuple[float, float]:
    """A display-coordinate range in the units the ticks are labelled in."""
    if scale is None:
        return (float(span[0]), float(span[1]))
    lo, hi = scale.to_raw(np.asarray(span, dtype=float))
    return (float(lo), float(hi))


def _display_range(scale: Scale | None, span: tuple[float, float]) -> tuple[float, float] | None:
    """A typed ``(min, max)`` in display coordinates, or ``None`` for automatic.

    An empty, inverted or non-finite range is how the window says "fit the
    data": there is no useful axis it could mean, and a field cleared to zero
    should not leave the plot stuck on a sliver.
    """
    lo, hi = float(span[0]), float(span[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    if scale is not None:
        lo, hi = (float(v) for v in scale.from_raw(np.asarray([lo, hi], dtype=float)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return (lo, hi)


def _panel_frame(
    axes: Axes2D,
    *,
    x_scale: Scale,
    y_scale: Scale | None,
    bins: int,
    row: float = 0.0,
    col: float = 0.0,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    mode_axis: bool = False,
) -> tuple[list[np.ndarray], list[list[float]], list[str], str]:
    """The labelled box around one panel, in canvas coordinates.

    Pure geometry, so both windows draw an axis the same way and cannot come to
    disagree about where a tick goes. ``row`` and ``col`` offset the whole frame,
    which is what lets several panels share a canvas.

    Parameters
    ----------
    axes
        Extent of the panel, mapping display coordinates to pixels.
    x_scale
        Scale placing the ticks along the bottom.
    y_scale
        Scale placing the ticks up the side. Unused when ``mode_axis`` is set.
    bins
        Side of the panel, in pixels.
    row, col
        Where the panel's top-left corner sits, in the same pixel units.
    x_label, y_label
        Axis names. ``y_label`` is ignored when ``mode_axis`` is set.
    title
        Heading drawn above the panel.
    mode_axis
        Label the vertical axis as per cent of mode rather than as a channel,
        which is what a histogram panel wants.

    Returns
    -------
    tuple
        ``(lines, coords, text, y_label)`` -- the frame and tick segments, the
        label positions, the label strings, and the name for the rotated y-axis
        label, which its caller draws on a layer of its own.
    """
    tick = 0.02 * bins
    lines: list[np.ndarray] = []
    coords: list[list[float]] = []
    text: list[str] = []
    dr, dc = float(row), float(col)
    b = bins

    lines += [
        np.array([[dr - 0.5, dc - 0.5], [dr - 0.5, dc + b - 0.5]]),
        np.array([[dr + b - 0.5, dc - 0.5], [dr + b - 0.5, dc + b - 0.5]]),
        np.array([[dr - 0.5, dc - 0.5], [dr + b - 0.5, dc - 0.5]]),
        np.array([[dr - 0.5, dc + b - 0.5], [dr + b - 0.5, dc + b - 0.5]]),
    ]
    xt = x_scale.ticks(axes.x_lo, axes.x_hi)
    for pos, lab in zip(xt.major, xt.labels):
        c = dc + axes.to_pixels(np.array([pos]), np.array([axes.y_lo]))[0, 1]
        lines.append(np.array([[dr + b - 0.5, c], [dr + b - 0.5 + tick, c]]))
        coords.append([dr + b + 3.0 * tick, c])
        text.append(lab)
    for pos in xt.minor:
        c = dc + axes.to_pixels(np.array([pos]), np.array([axes.y_lo]))[0, 1]
        lines.append(np.array([[dr + b - 0.5, c], [dr + b - 0.5 + tick / 2, c]]))
    coords.append([dr + b + 8.0 * tick, dc + b / 2.0])
    text.append(x_label)
    coords.append([dr - 3.5 * tick, dc + b / 2.0])
    text.append(title)

    if mode_axis:
        # The y axis is a density, not a channel.
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = dr + (b - 1) - fraction * (b - 1) / 1.05
            lines.append(np.array([[r, dc - 0.5], [r, dc - 0.5 - tick]]))
            coords.append([r, dc - 4.0 * tick])
            text.append(f"{round(fraction * _MODE_TOP)}")
        return lines, coords, text, "% of mode"

    yt = y_scale.ticks(axes.y_lo, axes.y_hi)
    for pos, lab in zip(yt.major, yt.labels):
        r = dr + axes.to_pixels(np.array([axes.x_lo]), np.array([pos]))[0, 0]
        lines.append(np.array([[r, dc - 0.5], [r, dc - 0.5 - tick]]))
        coords.append([r, dc - 4.0 * tick])
        text.append(lab)
    for pos in yt.minor:
        r = dr + axes.to_pixels(np.array([axes.x_lo]), np.array([pos]))[0, 0]
        lines.append(np.array([[r, dc - 0.5], [r, dc - 0.5 - tick / 2]]))
    return lines, coords, text, y_label


def faded_colormap(name: str, fade: float = _FADE):
    """A colormap whose bottom end is transparent rather than its darkest colour.

    The density image covers the whole plot, so without this the background is
    whatever the colormap maps zero counts to -- for ``turbo`` that is a dark
    blue-purple, which is what makes an untouched napari canvas look blue. With
    the low end faded, empty bins show the canvas through instead.

    Parameters
    ----------
    name
        Any colormap napari knows, e.g. one of :data:`COLORMAPS`.
    fade
        Fraction of the colormap that ramps from transparent to opaque. Small:
        a bin holding a single event should still be clearly visible.

    Returns
    -------
    napari.utils.colormaps.Colormap
        The same colours, with alpha ramped in at the bottom.
    """
    import numpy as np
    from napari.utils.colormaps import Colormap, ensure_colormap

    colours = np.array(ensure_colormap(name).colors, dtype=np.float32, copy=True)
    steps = max(round(len(colours) * float(fade)), 2)
    colours[:steps, 3] = np.linspace(0.0, 1.0, steps)
    return Colormap(colors=colours, name=f"{name}_faded", display_name=name)


def _hide_overlays(viewer) -> None:
    """Turn off napari's own axis cross and scale bar, across napari versions."""
    for path in (("scene", "overlays", "axes"), ("canvas", "overlays", "scale_bar")):
        target = viewer
        for attr in path:
            target = getattr(target, attr, None)
            if target is None:
                break
        if target is not None:
            target.visible = False
            continue
        legacy = getattr(viewer, path[-1], None)  # napari < 0.9
        if legacy is not None:
            legacy.visible = False


#: The window this session last opened, so a non-blocking call does not let it
#: be collected and ``current_viewer`` has something to hand back.
_CURRENT: CytoViewer | None = None


def current_viewer() -> CytoViewer | None:
    """The :class:`CytoViewer` behind the window :func:`open_napari` last opened.

    Reach for this only to drive the window from code -- changing the plot,
    applying a gate by name, asking what is selected. The gates themselves land
    on the AnnData that :func:`open_napari` returns, so ordinary use never
    needs it.

    Returns
    -------
    CytoViewer or None
        ``None`` before a window has been opened in this session.
    """
    return _CURRENT


def open_napari(
    adata,
    layer: str = "raw",
    *,
    x: str | None = None,
    y: str | None = None,
    samples: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
    ticks: str = "untransformed",
    block: bool | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Open the cytopy window on some data, and hand the data back when you close it.

    One window for both jobs. Look at the data, or draw gates on it, or both --
    the window is the same either way, which is why this is not called
    ``view`` or ``gate``.

    Each gate you apply becomes a boolean column in ``adata.obs``, with its
    outline recorded in ``adata.uns['cytopy']['gates']``. **Gates already on
    the data are loaded**, outlined where you drew them and offered as parents,
    so gating is something you come back to rather than do in one sitting.

    The bin count, colour map and background are chosen in the window, so they
    are not arguments here. ``ticks`` is, because which units you want to read
    is usually decided before you open anything -- but it is a starting point,
    and the **axis ticks** box owns it afterwards. The channels and the sample are,
    because knowing what you want to look at before you open it is common
    enough to be worth saving the clicks -- but they are only a starting
    point, and the window owns them afterwards.

    Parameters
    ----------
    adata
        What to open. One events x channels AnnData, e.g. from
        :func:`cytopy.read_fcs`; a path to an FCS file, an ``.h5ad`` or a
        directory of FCS files; or several of either as a list or a
        ``{name: object}`` mapping, concatenated on the channels they share.
        Modified in place.
    layer
        Layer to plot first, by name. Plotted exactly as stored -- run the
        transform you want before opening. You can change it in the window.
    x, y
        Channels to put on the axes to begin with, by marker or detector.
        ``None`` leaves the window on the first two channels.
    samples
        Samples to show to begin with, as they appear in ``obs['sample']``.
        ``None`` shows every one of them.
    names
        Names for the samples, one per object, overriding whatever they carry.
    ticks
        How the axes are labelled to begin with, one of :data:`TICK_CHOICES`.
        ``"untransformed"``, the default, labels them in the channel's original
        units when the layer records which transform produced it -- which is
        what turns an arcsinh layer into an axis that reads as biexponential.
        ``"transformed"`` labels the numbers actually stored in the layer. The
        **axis ticks** box changes it in the window.
    block
        Wait for the window to close before returning. The default detects the
        context: ``True`` from a script, ``False`` under IPython, where a Qt
        loop is already running and blocking would wedge the kernel.

        A non-blocking call returns **before you have drawn anything**. The
        data is modified in place, so the gates appear on the object you were
        handed as you draw them, but it is not gated at the moment it returns.
    verbose
        Print which gates were loaded and which were drawn.

    Returns
    -------
    AnnData
        The data, with whatever you gated on it. The window itself, if you
        need to drive it from code, is :func:`current_viewer`.
    """
    global _CURRENT
    import napari

    adata = as_one_anndata(adata, names=names)
    before = list(adata.uns.get("cytopy", {}).get("gates", {}))
    if verbose and before:
        print(f"loaded {len(before)} gate(s): {', '.join(before)}")

    _CURRENT = CytoViewer(adata, layer=layer, x=x, y=y, ticks=ticks)
    if samples is not None:
        _CURRENT.set_plot(samples=samples)

    if block is None:
        try:
            get_ipython  # type: ignore[name-defined]  # noqa: B018
            block = False
        except NameError:
            block = True
    if block:
        napari.run()

    if verbose:
        after = adata.uns.get("cytopy", {}).get("gates", {})
        drawn = [g for g in after if g not in before]
        print(f"gated: {', '.join(drawn) if drawn else 'nothing new'}")
    return adata
