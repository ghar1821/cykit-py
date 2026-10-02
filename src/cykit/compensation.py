"""The compensation window: gate the controls, compute a matrix, then tune it by eye.

Two steps, one window:

* **gate controls** -- one single-stain control as a distribution of its own
  detector, with the unstained tube overlaid in another colour as a reference
  for where unstained events sit. Drag an interval for the positive population
  and one for the negative. Both are gates on the **control**: the unstained is
  only there to look at, is never gated, and plays no part in the matrix.
  Gates land on the AnnData under the names
  :func:`~cykit.compute_spillover_matrix` looks for, so the matrix can be
  recomputed later without the window.
* **check compensation** -- the N x N grid of biaxial plots: one row per
  single-stain control, one column per detector, the control's own detector
  on x and the column's on y, all compensated with the matrix in the table.
  Tile ``(i, j)`` is the picture of coefficient ``S[i, j]``. Beside it, the same
  rows against a scatter channel of your choice: detector ``j`` on x, scatter
  on y. Or switch to two large biaxial plots of whatever you pick. Edit a cell
  and everything redraws: a control compensated correctly has its positive
  population level with its negative one in every detector but its own.

When the controls carry the ``$SPILLOVER`` the instrument wrote, that matrix is
loaded into the table and the window opens on the second step, so a matrix
you already have can be tuned without gating anything.

Nothing here compensates the data. What comes back is the matrix; applying it
is your own :func:`~cykit.compensate` call.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager

import anndata as ad
import numpy as np
import pandas as pd

from ._util import layer_matrix, subsample_indices
from .density import Axes2D, density_curve, density_image
from .gating import add_gate, gate_record
from .plotting import axis_limits, axis_scale
from .scales import AsinhScale, PretransformedScale
from .spillover import _resolve_detectors, compute_spillover_matrix
from .transforms import find_channel_name

__all__ = [
    "CompensationWindow",
    "current_compensation_window",
    "open_napari_compensation",
]

#: The window's two steps, in the order they are done.
STEPS = ["gate controls", "check compensation"]

#: What a drawn interval becomes: ``(label, value)`` for the **draw** box. Both
#: are gates on the control; the unstained is never gated.
TARGETS = [("positive", "positive"), ("negative", "negative")]

#: Events plotted per tube. Every gate is applied to every event regardless.
MAX_EVENTS = 50_000

#: Space between tiles of the check grid, as a fraction of a tile's width.
TILE_GAP = 0.12

#: How the check step shows the compensated controls.
VIEWS = ["grid", "two plots"]

#: What one press of the table's arrow buttons adds to a coefficient, to begin with.
DEFAULT_STEP = 0.001

#: Fewest bins a plot of the "two plots" view can be set to.
MIN_BINS = 16

#: Most bins a plot of the "two plots" view can be set to. The plots are drawn
#: one screen pixel per bin of the finer of the two, so this also caps how big
#: the canvas gets.
MAX_BINS = 1024

#: The **scatter y** choice that hides the scatter grid.
NO_SCATTER = "<none>"

_RANGE_TOOLTIP = (
    "Min and max, in the units the axis is ticked in. Press Enter to apply; "
    "clear a box to put that end back on automatic."
)

#: Space between the two big plots of the "two plots" view, as a fraction of
#: a plot's width: room for the right one's y-axis labels.
GUTTER = 0.45

_CURVE = "#4c78a8"
_POSITIVE = "#2ca02c"
_NEGATIVE = "#d62728"
_EDITED = "#fff2a8"
_FOCUS = "#e4572e"
_DIAGONAL = "#e6e6e6"
_CELL_TEXT = "#1e1e21"
_CELL_BOX_WIDTH = 60
_CELL_BUTTON_WIDTH = 18
_CELL_WIDTH = 104


class CompensationWindow:
    """A napari window for gating single-stain controls and tuning a spillover matrix.

    Parameters
    ----------
    controls
        Maps detector to its single-stain control, exactly as
        :func:`~cykit.compute_spillover_matrix` takes it -- ``{"FITC-A":
        adata}``, with markers accepted as keys. Must be ``AnnData`` rather than
        paths, because the gates are written onto them. Usually already cut
        down to singlets with :func:`~cykit.subset_controls`.
    unstained
        The unstained control, overlaid on the gating plot as a reference for
        where unstained events sit. Display only: it is never gated and never
        used to compute the matrix.
    layer
        Layer to **display**, by name -- a transformed one, e.g. ``"asinh"``.
        Gates are drawn on it. The check step re-applies the transform this
        layer recorded to the freshly compensated values, so the two steps read
        on the same axes; a layer that records no transform is plotted linear.
    spillover
        The matrix the table starts from:

        * ``"file"`` (default) -- the ``$SPILLOVER`` the controls carry in
          ``uns["spillover"]``. Controls carrying different matrices, or one
          whose diagonal is not ``1``, are an error rather than a guess. None
          carrying one starts from the identity.
        * a DataFrame -- e.g. one you read from a CSV.
        * ``None`` -- the identity.

        Either way it is cut down to the detectors there are controls for, so
        the matrix is always square over exactly those: a detector nobody
        stained is dropped rather than inverted. A loaded matrix (file or
        DataFrame) opens the window on the check step.
    raw_layer
        Layer the matrix is computed from and applied to. ``"raw"`` by
        default: spillover is linear in the values as acquired.
    positive_gate, negative_gate
        Names of the two gates written on each control.
    statistic
        ``"median"`` or ``"mean"``, passed to
        :func:`~cykit.compute_spillover_matrix`.
    max_events
        Events plotted per tube. ``None`` plots every one. Gates and the
        matrix always use every event.
    seed
        Seed for that draw.
    bins
        Resolution of the gating plot, per axis.
    tile_bins
        Resolution of each tile of the check grid, per axis. The grid is
        ``n_controls x n_detectors`` tiles, so this is what keeps a 30-colour
        panel responsive.
    smooth
        Gaussian smoothing, in bins.
    colormap
        Colormap for the check grid.
    background
        Canvas colour behind the plots.
    viewer
        An existing ``napari.Viewer`` to build into. One is created when omitted.
    title
        Window title, used only when creating a viewer.

    Attributes
    ----------
    spillover
        The matrix, as a DataFrame that is edited **in place** -- by the table,
        :meth:`set_coefficient`, :meth:`compute` and :meth:`reset` -- so a
        reference taken before any of those keeps seeing the current values.
    source
        Where the matrix came from: ``"file $SPILLOVER"``, ``"argument"``,
        ``"identity"`` or ``"computed"``.
    """

    def __init__(
        self,
        controls: Mapping[str, ad.AnnData],
        unstained: ad.AnnData,
        layer: str,
        *,
        spillover="file",
        raw_layer: str = "raw",
        positive_gate: str = "positive",
        negative_gate: str = "negative",
        statistic: str = "median",
        max_events: int | None = MAX_EVENTS,
        seed: int = 0,
        bins: int = 256,
        tile_bins: int = 64,
        smooth: float = 1.0,
        colormap: str = "turbo",
        background: str | None = None,
        viewer=None,
        title: str = "cykit compensation",
    ):
        """Resolve the controls and the starting matrix, build the panels, draw."""
        import napari

        from .viewer import DEFAULT_BACKGROUND, Panel, _hide_overlays, _is_light

        if not controls:
            raise ValueError("no controls given")
        for key, value in controls.items():
            if not isinstance(value, ad.AnnData):
                raise TypeError(
                    f"control {key!r} is a {type(value).__name__}; read it with "
                    "cykit.read_fcs first, so the gates drawn here have somewhere to go"
                )
        if not isinstance(unstained, ad.AnnData):
            raise TypeError("unstained must be an AnnData")

        primary, names = _resolve_detectors(controls, None)
        self.detectors: list[str] = names
        self.controls: dict[str, ad.AnnData] = {primary[k]: v for k, v in controls.items()}
        self.unstained = unstained
        self.layer = str(layer)
        self.raw_layer = str(raw_layer)
        # Fail here, not on the first redraw. The unstained needs a raw layer
        # only once it is used as a negative, so that is checked then.
        layer_matrix(unstained, self.layer)
        for adata in self.controls.values():
            layer_matrix(adata, self.layer)
            layer_matrix(adata, self.raw_layer)
        self.positive_gate = str(positive_gate)
        self.negative_gate = str(negative_gate)
        self.statistic = statistic
        self.bins = int(bins)
        self.tile = int(tile_bins)
        self.smooth = float(smooth)
        self.colormap = colormap
        self._updating = False
        self._filling = False
        self._confirm_overwrite = False

        loaded, self.source = self._starting_matrix(spillover)
        self.spillover: pd.DataFrame = loaded
        self._baseline = loaded.copy()

        rng = np.random.default_rng(seed)
        self._rows = {
            id(a): subsample_indices(a.n_obs, max_events, rng=rng)
            for a in (*self.controls.values(), unstained)
        }
        self._raw: dict[int, np.ndarray] = {}
        self._limits: dict[tuple[str, str], tuple[float, float]] = {}

        self.control = self.detectors[0]
        #: ``(control, detector)`` of the tile being read, or ``None``.
        self.focus: tuple[str, str] | None = None
        self.view = VIEWS[0]
        #: Whether the gating plot overlays the unstained tube.
        self.show_unstained = True
        #: The gate put back on the canvas to adjust, ``"positive"`` or
        #: ``"negative"``, or ``None``. Its band is hidden meanwhile, so the
        #: editable outline is the only copy on screen.
        self.editing: str | None = None
        #: Whether a gate has been applied or deleted since the matrix was
        #: last computed, so the table no longer follows from the gates.
        self.gates_changed = False
        #: The scatter channel on the y axis of the second grid, or ``None``.
        self.scatter: str | None = _default_scatter(self._reference)
        second = self.detectors[min(1, len(self.detectors) - 1)]
        #: What the "two plots" view shows: ``control``, ``x`` and ``y`` each.
        self.pair = [
            {"control": self.control, "x": self.control, "y": self._next(self.control)},
            {"control": second, "x": second, "y": self._next(second)},
        ]
        #: Each plot's manual axis ranges, ``{"x": [lo, hi], "y": [lo, hi]}``
        #: in the units the axis is ticked in; ``None`` at an end is automatic.
        self.ranges = [{"x": [None, None], "y": [None, None]} for _ in self.pair]
        #: Bins per axis of each of the two plots' densities.
        self.pair_bins = [self.bins, self.bins]
        #: What each range box was last filled with, to tell a typed value apart.
        self._shown: dict[tuple[int, str, int], str] = {}

        background = DEFAULT_BACKGROUND if background is None else background
        self.background = str(background)
        light = _is_light(self.background)
        self._foreground = "#1e1e21" if light else "#f0f1f2"
        self._grid_colour = "#9a9a9a" if light else "#5a5a5a"

        self.viewer = viewer if viewer is not None else napari.Viewer(title=title)
        _hide_overlays(self.viewer)

        # One panel does both jobs: the overlaid histograms while gating, and the
        # whole N x N grid, tiled into its one image, while checking. One image
        # rather than N^2 layers is what keeps a large panel usable.
        self.panel = Panel(self.viewer, 1, layer=self.layer, colormap=colormap)
        self._init_overlays()
        self._init_widget()
        self._init_table()
        self.viewer.window.add_dock_widget(self.widget, area="right", name="compensation")
        self.viewer.window.add_dock_widget(self.table_widget, area="right", name="spillover")
        self.viewer.mouse_drag_callbacks.append(self._clicked)

        density_image(np.zeros(1), np.zeros(1), Axes2D(0.0, 1.0, 0.0, 1.0, bins=8))
        self._set_background()
        loaded_matrix = self.source in ("file $SPILLOVER", "argument")
        self.step = STEPS[1] if loaded_matrix else STEPS[0]
        self._bind()
        self.refresh()
        self.viewer.reset_view()

    # ---------------------------------------------------------- starting matrix
    @property
    def _reference(self) -> ad.AnnData:
        return next(iter(self.controls.values()))

    def _starting_matrix(self, spillover) -> tuple[pd.DataFrame, str]:
        """The matrix the table opens on, over every detector it has to cover."""
        if isinstance(spillover, str):
            if spillover != "file":
                raise ValueError(
                    f"spillover must be 'file', a DataFrame or None, not {spillover!r}"
                )
            found = {d: a.uns.get("spillover") for d, a in self.controls.items()}
            carrying = {d: m for d, m in found.items() if m is not None}
            if not carrying:
                return self._over(None), "identity"
            if len(carrying) != len(found):
                missing = sorted(set(found) - set(carrying))
                raise ValueError(
                    f"controls {missing} carry no $SPILLOVER but the others do; pass "
                    "spillover= a DataFrame, or None to start from the identity"
                )
            first_name, first = next(iter(carrying.items()))
            for name, other in carrying.items():
                same = (
                    list(other.index) == list(first.index)
                    and list(other.columns) == list(first.columns)
                    and np.allclose(other.to_numpy(dtype=float), first.to_numpy(dtype=float))
                )
                if not same:
                    raise ValueError(
                        f"the controls for {first_name} and {name} carry different $SPILLOVER "
                        "matrices; pass the one you mean as spillover="
                    )
            return self._over(first, "the file's $SPILLOVER"), "file $SPILLOVER"
        if spillover is None:
            return self._over(None), "identity"
        if isinstance(spillover, pd.DataFrame):
            return self._over(spillover, "spillover"), "argument"
        raise TypeError(f"spillover must be 'file', a DataFrame or None, not {type(spillover)}")

    def _over(self, matrix: pd.DataFrame | None, what: str = "") -> pd.DataFrame:
        """``matrix`` cut down to exactly the control detectors, names resolved.

        Square, over the detectors there are controls for and nothing else. A
        file's ``$SPILLOVER`` covers the whole panel, but a detector nobody
        stained has no dye in it: kept, its row would be inverted as though one
        were there and push error into the real channels. So it is dropped,
        and :func:`~cykit.compensate` leaves that channel as it is. A control
        detector the matrix does not cover starts on an identity row.
        """
        ref = self._reference
        names = list(self.detectors)
        out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        if matrix is None:
            return out
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{what} is not square: {matrix.shape}")
        rows = [str(ref.var_names[find_channel_name(ref, n)]) for n in matrix.index]
        cols = [str(ref.var_names[find_channel_name(ref, n)]) for n in matrix.columns]
        if sorted(rows) != sorted(cols):
            raise ValueError(f"{what}: rows and columns do not name the same detectors")
        frame = pd.DataFrame(matrix.to_numpy(dtype=float), index=rows, columns=cols)
        kept = [d for d in names if d in cols]
        frame = frame.loc[kept, kept]
        values = frame.to_numpy()
        # Judged on what is kept: a dropped detector's diagonal is no concern.
        if not np.allclose(np.diag(values), 1.0):
            raise ValueError(
                f"{what} does not have 1 on its diagonal, so it is a compensation matrix, "
                "or percentages, or neither; read the file with "
                "read_fcs(..., convert_spillover=True) if it is percentages, or pass the "
                "spillover matrix you mean as a DataFrame"
            )
        out.loc[kept, kept] = values
        return out

    # ------------------------------------------------------------------ setup
    def _init_overlays(self) -> None:
        from .viewer import _TEXT_STYLE, Y_LABEL_ROTATION

        self.bands = self.viewer.add_shapes(name="applied gates", opacity=1.0)
        self.bands.editable = False
        # Vectors, like the frame: the grid has two median lines per tile, and
        # rebuilding hundreds of shapes on every edit is what would make
        # tuning feel slow.
        self.guides = self.viewer.add_vectors(
            np.zeros((0, 2, 2)), name="medians", edge_width=1.5, vector_style="line"
        )
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
        self.gates = self.viewer.add_shapes(
            name="gates", face_color="#ffcc0022", edge_color="#ffcc00", edge_width=2
        )

    def _set_background(self) -> None:
        try:
            self.viewer.canvas.background_color_override = self.background
        except AttributeError:  # pragma: no cover - older napari
            canvas = getattr(getattr(self.viewer.window, "_qt_viewer", None), "canvas", None)
            if canvas is not None:
                canvas.bgcolor = self.background

    def _init_widget(self) -> None:
        from magicgui.widgets import (
            CheckBox,
            ComboBox,
            Container,
            Label,
            LineEdit,
            PushButton,
            SpinBox,
        )

        self.w_step = ComboBox(label="step", choices=STEPS, value=STEPS[0])

        self.w_control = ComboBox(label="control", choices=self._control_choices())
        self.w_unstained = CheckBox(text="show unstained", value=self.show_unstained)
        self.w_unstained.changed.connect(self._unstained_toggled)
        self.w_use_unstained = CheckBox(text="use unstained as negative")
        self.w_use_unstained.tooltip = (
            "Take this control's negative population from the whole unstained tube "
            "instead of a negative gate on the control."
        )
        self.w_use_unstained.changed.connect(self._use_unstained_toggled)
        self.w_target = ComboBox(label="draw", choices=TARGETS, value="positive")
        self.w_apply = PushButton(text="apply gate from shapes")
        self.w_adjust = PushButton(text="adjust gate")
        self.w_adjust.tooltip = "Put the gate chosen under draw back on the canvas to move it."
        self.w_delete = PushButton(text="delete gate")
        self.w_delete.tooltip = "Remove the gate chosen under draw from this control."
        self.w_compute = PushButton(text="compute spillover matrix")
        self.gate_box = Container(
            widgets=[
                self.w_control,
                self.w_unstained,
                self.w_use_unstained,
                self.w_target,
                self.w_apply,
                self.w_adjust,
                self.w_delete,
                self.w_compute,
            ],
            label="gate controls",
        )

        channels = [str(n) for n in self._reference.var_names]
        self.w_view = ComboBox(label="view", choices=VIEWS, value=self.view)
        self.w_scatter = ComboBox(
            label="scatter y",
            choices=[NO_SCATTER, *_scatter_channels(self._reference)],
            value=self.scatter or NO_SCATTER,
        )
        hint = Label(
            value="rows: controls\n"
            "left grid: x = own detector, y = column\n"
            "right grid: x = column, y = scatter\n"
            "click a tile or a table cell to read it"
        )
        self.grid_box = Container(widgets=[self.w_scatter, hint], labels=True)
        self.w_pair = []
        self.w_range: list[dict[str, tuple]] = []
        self.w_bins = []
        columns = []
        for i, side in enumerate(("left", "right")):
            control = ComboBox(label="control", choices=list(self.detectors))
            x = ComboBox(label="x", choices=channels)
            y = ComboBox(label="y", choices=channels)
            bins = SpinBox(label="bins", min=MIN_BINS, max=MAX_BINS, value=self.pair_bins[i])
            bins.changed.connect(lambda e, i=i: self._pair_bins_changed(i))
            self.w_pair.append((control, x, y))
            self.w_bins.append(bins)
            boxes = {}
            rows = []
            for axis in ("x", "y"):
                lo, hi = (LineEdit(label="", tooltip=_RANGE_TOOLTIP) for _ in range(2))
                for box in (lo, hi):
                    box.max_width = 90
                    # editingFinished, not ``changed``: that fires per keystroke.
                    box.native.editingFinished.connect(
                        lambda i=i, axis=axis: self._range_entered(i, axis)
                    )
                boxes[axis] = (lo, hi)
                row = Container(
                    widgets=[lo, hi],
                    layout="horizontal",
                    label=f"{axis} range",
                    labels=False,
                )
                row.margins = (0, 0, 0, 0)
                rows.append(row)
            self.w_range.append(boxes)
            # One column per plot, side by side like the plots themselves.
            columns.append(
                Container(
                    widgets=[Label(value=f"{side} plot"), control, x, rows[0], y, rows[1], bins],
                    labels=True,
                )
            )
            control.changed.connect(lambda e, i=i: self._pair_control_changed(i))
            x.changed.connect(lambda e, i=i: self._pair_channel_changed(i))
            y.changed.connect(lambda e, i=i: self._pair_channel_changed(i))
        self.pair_box = Container(widgets=columns, layout="horizontal", labels=False)
        self.w_fit = PushButton(text="fit axes")
        self.check_box = Container(
            widgets=[self.w_view, self.grid_box, self.pair_box, self.w_fit],
            label="check compensation",
        )
        self.w_view.changed.connect(self._view_changed)
        self.w_scatter.changed.connect(self._scatter_changed)

        self.w_head = Label(value="")
        self.w_status = Label(value="")
        self.widget = Container(
            widgets=[self.w_step, self.gate_box, self.check_box, self.w_head, self.w_status]
        )

        self.w_step.changed.connect(self._step_changed)
        self.w_control.changed.connect(self._control_changed)
        self.w_target.changed.connect(self._target_changed)
        self.w_apply.changed.connect(self._apply_clicked)
        self.w_adjust.changed.connect(self._adjust_clicked)
        self.w_delete.changed.connect(self._delete_clicked)
        self.w_compute.changed.connect(self._compute_clicked)
        self.w_fit.changed.connect(self._fit_clicked)

        self.viewer.bind_key("n", lambda v: self.step_control(1), overwrite=True)
        self.viewer.bind_key("p", lambda v: self.step_control(-1), overwrite=True)

    def _init_table(self) -> None:
        """The matrix as an editable grid, with its buttons underneath."""
        from qtpy.QtWidgets import (
            QDoubleSpinBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QTableWidget,
            QVBoxLayout,
            QWidget,
        )

        self.table = QTableWidget()
        self.table.itemChanged.connect(self._cell_edited)
        self.table.currentCellChanged.connect(self._cell_focused)
        #: ``(row, column) -> (minus, box, plus)`` for each off-diagonal cell.
        self._cells: dict[tuple[int, int], tuple] = {}
        self.w_reset = QPushButton("reset")
        self.w_reset.setToolTip(
            "Put back the matrix the window started from, or the last one computed."
        )
        self.w_reset.clicked.connect(lambda *_: self.reset())
        self.w_copy = QPushButton("copy as Python")
        self.w_copy.clicked.connect(lambda *_: self._copy())
        buttons = QHBoxLayout()
        buttons.addWidget(self.w_reset)
        buttons.addWidget(self.w_copy)

        # How far a cell's - and + move it. Yours to set, because how fine a
        # step means anything depends on how much a dye spills.
        self.w_step_size = QDoubleSpinBox()
        self.w_step_size.setDecimals(4)
        self.w_step_size.setRange(0.0001, 1.0)
        self.w_step_size.setSingleStep(0.0005)
        self.w_step_size.setValue(DEFAULT_STEP)
        self.w_step_size.setToolTip("How much one press of a cell's - or + changes it.")
        step = QHBoxLayout()
        step.addWidget(QLabel("- / + step"))
        step.addWidget(self.w_step_size)
        step.addStretch(1)

        layout = QVBoxLayout()
        note = QLabel("row: dye   column: detector it spills into")
        layout.addWidget(note)
        layout.addWidget(self.table)
        layout.addLayout(step)
        layout.addLayout(buttons)
        self.table_widget = QWidget()
        self.table_widget.setLayout(layout)
        self._fill_table()

    def _build_table(self, names) -> None:
        """Lay the grid out once: an item per cell, and - box + over the off-diagonal ones.

        The widgets are kept and only their text is rewritten afterwards:
        rebuilding a few hundred of them on every press would be what made a
        large panel feel slow.
        """
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QHBoxLayout, QLineEdit, QTableWidgetItem, QToolButton, QWidget

        n = len(names)
        self.table.setRowCount(n)
        self.table.setColumnCount(n)
        self.table.setVerticalHeaderLabels(names)
        self.table.setHorizontalHeaderLabels(names)
        self._cells = {}
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem("")
                if i == j:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, j, item)
                if i == j:
                    continue
                cell = QWidget()
                row = QHBoxLayout(cell)
                row.setContentsMargins(1, 1, 1, 1)
                row.setSpacing(1)
                minus, plus = QToolButton(), QToolButton()
                minus.setText("−")
                plus.setText("+")
                for button in (minus, plus):
                    button.setFixedWidth(_CELL_BUTTON_WIDTH)
                    button.setAutoRaise(True)
                box = QLineEdit()
                box.setAlignment(Qt.AlignCenter)
                box.setFixedWidth(_CELL_BOX_WIDTH)
                minus.clicked.connect(lambda *_, i=i, j=j: self._cell_nudged(i, j, -1))
                plus.clicked.connect(lambda *_, i=i, j=j: self._cell_nudged(i, j, 1))
                box.editingFinished.connect(lambda i=i, j=j: self._cell_typed(i, j))
                row.addWidget(minus)
                row.addWidget(box)
                row.addWidget(plus)
                self.table.setCellWidget(i, j, cell)
                self._cells[(i, j)] = (minus, box, plus)
        for j in range(n):
            self.table.setColumnWidth(j, _CELL_WIDTH)
        # Tall enough to show every row (up to a point) when the dock opens,
        # rather than squeezed to two under the controls.
        rows = self.table.verticalHeader().length() + self.table.horizontalHeader().height()
        self.table.setMinimumHeight(min(rows + 8, 420))

    def _fill_table(self) -> None:
        """Rewrite every cell from :attr:`spillover`, marking the edited ones."""
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QColor

        names = list(self.spillover.index)
        self._filling = True
        try:
            if self.table.rowCount() != len(names) or (not self._cells and len(names) > 1):
                self._build_table(names)
            values = self.spillover.to_numpy()
            base = self._baseline.loc[names, names].to_numpy()
            for i in range(len(names)):
                for j in range(len(names)):
                    text = f"{values[i, j]:.4f}"
                    item = self.table.item(i, j)
                    item.setText(text)
                    # A cell given its own background gets its own text colour
                    # too: napari's dark theme writes white, which on these
                    # light fills cannot be read.
                    if i == j:
                        item.setBackground(QColor(_DIAGONAL))
                        item.setForeground(QColor(_CELL_TEXT))
                        continue
                    edited = values[i, j] != base[i, j]
                    if edited:
                        item.setBackground(QColor(_EDITED))
                        item.setForeground(QColor(_CELL_TEXT))
                    else:
                        # Back to the theme's own colours.
                        item.setData(Qt.BackgroundRole, None)
                        item.setData(Qt.ForegroundRole, None)
                    _, box, _ = self._cells[(i, j)]
                    box.setText(text)
                    box.setStyleSheet(
                        f"background: {_EDITED}; color: {_CELL_TEXT};" if edited else ""
                    )
        finally:
            self._filling = False

    def _cell_nudged(self, i: int, j: int, steps: int) -> None:
        names = list(self.spillover.index)
        try:
            self.nudge(steps, names[i], names[j])
        except ValueError as exc:
            self.w_status.value = str(exc)

    def _cell_typed(self, i: int, j: int) -> None:
        """A value typed into a cell's box, committed on Enter or on leaving it."""
        if self._filling:
            return
        names = list(self.spillover.index)
        _, box, _ = self._cells[(i, j)]
        text = box.text().strip()
        if text == f"{self.spillover.iat[i, j]:.4f}":
            return  # left without changing it
        try:
            value = float(text)
        except ValueError:
            self.w_status.value = f"{text!r} is not a number"
            self._fill_table()
            return
        self.set_coefficient(names[i], names[j], value)
        self._select_cell(i, j)

    def _select_cell(self, i: int, j: int) -> None:
        """Make ``(i, j)`` the table's current cell, which also picks out its tile."""
        if (self.table.currentRow(), self.table.currentColumn()) != (i, j):
            self.table.setCurrentCell(i, j)

    # -------------------------------------------------------------- selection
    @contextmanager
    def _quiet(self):
        """Set widget values without their ``changed`` callbacks firing back."""
        self._updating = True
        try:
            yield
        finally:
            self._updating = False

    def gate_state(self, detector: str) -> tuple[bool, bool]:
        """Whether a control has its positive population and its negative one.

        Parameters
        ----------
        detector
            The control's detector.

        Returns
        -------
        tuple of bool
            ``(has_positive, has_negative)``. The negative is there when the
            control has its negative gate, or when it uses the unstained.
        """
        obs = self.controls[detector].obs
        negative = self.uses_unstained(detector) or self.negative_gate in obs
        return self.positive_gate in obs, negative

    def uses_unstained(self, detector: str) -> bool:
        """Whether a control takes its negative population from the unstained tube.

        Parameters
        ----------
        detector
            The control's detector.

        Returns
        -------
        bool
            Recorded on the control as ``uns["cykit"]["use_unstained"]``, so
            the choice is still there when the window is opened again.
        """
        info = self.controls[detector].uns.get("cykit", {})
        return bool(info.get("use_unstained", False))

    def set_use_unstained(self, use: bool, detector: str | None = None) -> None:
        """Take a control's negative population from the unstained tube, or not.

        When on, the control's negatives are every event of the unstained tube
        -- the same cells, cut down the same way, with nothing stained -- and
        its own negative gate, if it has one, is kept but not used. Turning it
        off puts that gate back in use.

        Parameters
        ----------
        use
            On or off.
        detector
            The control; the current one by default.

        Raises
        ------
        KeyError
            If the unstained tube has no ``raw_layer`` to measure.
        """
        detector = self.control if detector is None else self._detector(detector)
        if use:
            layer_matrix(self.unstained, self.raw_layer)
        info = self.controls[detector].uns.setdefault("cykit", {})
        if use:
            info["use_unstained"] = True
        else:
            info.pop("use_unstained", None)
        self.gates_changed = True
        self._bind()
        self.refresh()

    def _control_choices(self):
        out = []
        for name in self.detectors:
            positive, negative = self.gate_state(name)
            mark = "●" if positive and negative else "○"
            pos = "✓" if positive else "—"
            neg = "unstained" if self.uses_unstained(name) else ("✓" if negative else "—")
            out.append((f"{mark} {name}   pos {pos}   neg {neg}", name))
        return out

    def _bind(self) -> None:
        """Point the widgets at the current state, without firing their callbacks."""
        with self._quiet():
            self.w_step.value = self.step
            self.w_control.choices = self._control_choices()
            self.w_control.value = self.control
            self.w_use_unstained.value = self.uses_unstained(self.control)
            self.w_view.value = self.view
            self.w_scatter.value = self.scatter or NO_SCATTER
            for (control, x, y), state in zip(self.w_pair, self.pair):
                control.value = state["control"]
                x.value = state["x"]
                y.value = state["y"]
        self.grid_box.visible = self.view == VIEWS[0]
        self.pair_box.visible = self.view == VIEWS[1]
        gating = self.step == STEPS[0]
        self.gate_box.visible = gating
        self.check_box.visible = not gating
        self.gates.visible = gating
        # Not once the window is closing: Qt still fires the combo boxes as it
        # tears them down, after napari has removed the layers.
        if gating and self.gates in self.viewer.layers:
            self.viewer.layers.selection = {self.gates}

    def set_step(self, step: str) -> None:
        """Switch between gating the controls and checking the compensation.

        Parameters
        ----------
        step
            One of :data:`STEPS`.
        """
        if step not in STEPS:
            raise ValueError(f"step must be one of {STEPS}, not {step!r}")
        changed = step != self.step
        self.step = step
        self._stop_editing()
        self._bind()
        self.refresh()
        if changed:
            self.viewer.reset_view()  # the two steps are very different sizes

    def set_show_unstained(self, show: bool) -> None:
        """Overlay the unstained tube on the gating plot, or hide it.

        Parameters
        ----------
        show
            ``False`` leaves the control on its own. The axis does not move.
        """
        self.show_unstained = bool(show)
        with self._quiet():
            self.w_unstained.value = self.show_unstained
        self.refresh()

    def _use_unstained_toggled(self, *_) -> None:
        if self._updating:
            return
        try:
            self.set_use_unstained(self.w_use_unstained.value)
        except KeyError as exc:
            self.w_status.value = f"the unstained tube has no {self.raw_layer!r} layer: {exc}"
            with self._quiet():
                self.w_use_unstained.value = False

    def _unstained_toggled(self, *_) -> None:
        if not self._updating:
            self.set_show_unstained(self.w_unstained.value)

    def set_control(self, detector: str) -> None:
        """Gate a different control.

        Parameters
        ----------
        detector
            Its detector, marker or ``var_name``.
        """
        self.control = self._detector(detector)
        self._stop_editing()
        self._bind()
        self.refresh()

    def step_control(self, delta: int = 1) -> None:
        """Move to the next (or previous) control, wrapping round.

        Parameters
        ----------
        delta
            How many controls to move by.
        """
        i = (self.detectors.index(self.control) + delta) % len(self.detectors)
        self.set_control(self.detectors[i])

    def set_view(self, view: str) -> None:
        """Switch the check step between the grids and two plots of your choosing.

        Parameters
        ----------
        view
            One of :data:`VIEWS`.
        """
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
        changed = view != self.view
        self.view = view
        self._bind()
        self.refresh()
        if changed:
            self.viewer.reset_view()

    def set_scatter(self, channel: str | None) -> None:
        """The scatter channel on the y axis of the second grid.

        Parameters
        ----------
        channel
            Any channel, by name; ``None`` hides the second grid.
        """
        if channel in (None, NO_SCATTER):
            self.scatter = None
        else:
            ref = self._reference
            self.scatter = str(ref.var_names[find_channel_name(ref, channel)])
        self._bind()
        self.refresh()
        self.viewer.reset_view()

    def set_pair(
        self, plot: int, *, control: str | None = None, x: str | None = None, y: str | None = None
    ) -> None:
        """What one of the two plots shows, in the "two plots" view.

        Parameters
        ----------
        plot
            ``0`` for the left plot, ``1`` for the right.
        control
            Which control. Changing it puts its own detector on x and the next
            control's detector on y, unless ``x`` or ``y`` say otherwise.
        x, y
            Any channels -- fluorescence, compensated with the matrix, or
            scatter, as stored.
        """
        state = self.pair[plot]
        before = dict(state)
        if control is not None:
            state["control"] = self._detector(control)
            state["x"] = state["control"]
            state["y"] = self._next(state["control"])
        ref = self._reference
        if x is not None:
            state["x"] = str(ref.var_names[find_channel_name(ref, x)])
        if y is not None:
            state["y"] = str(ref.var_names[find_channel_name(ref, y)])
        # A range typed for one channel means nothing on another.
        for axis in ("x", "y"):
            if state[axis] != before[axis] or state["control"] != before["control"]:
                self.ranges[plot][axis] = [None, None]
        self._bind()
        self.refresh()

    def set_pair_range(
        self, plot: int, axis: str, lo: float | None = None, hi: float | None = None
    ) -> None:
        """Fix one axis of one of the two plots.

        Parameters
        ----------
        plot
            ``0`` for the left plot, ``1`` for the right.
        axis
            ``"x"`` or ``"y"``.
        lo, hi
            The ends, in the units the axis is ticked in -- raw units on an
            arcsinh or logicle axis. ``None`` puts that end back on automatic.

        Raises
        ------
        ValueError
            If ``lo`` is not below ``hi``.
        """
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', not {axis!r}")
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(f"min ({lo:g}) must be below max ({hi:g})")
        self.ranges[plot][axis] = [lo, hi]
        self.refresh()

    def set_pair_bins(self, plot: int, bins: int) -> None:
        """How finely one of the two plots bins its density.

        Both plots are drawn at the size of the finer one, a pixel per bin, and
        the coarser one is scaled up to match -- its bins drawn as bigger
        squares -- so the two always sit side by side at the same size. The
        view is refitted when that size changes.

        Parameters
        ----------
        plot
            ``0`` for the left plot, ``1`` for the right.
        bins
            Bins per axis, from :data:`MIN_BINS` to :data:`MAX_BINS`.
        """
        bins = int(bins)
        if not MIN_BINS <= bins <= MAX_BINS:
            raise ValueError(f"bins must be between {MIN_BINS} and {MAX_BINS}, not {bins}")
        size = self._pair_size
        self.pair_bins[plot] = bins
        with self._quiet():
            self.w_bins[plot].value = bins
        self.refresh()
        if self._pair_size != size and self.view == VIEWS[1] and self.step == STEPS[1]:
            self.viewer.reset_view()

    @property
    def _pair_size(self) -> int:
        """Side of each of the two plots on the canvas: the finer one's bins."""
        return max(self.pair_bins)

    def _pair_bins_changed(self, i: int) -> None:
        if not self._updating:
            self.set_pair_bins(i, self.w_bins[i].value)

    def _next(self, detector: str) -> str:
        """The next control's detector: a y axis that shows some spillover."""
        names = self.detectors
        return names[(names.index(detector) + 1) % len(names)]

    def set_focus(self, control: str, detector: str) -> None:
        """Pick out one tile of the check grid, and its cell of the table.

        The tile is outlined and the status line reads it: the coefficient,
        and how far the positive median sits from the negative one in that
        detector after compensation, in raw units.

        Parameters
        ----------
        control
            The row: a control's detector.
        detector
            The column.
        """
        control = self._detector(control)
        ref = self._reference
        detector = str(ref.var_names[find_channel_name(ref, detector)])
        names = list(self.spillover.index)
        if detector not in names:
            raise KeyError(f"{detector!r} is not in the matrix")
        self.focus = (control, detector)
        i, j = names.index(control), names.index(detector)
        if (self.table.currentRow(), self.table.currentColumn()) != (i, j):
            self._filling = True
            try:
                self.table.setCurrentCell(i, j)
            finally:
                self._filling = False
        if self.step == STEPS[1]:
            self._draw_focus()

    def _detector(self, name: str) -> str:
        ref = self._reference
        resolved = str(ref.var_names[find_channel_name(ref, name)])
        if resolved not in self.controls:
            raise KeyError(f"no control for {resolved!r}; controls are {self.detectors}")
        return resolved

    # --------------------------------------------------------------- callbacks
    def _step_changed(self, *_) -> None:
        if not self._updating:
            self.set_step(self.w_step.value)

    def _control_changed(self, *_) -> None:
        if not self._updating and self.w_control.value is not None:
            self.set_control(self.w_control.value)

    def _target_changed(self, *_) -> None:
        if not self._updating and self.editing is not None:
            self._stop_editing()
            self.refresh()

    def _view_changed(self, *_) -> None:
        if not self._updating:
            self.set_view(self.w_view.value)

    def _scatter_changed(self, *_) -> None:
        if not self._updating:
            self.set_scatter(self.w_scatter.value)

    def _pair_control_changed(self, i: int) -> None:
        if not self._updating:
            self.set_pair(i, control=self.w_pair[i][0].value)

    def _pair_channel_changed(self, i: int) -> None:
        if not self._updating:
            _, x, y = self.w_pair[i]
            self.set_pair(i, x=x.value, y=y.value)

    def _cell_focused(self, row: int, col: int, *_) -> None:
        if self._filling or row < 0 or col < 0:
            return
        names = list(self.spillover.index)
        if names[row] in self.controls:
            self.set_focus(names[row], names[col])

    def _clicked(self, viewer, event) -> None:
        """A click on the check grid picks out the tile under it."""
        if self.step != STEPS[1] or event.type != "mouse_press":
            return
        tile = self.tile_at(*np.asarray(event.position, dtype=float)[-2:])
        if tile is not None:
            self.set_focus(*tile)

    def _apply_clicked(self, *_) -> None:
        try:
            self.apply_gate(self.w_target.value)
        except ValueError as exc:
            self.w_status.value = str(exc)

    def _adjust_clicked(self, *_) -> None:
        try:
            self.adjust_gate(self.w_target.value)
        except ValueError as exc:
            self.w_status.value = str(exc)

    def _delete_clicked(self, *_) -> None:
        try:
            self.delete_gate(self.w_target.value)
        except ValueError as exc:
            self.w_status.value = str(exc)

    def _compute_clicked(self, *_) -> None:
        self.compute()

    def _fit_clicked(self, *_) -> None:
        self._limits.clear()
        self.ranges = [{"x": [None, None], "y": [None, None]} for _ in self.pair]
        self.refresh()

    def _range_entered(self, plot: int, axis: str) -> None:
        values = []
        for k, box in enumerate(self.w_range[plot][axis]):
            text = (box.value or "").strip()
            if text == self._shown.get((plot, axis, k)):
                # Untouched: still the readout, rounded for display, so keep
                # whatever that end was rather than pinning it to the rounding.
                values.append(self.ranges[plot][axis][k])
                continue
            try:
                values.append(float(text.replace(",", "")) if text else None)
            except ValueError:
                self.refresh()  # puts the boxes back to what is drawn
                self.w_status.value = f"{text!r} is not a number"
                return
        try:
            self.set_pair_range(plot, axis, *values)
        except ValueError as exc:
            self.refresh()
            self.w_status.value = str(exc)

    def _cell_edited(self, item) -> None:
        if self._filling:
            return
        names = list(self.spillover.index)
        row, col = names[item.row()], names[item.column()]
        try:
            value = float(item.text())
        except ValueError:
            self.w_status.value = f"{item.text()!r} is not a number"
            self._fill_table()
            return
        self.set_coefficient(row, col, value)

    def _copy(self) -> None:
        from qtpy.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.to_source())
        self.w_status.value = "copied — paste it into your notebook"

    # ------------------------------------------------------------------ gating
    def _gate_name(self, target: str) -> str:
        """The control's gate a drawn interval becomes."""
        if target == "positive":
            return self.positive_gate
        if target == "negative":
            return self.negative_gate
        raise ValueError(f"target must be one of {[v for _, v in TARGETS]}, not {target!r}")

    def drawn_intervals(self) -> list[tuple[float, float]]:
        """The shapes on the **gates** layer, as intervals of the stained detector.

        A shape's extent along the x axis is what counts, so a box dragged over
        a peak gates the events under it -- whether or not the box also covers
        the unstained curve, which is never gated.

        Returns
        -------
        list of tuple
            ``(lo, hi)`` per shape, in the values of :attr:`layer`.
        """
        panel = self.panel
        out = []
        for verts in self.gates.data:
            rc = np.asarray(verts, dtype=float) - np.array([panel.row, panel.col])
            xy = panel.axes.to_display(rc)
            out.append((float(xy[:, 0].min()), float(xy[:, 0].max())))
        return out

    def apply_gate(self, target: str, intervals=None) -> np.ndarray:
        """Turn an interval of the stained detector into a gate.

        What the *apply gate from shapes* button does, callable directly.

        Parameters
        ----------
        target
            ``"positive"`` or ``"negative"``. Either way it is a gate on the
            current control, over the control's events only; the unstained
            overlaid on the same plot is not touched.
        intervals
            ``(lo, hi)``, or a list of them, in the values of :attr:`layer`.
            ``None`` reads them off the shapes drawn on the canvas.

        Returns
        -------
        ndarray
            Boolean mask over every event of the control.

        Raises
        ------
        ValueError
            If nothing has been drawn, the gate holds no control events, or it
            overlaps the control's other gate.
        """
        name = self._gate_name(target)
        adata = self.controls[self.control]
        if intervals is None:
            intervals = self.drawn_intervals()
        elif np.ndim(intervals) == 1:
            intervals = [intervals]
        intervals = [(float(min(i)), float(max(i))) for i in intervals]
        if not intervals:
            raise ValueError("draw a box over the population on the canvas first")

        detector = self.control
        values = self._display_column(adata, detector)
        mask = np.zeros(adata.n_obs, dtype=bool)
        for lo, hi in intervals:
            mask |= (values >= lo) & (values <= hi)
        if not mask.any():
            raise ValueError("that interval holds no events")
        other = self.negative_gate if target == "positive" else self.positive_gate
        if other in adata.obs and (mask & adata.obs[other].to_numpy(dtype=bool)).any():
            raise ValueError(f"the {target} interval overlaps this control's {other!r} gate")

        add_gate(
            adata,
            name,
            mask,
            meta={
                "kind": "histogram",
                # The interval is recorded as a square on the diagonal of
                # (detector, detector): an event is inside it exactly when its
                # value lies in the interval, so gate_mask and plot_gate can
                # recompute and redraw it without knowing it was a histogram.
                "x": detector,
                "y": detector,
                "layer": self.layer,
                "vertices": [_diagonal_square(lo, hi) for lo, hi in intervals],
                "shape_types": ["polygon"] * len(intervals),
            },
        )
        replaced = self.editing == target
        self._stop_editing()
        self.gates_changed = True
        self._bind()
        self.refresh()
        verb = "replaced" if replaced else name
        self.w_status.value = f"{verb}: {int(mask.sum()):,} events ({mask.mean():.1%})"
        if target == "negative" and self.uses_unstained(detector):
            self.w_status.value += (
                " — kept, but not used while 'use unstained as negative' is ticked"
            )
        return mask

    def adjust_gate(self, target: str) -> None:
        """Put one of the current control's gates back on the canvas, to move it.

        Its intervals come back as boxes on the **gates** layer, in select mode
        so they can be dragged and resized straight away; its band is hidden
        meanwhile. Applying again replaces the gate. Picking another control,
        step or gate drops the edit and leaves the gate as it was.

        Parameters
        ----------
        target
            ``"positive"`` or ``"negative"``.

        Raises
        ------
        ValueError
            If the control has no such gate, or it was not drawn here -- one
            from another window has no interval to put back.
        """
        name = self._gate_name(target)
        adata = self.controls[self.control]
        intervals = self._intervals(adata, name, self.control)
        if not intervals:
            missing = name not in adata.obs
            raise ValueError(
                f"this control has no {target} gate"
                if missing
                else f"{name!r} was not drawn in this window, so there is no interval to adjust"
            )
        with self._quiet():
            self.w_target.value = target
        self.editing = target
        self.refresh()  # hides the band before the outline goes on top of it
        panel = self.panel
        ax = panel.axes
        top, bottom = panel.row - 0.5, panel.row + self.bins - 0.5
        self.gates.data = []
        for lo, hi in intervals:
            lo, hi = max(lo, ax.x_lo), min(hi, ax.x_hi)
            c0, c1 = panel.col + ax.to_pixels(np.array([lo, hi]), np.full(2, ax.y_lo))[:, 1]
            self.gates.add(
                np.array([[top, c0], [top, c1], [bottom, c1], [bottom, c0]]),
                shape_type="rectangle",
            )
        if self.gates in self.viewer.layers:
            self.viewer.layers.selection = {self.gates}
            self.gates.mode = "select"
        self.w_status.value = f"{name} loaded: drag or resize it, then apply to replace it"

    def delete_gate(self, target: str) -> None:
        """Remove one of the current control's gates.

        Parameters
        ----------
        target
            ``"positive"`` or ``"negative"``.

        Raises
        ------
        ValueError
            If the control has no such gate.
        """
        name = self._gate_name(target)
        adata = self.controls[self.control]
        if name not in adata.obs:
            raise ValueError(f"this control has no {target} gate")
        _drop_gate(adata, name)
        self._stop_editing()
        self.gates_changed = True
        self._bind()
        self.refresh()
        self.w_status.value = f"deleted {name} on {self.control}"

    def _stop_editing(self) -> None:
        """Drop whatever is on the canvas, and any gate being adjusted."""
        self.editing = None
        self.gates.data = []

    def _display_column(self, adata: ad.AnnData, channel: str) -> np.ndarray:
        j = find_channel_name(adata, channel)
        return np.asarray(layer_matrix(adata, self.layer)[:, j], dtype=np.float64).ravel()

    def _intervals(self, adata: ad.AnnData, name: str, detector: str):
        """The recorded intervals of a gate drawn here, or ``[]``."""
        if name not in adata.obs:
            return []
        try:
            record = gate_record(adata, name)
        except KeyError:
            return []
        if record.kind != "histogram" or record.x != detector or record.layer != self.layer:
            return []
        return [(float(v[:, 0].min()), float(v[:, 0].max())) for v in record.vertices]

    # ---------------------------------------------------------------- matrix
    @property
    def edited(self) -> bool:
        """Whether the matrix differs from the one it started from, or was computed as."""
        return not self.spillover.equals(self._baseline)

    def missing_gates(self) -> list[str]:
        """Controls without a positive gate, or without a negative from either side.

        Returns
        -------
        list of str
            Detector names, in matrix order.
        """
        return [d for d in self.detectors if not all(self.gate_state(d))]

    def compute(self, *, overwrite_edits: bool = False) -> pd.DataFrame | None:
        """Compute the control rows of the matrix from the gates, in place.

        Every control must be gated first: nothing is split automatically
        here. Every row is a control's, so every row is replaced.

        Parameters
        ----------
        overwrite_edits
            Replace hand edits without asking. From the button, the first press
            on an edited matrix warns and the second goes ahead.

        Returns
        -------
        DataFrame or None
            :attr:`spillover`, or ``None`` if nothing was computed; the status
            line says why.
        """
        missing = self.missing_gates()
        if missing:
            self.w_status.value = f"gate these first: {', '.join(missing)}"
            return None
        if self.edited and not (overwrite_edits or self._confirm_overwrite):
            self._confirm_overwrite = True
            self.w_status.value = (
                "the matrix has hand edits; computing replaces the control rows — "
                "press compute again to go ahead"
            )
            return None
        self._confirm_overwrite = False
        try:
            chosen = [d for d in self.detectors if self.uses_unstained(d)]
            computed = compute_spillover_matrix(
                # The unstained only for the controls that chose it; every
                # other negative is the one gated on the control itself.
                self.controls,
                unstained=self.unstained if chosen else None,
                use_unstained=chosen or None,
                channels=list(self.spillover.index),
                layer=self.raw_layer,
                statistic=self.statistic,
                positive_gate=self.positive_gate,
                negative_gate=self.negative_gate,
            )
        except (ValueError, KeyError) as exc:
            self.w_status.value = str(exc)
            return None
        columns = list(self.spillover.columns)
        for detector in self.detectors:
            self.spillover.loc[detector, columns] = computed.loc[detector, columns].to_numpy()
        self.spillover.attrs["cykit"] = computed.attrs.get("cykit", {})
        self._baseline = self.spillover.copy()
        self.source = "computed"
        self.gates_changed = False
        self._fill_table()
        self.set_step(STEPS[1])
        return self.spillover

    def set_coefficient(self, row: str, column: str, value: float) -> None:
        """Set one coefficient and redraw the check plots.

        Parameters
        ----------
        row
            The dye, by detector, marker or ``var_name``.
        column
            The detector it spills into.
        value
            Fraction of the dye's signal that lands in ``column``.

        Raises
        ------
        ValueError
            On the diagonal, which is ``1`` by definition.
        """
        ref = self._reference
        r = str(ref.var_names[find_channel_name(ref, row)])
        c = str(ref.var_names[find_channel_name(ref, column)])
        if r == c:
            raise ValueError("the diagonal is 1 by definition")
        self.spillover.loc[r, c] = float(value)
        self._confirm_overwrite = False
        self._fill_table()
        self.refresh()

    def nudge(self, steps: int = 1, row: str | None = None, column: str | None = None) -> float:
        """Move one coefficient by whole steps of the **step** box.

        What a cell's - and + buttons do.

        Parameters
        ----------
        steps
            How many steps, negative to lower it.
        row, column
            The cell; the one selected in the table by default.

        Returns
        -------
        float
            The coefficient's new value.

        Raises
        ------
        ValueError
            If no cell is selected, or it is on the diagonal.
        """
        names = list(self.spillover.index)
        if row is None or column is None:
            i, j = self.table.currentRow(), self.table.currentColumn()
            if i < 0 or j < 0:
                raise ValueError("select a cell of the table first")
            row, column = names[i], names[j]
        step = float(self.w_step_size.value())
        # Rounded to the step box's precision, so repeated presses do not
        # accumulate float noise into the fifth decimal.
        value = round(float(self.spillover.loc[row, column]) + steps * step, 6)
        self.set_coefficient(row, column, value)
        self._select_cell(names.index(row), names.index(column))
        return value

    def reset(self) -> None:
        """Put back the matrix the window started from, or the last one computed."""
        self.spillover.iloc[:, :] = self._baseline.loc[
            self.spillover.index, self.spillover.columns
        ].to_numpy()
        self._fill_table()
        self.refresh()

    def to_source(self, name: str = "SPILLOVER") -> str:
        """The matrix as a Python assignment, for pasting into a notebook.

        Parameters
        ----------
        name
            Variable to assign to.

        Returns
        -------
        str
            ``name = pd.DataFrame(...)``.
        """
        names = [str(n) for n in self.spillover.index]
        rows = ",\n".join(
            "        [" + ", ".join(f"{v:.6g}" for v in row) + "]"
            for row in self.spillover.to_numpy()
        )
        return (
            f"{name} = pd.DataFrame(\n    [\n{rows},\n    ],\n"
            f"    index={names!r},\n    columns={names!r},\n)"
        )

    # ------------------------------------------------------------------- data
    def _raw_rows(self, adata: ad.AnnData) -> np.ndarray:
        """The plotted events of ``raw_layer``, every channel, read once."""
        hit = self._raw.get(id(adata))
        if hit is None:
            matrix = np.asarray(layer_matrix(adata, self.raw_layer), dtype=np.float64)
            rows = self._rows[id(adata)]
            hit = matrix if rows is None else matrix[rows]
            self._raw[id(adata)] = hit
        return hit

    def _rows_of(self, adata: ad.AnnData, mask: np.ndarray) -> np.ndarray:
        rows = self._rows[id(adata)]
        return mask if rows is None else mask[rows]

    def _matrix_rows(self, adata: ad.AnnData) -> np.ndarray:
        """The plotted events of ``raw_layer``, in the matrix's detectors only."""
        key = ("matrix", id(adata))
        hit = self._raw.get(key)
        if hit is None:
            cols = [find_channel_name(adata, n) for n in self.spillover.index]
            # float32: arcsinh over a million values is three times faster than
            # in float64, and this runs over every tile on every edit.
            hit = np.ascontiguousarray(self._raw_rows(adata)[:, cols], dtype=np.float32)
            self._raw[key] = hit
        return hit

    def compensated_matrix(self, adata: ad.AnnData, inverse: np.ndarray) -> np.ndarray:
        """Every matrix detector of the plotted events, compensated and transformed.

        The transform is the one :attr:`layer` recorded, so the check grid
        reads on the same axes as the gating plot did.

        Parameters
        ----------
        adata
            A control.
        inverse
            ``inv(spillover)``.

        Returns
        -------
        ndarray
            ``(events, detectors)`` display values, columns in the matrix's
            order.
        """
        out = self._matrix_rows(adata) @ inverse.astype(np.float32)
        # The arcsinh columns go through one vectorised call, each divided by
        # its own cofactor first; anything else through its scale.
        divisors = np.ones(out.shape[1], dtype=np.float32)
        asinh = np.zeros(out.shape[1], dtype=bool)
        for j, channel in enumerate(self.spillover.index):
            scale = axis_scale(adata, str(channel), self.layer)
            if not isinstance(scale, PretransformedScale):
                continue
            if isinstance(scale.inner, AsinhScale):
                divisors[j], asinh[j] = scale.inner.cofactor, True
            else:
                out[:, j] = scale.inner.forward(out[:, j])
        if asinh.all():
            np.arcsinh(out / divisors, out=out)
        elif asinh.any():
            out[:, asinh] = np.arcsinh(out[:, asinh] / divisors[asinh])
        return out

    def compensated(self, adata: ad.AnnData, channel: str, inverse: np.ndarray) -> np.ndarray:
        """One detector of :meth:`compensated_matrix`.

        Parameters
        ----------
        adata
            A control.
        channel
            ``var_name`` of a detector in the matrix.
        inverse
            ``inv(spillover)``.

        Returns
        -------
        ndarray
            Display values, one per plotted event.
        """
        names = list(self.spillover.index)
        return self.compensated_matrix(adata, inverse)[:, names.index(channel)]

    # -------------------------------------------------------------- rendering
    def refresh(self, *_) -> None:
        """Redraw the canvas for the current step."""
        if self._updating:
            return
        frames = []
        try:
            if self.step == STEPS[0]:
                frames = self._draw_gating()
            else:
                frames = self._draw_check()
        finally:
            self._draw_frames(frames)
        self._head()

    def _draw_gating(self):
        """One panel: the unstained and the control overlaid on the stained detector.

        The two are drawn in different colours on the same axis, the way the
        main window overlays several samples, so where the control's negatives
        sit against the unstained is read directly rather than across a gap.
        The second panel is not used in this step.
        """
        from .scales import LinearScale
        from .viewer import CURVE_COLOURS

        detector = self.control
        control = self.controls[detector]
        unstained_values = self._display_column(self.unstained, detector)
        control_values = self._display_column(control, detector)
        lo, hi = axis_limits(
            control, detector, self.layer, np.concatenate([unstained_values, control_values])
        )

        panel = self.panel
        panel.kind = "histogram"
        panel.x, panel.y = detector, ""
        panel.axes = Axes2D(lo, hi, 0.0, 1.0, bins=self.bins)
        panel.x_scale = axis_scale(control, detector, self.layer)
        panel.y_scale = LinearScale()
        panel.image.visible = False
        panel.curves.visible = True

        # The axis spans both tubes either way, so hiding the unstained takes
        # its curve away without moving the control's; the control keeps its
        # colour for the same reason.
        series = [(_sample(control, detector), control, control_values, CURVE_COLOURS[1])]
        if self.show_unstained:
            name = _sample(self.unstained, "unstained")
            series.insert(0, (name, self.unstained, unstained_values, CURVE_COLOURS[0]))
        legend = self._draw_curves(
            panel,
            [(name, self._subsampled(adata, v)) for name, adata, v, _ in series],
            [colour for *_, colour in series],
            lo,
            hi,
        )

        bands, faces = [], []
        gated = [
            (control, self.positive_gate, _POSITIVE, "positive"),
            (control, self.negative_gate, _NEGATIVE, "negative"),
        ]
        shares = []
        from_unstained = self.uses_unstained(detector)
        if from_unstained:
            # The control's own negative gate is not used, so it is not drawn.
            gated = gated[:1]
            shares_tail = ["negative: unstained"]
        else:
            shares_tail = []
        # (display value, colour) of each population's median, drawn as a line
        # through its band.
        medians: list[tuple[float, str]] = []
        for adata, name, colour, label in gated:
            editing = self.editing is not None and name == self._gate_name(self.editing)
            for a, b in [] if editing else self._intervals(adata, name, detector):
                band = self._band(panel, a, b)
                if band is not None:
                    bands.append(band)
                    faces.append(colour + "40")
            if name in adata.obs:
                mask = adata.obs[name].to_numpy(dtype=bool)
                shares.append(f"{label} {float(np.mean(mask)):.1%}")
                inside = control_values[mask]
                inside = inside[np.isfinite(inside)]
                if inside.size and not editing:
                    medians.append((float(np.median(inside)), colour))
        if from_unstained:
            # The whole tube is the negative, so there is no box to draw --
            # only where its median sits.
            everything = unstained_values[np.isfinite(unstained_values)]
            if everything.size:
                medians.append((float(np.median(everything)), _NEGATIVE))
        shares += shares_tail
        if shares:
            # On the control's line of the key, which is always in view; the
            # bands themselves carry the gates' colours.
            row, col, text, colour = legend[-1]
            legend[-1] = (row, col, f"{text}   ·   {'   '.join(shares)}", colour)

        self.bands.data = []
        if bands:
            self.bands.add(bands, shape_type="polygon", face_color=faces, edge_color=faces)
        self.bands.visible = bool(bands)

        lines, colours = [], []
        ax, b = panel.axes, self.bins
        for value, colour in medians:
            if not ax.x_lo <= value <= ax.x_hi:
                continue
            c = panel.col + ax.to_pixels(np.array([value]), np.array([ax.y_lo]))[0, 1]
            lines.append([[panel.row - 0.5, c], [float(b), 0.0]])
            colours.append(colour)
        self.guides.data = (
            np.asarray(lines, dtype=float) if lines else np.zeros((0, 2, 2), dtype=float)
        )
        if lines:
            self.guides.edge_color = colours
        self.guides.visible = bool(lines)
        title = f"{detector} — control against unstained" if self.show_unstained else detector
        return [(panel, title, "", True), legend]

    def _subsampled(self, adata: ad.AnnData, values: np.ndarray) -> np.ndarray:
        rows = self._rows[id(adata)]
        out = values if rows is None else values[rows]
        return out[np.isfinite(out)]

    def _band(self, panel, lo: float, hi: float):
        ax = panel.axes
        lo, hi = max(lo, ax.x_lo), min(hi, ax.x_hi)
        if hi <= lo:
            return None
        c0, c1 = panel.col + ax.to_pixels(np.array([lo, hi]), np.full(2, ax.y_lo))[:, 1]
        top, bottom = panel.row - 0.5, panel.row + self.bins - 0.5
        return np.array([[top, c0], [top, c1], [bottom, c1], [bottom, c0]])

    def _draw_curves(self, panel, series, colours, lo, hi) -> list:
        """One distribution per tube, each as per cent of its own mode.

        Drawn the way the main window draws a multi-sample histogram:
        translucent fills first, outlines on top, a key in the corner.

        Returns
        -------
        list
            ``(row, col, text)`` for the key's labels.
        """
        from .viewer import _MODE_TOP, CURVE_FILL_ALPHA

        b = self.bins
        scale = (b - 1) / (_MODE_TOP * 1.05)
        columns = np.arange(b, dtype=float) - 0.5 + panel.col
        floor = panel.row + b - 1.0
        fills, lines, labels = [], [], []
        for (name, values), colour in zip(series, colours):
            curve = density_curve(values, lo, hi, b, smooth=max(self.smooth, 0.5) * 2)
            peak = float(curve.max()) if curve.size else 0.0
            curve = curve / peak * _MODE_TOP if peak > 0 else curve
            line = np.column_stack([floor - curve * scale, columns])
            # Closed just below the floor, so a curve touching zero still has
            # area to triangulate.
            base = floor + 1.0
            fills.append(np.vstack([line, [[base, columns[-1]], [base, columns[0]]]]))
            lines.append(line)
            labels.append(f"{name} ({values.size:,})")

        shapes = fills + lines
        kinds = ["polygon"] * len(fills) + ["path"] * len(lines)
        faces = [c + CURVE_FILL_ALPHA for c in colours] + ["#00000000"] * len(lines)
        edges = ["#00000000"] * len(fills) + list(colours)
        # The key is the names themselves, each in its curve's colour, above
        # the plot: inside it, it would sit on a peak whichever corner it took.
        legend = [
            (panel.row - (0.22 - 0.06 * i) * b, panel.col + 0.5 * b, label, colour)
            for i, (colour, label) in enumerate(zip(colours, labels))
        ]

        panel.curves.data = []
        panel.curves.add(shapes, shape_type=kinds, face_color=faces, edge_color=edges, edge_width=2)
        return legend

    @property
    def _step(self) -> int:
        """Tile plus gap, in whole pixels, so every tile lands on the pixel grid."""
        return self.tile + max(1, round(self.tile * TILE_GAP))

    @property
    def _grid_left(self) -> int:
        """Canvas column where the first grid starts.

        The row labels go in the space before it, inside the canvas: napari
        frames the view on the image, and a label hanging off its edge is cut.
        """
        return round(1.8 * self.tile)

    @property
    def _scatter_left(self) -> int:
        """Canvas column where the scatter grid starts: half a tile after the first."""
        return self._grid_left + len(self.detectors) * self._step + self.tile // 2

    def tile_origin(self, i: int, j: int, scatter: bool = False) -> tuple[float, float]:
        """Canvas ``(row, col)`` of the top-left corner of tile ``(i, j)``.

        Parameters
        ----------
        i, j
            Row (control) and column (detector) of the grid.
        scatter
            The tile of the scatter grid rather than the fluorescence one.

        Returns
        -------
        tuple of float
        """
        step = self._step
        left = self._scatter_left if scatter else self._grid_left
        return float(i * step), float(left + j * step)

    def tile_at(self, row: float, col: float) -> tuple[str, str] | None:
        """``(control, detector)`` of the tile at a canvas point, in either grid.

        Both grids map a tile to the same coefficient: row *i*, column *j* is
        control *i*'s spill into detector *j* whichever y axis it is drawn on.

        Parameters
        ----------
        row, col
            Canvas coordinates.

        Returns
        -------
        tuple or None
            ``None`` between tiles, off the grids, or on a diagonal.
        """
        step = self._step
        n = len(self.detectors)
        scatter = self.scatter is not None and col >= self._scatter_left
        left = self._scatter_left if scatter else self._grid_left
        i, j = int(np.floor(row / step)), int(np.floor((col - left) / step))
        if not (0 <= i < n and 0 <= j < n):
            return None
        r0, c0 = self.tile_origin(i, j, scatter)
        if row - r0 > self.tile or col - c0 > self.tile:
            return None
        control, detector = self.detectors[i], self.detectors[j]
        return None if control == detector else (control, detector)

    def shown_channel(self, control: str, channel: str, shown: np.ndarray) -> np.ndarray:
        """Any channel of a control's plotted events, as the check step plots it.

        A detector in the matrix comes from ``shown``, the output of
        :meth:`compensated_matrix`. Anything else -- scatter -- is read from
        ``raw_layer`` and put through the transform :attr:`layer` recorded for
        it, which for scatter is usually none at all.

        Parameters
        ----------
        control
            The control's detector.
        channel
            ``var_name`` of the channel.
        shown
            :meth:`compensated_matrix` of that control.

        Returns
        -------
        ndarray
            Display values, one per plotted event.
        """
        return self._values(self.controls[control], channel, shown)

    def _values(self, adata: ad.AnnData, channel: str, shown: np.ndarray) -> np.ndarray:
        """:meth:`shown_channel`, for any tube -- the unstained included."""
        names = list(self.spillover.index)
        if channel in names:
            return shown[:, names.index(channel)]
        values = self._raw_rows(adata)[:, find_channel_name(adata, channel)]
        scale = axis_scale(adata, channel, self.layer)
        if isinstance(scale, PretransformedScale):
            return scale.inner.forward(values)
        return values

    def _draw_check(self):
        """The compensated controls, as the grids or as two plots."""
        try:
            inverse = np.linalg.inv(self.spillover.to_numpy(dtype=float))
        except np.linalg.LinAlgError:
            self.w_status.value = "the matrix is singular and cannot be inverted"
            return []
        if self.view == VIEWS[0]:
            return self._draw_grid(inverse)
        return self._draw_pair(inverse)

    def _show(self, canvas: np.ndarray, guides, colours) -> None:
        """Put a finished canvas and its median lines on screen."""
        from .viewer import faded_colormap

        panel = self.panel
        panel.kind = "density"
        panel.image.data = canvas
        panel.image.colormap = faded_colormap(self.colormap)
        panel.image.contrast_limits = (0.0, 1.0)
        panel.image.visible = True
        panel.curves.visible = False
        panel.curves.data = []
        self.guides.data = (
            np.asarray(guides, dtype=float) if guides else np.zeros((0, 2, 2), dtype=float)
        )
        if guides:
            self.guides.edge_color = colours
        self.guides.visible = bool(guides)

    def _tile_image(self, canvas, r0, c0, xv, yv, ax, size) -> None:
        """One density, scaled to its own peak, written into the canvas.

        Each on its own scale: a dim control would otherwise be a blank square
        next to a bright one.
        """
        image = density_image(xv, yv, ax, smooth=self.smooth, log=True)
        top = float(image.max())
        ri, ci = int(r0), int(c0)
        canvas[ri : ri + size, ci : ci + size] = image / top if top > 0 else image

    def _draw_grid(self, inverse: np.ndarray):
        """The N x N grid, and beside it the same rows against a scatter channel.

        Rows and columns are the matrix's detectors, which are exactly the
        ones there are controls for.

        In the first grid tile ``(i, j)`` has control *i*'s own detector on x and
        detector *j* on y; in the second, detector *j* on x and the scatter
        channel on y. Either way it is coefficient ``S[i, j]`` that moves the
        positive population, up in the first and sideways in the second.
        """
        names = list(self.spillover.index)
        columns = self.detectors
        n = len(columns)
        t = self.tile
        right = (self._scatter_left if self.scatter else self._grid_left) + (n - 1) * self._step + t
        canvas = np.zeros(((n - 1) * self._step + t, right), dtype=np.float32)
        lines, guides, colours, texts = [], [], [], []
        self._tile_axes: dict[tuple[str, str, bool], Axes2D] = {}

        for i, control_name in enumerate(self.detectors):
            control = self.controls[control_name]
            shown = self.compensated_matrix(control, inverse)
            own = shown[:, names.index(control_name)]
            own_lo, own_hi = self._axis(control_name, control_name, own)
            medians = self._population_medians(control_name, shown, inverse)
            if self.scatter:
                sv = self.shown_channel(control_name, self.scatter, shown)
                s_lo, s_hi = self._axis(control_name, self.scatter, sv)

            for j, detector in enumerate(columns):
                k = names.index(detector)
                values = shown[:, k]
                lo, hi = self._axis(control_name, detector, values)

                # Fluorescence: own detector on x, this one on y.
                r0, c0 = self.tile_origin(i, j)
                lines += _box(r0, c0, t)
                if detector == control_name:
                    texts.append((r0 + t / 2.0, c0 + t / 2.0, _short_name(detector)))
                else:
                    ax = Axes2D(own_lo, own_hi, lo, hi, bins=t)
                    self._tile_axes[(control_name, detector, False)] = ax
                    self._tile_image(canvas, r0, c0, own, values, ax, t)
                    for which, colour in (("positive", _POSITIVE), ("negative", _NEGATIVE)):
                        value = medians.get(which)
                        if value is None or not lo <= value[k] <= hi:
                            continue
                        r = r0 + ax.to_pixels(np.array([own_lo]), np.array([value[k]]))[0, 0]
                        guides.append([[r, c0 - 0.5], [0.0, float(t)]])
                        colours.append(colour)

                if not self.scatter:
                    continue
                # Scatter: this detector on x, scatter on y. Diagonal included:
                # it is the control's own split, which is worth seeing here.
                r0, c0 = self.tile_origin(i, j, scatter=True)
                lines += _box(r0, c0, t)
                ax = Axes2D(lo, hi, s_lo, s_hi, bins=t)
                self._tile_axes[(control_name, detector, True)] = ax
                self._tile_image(canvas, r0, c0, values, sv, ax, t)
                if detector == control_name:
                    continue
                for which, colour in (("positive", _POSITIVE), ("negative", _NEGATIVE)):
                    value = medians.get(which)
                    if value is None or not lo <= value[k] <= hi:
                        continue
                    c = c0 + ax.to_pixels(np.array([value[k]]), np.array([s_lo]))[0, 1]
                    guides.append([[r0 - 0.5, c], [float(t), 0.0]])
                    colours.append(colour)

        for i, control_name in enumerate(self.detectors):
            r0, _ = self.tile_origin(i, 0)
            texts.append((r0 + t / 2.0, self._grid_left - 0.9 * t, _short_name(control_name)))
        grids = [(False, "x = row's own detector  ·  y = column")]
        if self.scatter:
            grids.append((True, f"x = column  ·  y = {self.scatter}"))
        for scatter, title in grids:
            for j, detector in enumerate(columns):
                _, c0 = self.tile_origin(0, j, scatter)
                texts.append((-0.2 * t, c0 + t / 2.0, _short_name(detector)))
            _, first = self.tile_origin(0, 0, scatter)
            texts.append((-0.45 * t, first + (n * self._step) / 2.0, title))

        self._show(canvas, guides, colours)
        self._draw_focus()
        return [("lines", lines), texts]

    def _draw_pair(self, inverse: np.ndarray):
        """Two large biaxial plots of whatever was picked, with full axes."""
        from types import SimpleNamespace

        b = self._pair_size
        gutter = round(b * GUTTER)
        canvas = np.zeros((b, 2 * b + gutter), dtype=np.float32)
        frames, guides, colours, report = [], [], [], []
        self._tile_axes = {}
        #: The two plots' axes, in display coordinates, as last drawn.
        self.pair_axes: list[Axes2D] = []
        for index, state in enumerate(self.pair):
            name, x, y = state["control"], state["x"], state["y"]
            control = self.controls[name]
            shown = self.compensated_matrix(control, inverse)
            xv = self.shown_channel(name, x, shown)
            yv = self.shown_channel(name, y, shown)
            x_lo, x_hi = self._ranged(index, "x", name, x, self._axis(name, x, xv))
            y_lo, y_hi = self._ranged(index, "y", name, y, self._axis(name, y, yv))
            self._show_range(index, "x", name, x, (x_lo, x_hi))
            self._show_range(index, "y", name, y, (y_lo, y_hi))
            col = index * (b + gutter)
            ax = Axes2D(x_lo, x_hi, y_lo, y_hi, bins=b)
            self.pair_axes.append(ax)
            n = self.pair_bins[index]
            image = density_image(
                xv, yv, Axes2D(x_lo, x_hi, y_lo, y_hi, bins=n), smooth=self.smooth, log=True
            )
            top = float(image.max())
            image = image / top if top > 0 else image
            # Each bin drawn as a block of screen pixels, so the plot keeps its size.
            blocks = np.arange(b) * n // b
            canvas[:, col : col + b] = image[np.ix_(blocks, blocks)]

            populations = self._populations(name, shown, inverse)
            for which, colour in (("positive", _POSITIVE), ("negative", _NEGATIVE)):
                if which not in populations:
                    continue
                tube, matrix, mask = populations[which]
                value = float(np.median(self._values(tube, y, matrix)[mask]))
                if y_lo <= value <= y_hi:
                    r = ax.to_pixels(np.array([x_lo]), np.array([value]))[0, 0]
                    guides.append([[r, col - 0.5], [0.0, float(b)]])
                    colours.append(colour)
            if y in self.spillover.index and y != name:
                report.append(self.describe(name, y))

            frames.append(
                (
                    SimpleNamespace(
                        axes=ax,
                        x=x,
                        row=0.0,
                        col=float(col),
                        size=b,
                        x_scale=axis_scale(control, x, self.layer),
                        y_scale=axis_scale(control, y, self.layer),
                    ),
                    f"{_sample(control, name)} — compensated",
                    y,
                    False,
                )
            )

        self._show(canvas, guides, colours)
        self.bands.data = []
        self.bands.visible = False
        if report:
            self.w_status.value = "   ·   ".join(report)
        return frames

    def _scale_of(self, control: str, channel: str):
        """The transform behind a channel's axis, or ``None`` when it is linear."""
        scale = axis_scale(self.controls[control], channel, self.layer)
        return scale.inner if isinstance(scale, PretransformedScale) else None

    def _ranged(self, plot, axis, control, channel, auto) -> tuple[float, float]:
        """An axis's display range, with any typed ends in place of the automatic ones."""
        inner = self._scale_of(control, channel)
        ends = list(auto)
        for k, typed in enumerate(self.ranges[plot][axis]):
            if typed is not None:
                ends[k] = float(inner.forward(np.array([typed]))[0]) if inner else float(typed)
        if not ends[0] < ends[1]:
            # One end typed past the automatic other one.
            self.w_status.value = f"{axis} range is empty, so it is back on automatic"
            self.ranges[plot][axis] = [None, None]
            return auto
        return ends[0], ends[1]

    def _show_range(self, plot, axis, control, channel, ends) -> None:
        """Put the range the plot is drawn with into its boxes, in tick units."""
        inner = self._scale_of(control, channel)
        for k, (box, value) in enumerate(zip(self.w_range[plot][axis], ends)):
            raw = float(inner.inverse(np.array([value]))[0]) if inner else float(value)
            text = f"{raw:,.0f}" if abs(raw) >= 100 else f"{raw:.3g}"
            box.value = text
            self._shown[(plot, axis, k)] = text

    def _draw_focus(self) -> None:
        """Outline the tile being read, in both grids, and say what it shows."""
        self.bands.data = []
        self.bands.visible = False
        if self.focus is None:
            return
        control, detector = self.focus
        self.w_status.value = self.describe(control, detector)
        if self.view != VIEWS[0]:
            return
        t = self.tile
        boxes = []
        for scatter in (False, True):
            if (control, detector, scatter) not in getattr(self, "_tile_axes", {}):
                continue
            r0, c0 = self.tile_origin(
                self.detectors.index(control), self.detectors.index(detector), scatter
            )
            boxes.append(
                np.array([[r0 - 1, c0 - 1], [r0 - 1, c0 + t], [r0 + t, c0 + t], [r0 + t, c0 - 1]])
            )
        if boxes:
            self.bands.add(
                boxes,
                shape_type="polygon",
                face_color="#00000000",
                edge_color=_FOCUS,
                edge_width=3,
            )
            self.bands.visible = True

    def describe(self, control: str, detector: str) -> str:
        """One tile in words: the coefficient, and what it leaves behind.

        Parameters
        ----------
        control
            The row: a control's detector.
        detector
            The column.

        Returns
        -------
        str
            ``"<control> → <detector>: <coefficient>   pos − neg <gap>"``, the gap
            being the difference of the gated populations' medians in
            ``detector`` after compensation, in raw units; near zero when the
            coefficient is right. Without both gates there is no gap to give.
        """
        coefficient = float(self.spillover.loc[control, detector])
        out = f"{control} → {detector}: {coefficient:.4f}"
        adata = self.controls[control]
        positive, negative = self.gate_state(control)
        if not (positive and negative):
            return out + "   (gate this control to see its medians)"
        inverse = np.linalg.inv(self.spillover.to_numpy(dtype=float))
        column = inverse[:, list(self.spillover.index).index(detector)]
        pos = self._rows_of(adata, adata.obs[self.positive_gate].to_numpy(dtype=bool))
        if self.uses_unstained(control):
            negatives = self._matrix_rows(self.unstained) @ column
        else:
            neg = self._rows_of(adata, adata.obs[self.negative_gate].to_numpy(dtype=bool))
            negatives = (self._matrix_rows(adata) @ column)[neg]
        positives = (self._matrix_rows(adata) @ column)[pos]
        if not (positives.size and negatives.size):
            return out
        gap = float(np.median(positives) - np.median(negatives))
        return f"{out}   pos − neg {gap:+,.0f}"

    def _axis(self, detector: str, channel: str, values: np.ndarray) -> tuple[float, float]:
        """Axis range, fixed per control and channel so tuning cannot move it.

        Fluorescence follows the usual rule. A channel outside the matrix --
        scatter -- is fitted to the 0.1-99.9th percentile of the events instead
        of its detector's ``$PnR``: across 0..262144 a population is a strip
        along the bottom, and a saturated event or two would do the same. Only
        the picture is clipped; nothing is gated in this step.
        """
        key = (detector, channel)
        if key not in self._limits:
            finite = values[np.isfinite(values)]
            control = self.controls[detector]
            if channel in self.spillover.index:
                self._limits[key] = axis_limits(control, channel, self.layer, finite)
            else:
                scale = axis_scale(control, channel, self.layer)
                self._limits[key] = scale.limits(finite, quantiles=(0.001, 0.999))
        return self._limits[key]

    def _populations(self, detector: str, shown: np.ndarray, inverse: np.ndarray) -> dict:
        """``{which: (tube, its shown matrix, mask over its plotted events)}``.

        The positive population is always the control's gate. The negative is
        the control's gate too, unless the control uses the unstained tube, in
        which case it is every plotted event of that tube, compensated the same
        way.
        """
        control = self.controls[detector]
        out = {}
        if self.positive_gate in control.obs:
            mask = self._rows_of(control, control.obs[self.positive_gate].to_numpy(dtype=bool))
            out["positive"] = (control, shown, mask)
        if self.uses_unstained(detector):
            tube = self.unstained
            matrix = self.compensated_matrix(tube, inverse)
            out["negative"] = (tube, matrix, np.ones(matrix.shape[0], dtype=bool))
        elif self.negative_gate in control.obs:
            mask = self._rows_of(control, control.obs[self.negative_gate].to_numpy(dtype=bool))
            out["negative"] = (control, shown, mask)
        return {k: v for k, v in out.items() if v[2].any()}

    def _population_medians(self, detector: str, shown: np.ndarray, inverse: np.ndarray) -> dict:
        """Median display value of the two populations, per matrix detector.

        A control compensated correctly has the two level in every detector
        but its own, which is what the two lines across each tile show.
        """
        return {
            which: np.median(matrix[mask], axis=0)
            for which, (_, matrix, mask) in self._populations(detector, shown, inverse).items()
        }

    def _draw_frames(self, frames) -> None:
        """A labelled box around each panel, plus any loose labels."""
        from .viewer import _TEXT_STYLE, Y_LABEL_ROTATION, _panel_frame

        lines: list[np.ndarray] = []
        coords: list[list[float]] = []
        text: list[str] = []
        colours: list[str] = []
        y_coords: list[list[float]] = []
        y_text: list[str] = []
        for frame in frames:
            if isinstance(frame, list):  # loose labels
                for row, col, label, *colour in frame:
                    coords.append([row, col])
                    text.append(label)
                    colours.append(colour[0] if colour else self._foreground)
                continue
            if frame[0] == "lines":  # segments drawn by the caller
                lines += frame[1]
                continue
            panel, title, y_label, mode = frame
            # A plot can be bigger than the window's own bins; the two plots
            # say so, everything else is drawn at that size.
            size = getattr(panel, "size", self.bins)
            tick = 0.02 * size
            pl, pc, pt, yl = _panel_frame(
                panel.axes,
                x_scale=panel.x_scale,
                y_scale=panel.y_scale,
                bins=size,
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
            colours += [self._foreground] * len(pt)
            y_coords.append([panel.row + size / 2.0, panel.col - 12.0 * tick])
            y_text.append(yl)

        if lines:
            ends = np.asarray(lines, dtype=float)
            self.grid.data = np.stack([ends[:, 0], ends[:, 1] - ends[:, 0]], axis=1)
        else:
            self.grid.data = np.zeros((0, 2, 2))
        self.labels.data = np.asarray(coords, dtype=float) if coords else np.empty((0, 2))
        self.labels.text = {**_TEXT_STYLE, "string": text, "color": colours or self._foreground}
        self.ylabel.data = np.asarray(y_coords, dtype=float) if y_coords else np.empty((0, 2))
        self.ylabel.text = {
            **_TEXT_STYLE,
            "string": y_text,
            "color": self._foreground,
            "rotation": Y_LABEL_ROTATION,
        }

    def _head(self) -> None:
        edited = " · edited" if self.edited else ""
        if self.gates_changed and self.source == "computed":
            edited += " · gates changed since, compute again"
        done = len(self.detectors) - len(self.missing_gates())
        self.w_head.value = (
            f"matrix: {self.source}{edited}   —   {done} / {len(self.detectors)} controls gated"
        )


def _diagonal_square(lo: float, hi: float) -> list[list[float]]:
    return [[lo, lo], [hi, lo], [hi, hi], [lo, hi]]


def _box(row: float, col: float, size: int) -> list[np.ndarray]:
    """The four edges of a tile, as segments."""
    top, left = row - 0.5, col - 0.5
    bottom, right = top + size, left + size
    return [
        np.array([[top, left], [top, right]]),
        np.array([[bottom, left], [bottom, right]]),
        np.array([[top, left], [bottom, left]]),
        np.array([[top, right], [bottom, right]]),
    ]


def _drop_gate(adata: ad.AnnData, name: str) -> None:
    """Remove a gate's column and its record."""
    adata.obs.drop(columns=[name], inplace=True, errors="ignore")
    adata.uns.get("cykit", {}).get("gates", {}).pop(name, None)


def _scatter_channels(adata: ad.AnnData) -> list[str]:
    """Scatter channels, by the ``kind`` read_fcs records, else by name."""
    if "kind" in adata.var:
        found = [str(n) for n, k in zip(adata.var_names, adata.var["kind"]) if k == "scatter"]
        if found:
            return found
    return [str(n) for n in adata.var_names if str(n).upper().startswith(("FSC", "SSC"))]


def _default_scatter(adata: ad.AnnData) -> str | None:
    """``SSC-A`` when there is one -- it is what cytometrists read spread against."""
    channels = _scatter_channels(adata)
    for name in channels:
        if name.upper().startswith("SSC") and name.upper().endswith("-A"):
            return name
    return channels[0] if channels else None


def _short_name(name: str) -> str:
    """``"CD3 (FITC-A)"`` stays as it is; only very long names are cut."""
    return name if len(name) <= 18 else name[:17] + "…"


def _sample(adata: ad.AnnData, fallback: str) -> str:
    if "sample" in adata.obs and adata.n_obs:
        return str(adata.obs["sample"].iloc[0])
    return fallback


#: The window this session last opened, so a non-blocking call does not let it
#: be collected and :func:`current_compensation_window` has something to hand back.
_CURRENT_COMPENSATION: CompensationWindow | None = None


def current_compensation_window() -> CompensationWindow | None:
    """The window :func:`open_napari_compensation` last opened.

    Returns
    -------
    CompensationWindow or None
        ``None`` before one has been opened in this session.
    """
    return _CURRENT_COMPENSATION


def open_napari_compensation(
    controls: Mapping[str, ad.AnnData],
    unstained: ad.AnnData,
    layer: str,
    *,
    spillover="file",
    raw_layer: str = "raw",
    positive_gate: str = "positive",
    negative_gate: str = "negative",
    statistic: str = "median",
    max_events: int | None = MAX_EVENTS,
    tile_bins: int = 64,
    block: bool | None = None,
) -> pd.DataFrame:
    """Open the compensation window, and hand the spillover matrix back.

    Gate each single-stain control against the unstained, compute the matrix,
    then edit it while watching the compensated controls. When the controls
    carry the instrument's ``$SPILLOVER`` it is loaded into the table and the
    window opens ready to tune it. See :class:`CompensationWindow`.

    Parameters
    ----------
    controls
        Detector to single-stain control, as ``AnnData``; gates are written
        onto them.
    unstained
        The unstained control.
    layer
        Transformed layer to display and gate on, e.g. ``"asinh"``.
    spillover
        Starting matrix: ``"file"`` (the controls' ``$SPILLOVER``), a
        DataFrame, or ``None`` for the identity.
    raw_layer
        Layer the matrix is computed from and applied to.
    positive_gate, negative_gate
        Gate names, as :func:`~cykit.compute_spillover_matrix` takes them.
    statistic
        ``"median"`` or ``"mean"``.
    max_events
        Events plotted per tube.
    tile_bins
        Resolution of each tile of the N x N check grid.
    block
        Wait for the window to close before returning. The default detects the
        context: ``True`` from a script, ``False`` under IPython.

        A non-blocking call returns before you have done anything. The
        DataFrame it returns is the live one, edited in place as you work, so
        re-evaluating it in a later cell shows the current matrix.

    Returns
    -------
    DataFrame
        The spillover matrix, ready for :func:`~cykit.compensate`.
    """
    global _CURRENT_COMPENSATION
    import napari

    _CURRENT_COMPENSATION = CompensationWindow(
        controls,
        unstained,
        layer,
        spillover=spillover,
        raw_layer=raw_layer,
        positive_gate=positive_gate,
        negative_gate=negative_gate,
        statistic=statistic,
        max_events=max_events,
        tile_bins=tile_bins,
    )
    if block is None:
        try:
            get_ipython  # type: ignore[name-defined]  # noqa: B018
            block = False
        except NameError:
            block = True
    if block:
        napari.run()
    return _CURRENT_COMPENSATION.spillover
