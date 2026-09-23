"""Static matplotlib figures: biaxial plots, gate overlays, and QC lines.

The napari viewer is for looking around; this is for the figures that go into
a report and stay put. Both draw the same thing the same way -- a smoothed 2-D
histogram from :func:`~cytopy.density_image` -- so a plot in a report matches
what was on screen when the gate was drawn.

Biaxial plots are coloured by event density by default. At a few million
events a per-point scatter is neither fast nor readable: the interesting
structure is where events pile up, and only a density does that honestly.
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np

from ._util import LAYER_X, layer_matrix, subsample_indices
from .density import Axes2D, density_image
from .scales import (
    AsinhScale,
    LinearScale,
    LogicleScale,
    PretransformedScale,
    Scale,
    pad_range,
)
from .transforms import channel_index

__all__ = ["plot_biaxial", "plot_compensation", "plot_gate"]

#: Events drawn when a plot colours points individually rather than by density.
MAX_SCATTER = 50_000


def axis_scale(adata: ad.AnnData, channel: str, layer: str | None) -> Scale:
    """How to label an axis for ``channel`` in ``layer``, in raw units if known.

    The same rule the viewer uses: a layer that recorded the transform which
    produced it gets ticks at decades of the original units, and anything else
    gets the stored values.

    Parameters
    ----------
    adata
        The AnnData being plotted.
    channel
        Channel on the axis.
    layer
        Layer being plotted. ``None``, ``""`` and ``"X"`` all mean
        ``adata.X``, which carries no record of a transform and so is
        labelled with the values as stored.

    Returns
    -------
    Scale
        Maps plotted values to themselves; its inner transform, when it has
        one, places the raw-unit ticks.
    """
    if layer in (None, "", LAYER_X):
        return LinearScale()
    info = adata.uns.get("cytopy", {})
    j = channel_index(adata, channel)
    name = str(adata.var_names[j])
    # Per-layer first: a file transformed twice keeps both.
    cofactor = info.get("asinh_layers", {}).get(layer, {}).get(name)
    if cofactor is not None:
        return PretransformedScale(AsinhScale(cofactor=float(cofactor)))
    params = info.get("logicle_layers", {}).get(layer, {}).get(name)
    if params is not None:
        return PretransformedScale(LogicleScale(**dict(params)))
    if layer == info.get("asinh_layer") and "cofactor" in adata.var:
        cofactor = float(adata.var["cofactor"].iloc[j])
        if np.isfinite(cofactor):
            return PretransformedScale(AsinhScale(cofactor=cofactor))
    if layer == info.get("logicle_layer"):
        params = info.get("logicle_params", {}).get(str(adata.var_names[j]))
        if params:
            return PretransformedScale(LogicleScale(**params))
    return LinearScale()


def instrument_range(adata: ad.AnnData, channel: str) -> tuple[float, float] | None:
    """The detector's full range for ``channel``, from the FCS ``$PnR`` keyword.

    Parameters
    ----------
    adata
        The AnnData; ``adata.var['pnr']`` is written by :func:`~cytopy.read_fcs`.
    channel
        Channel to look up.

    Returns
    -------
    tuple of float or None
        ``(0.0, pnr)``, or ``None`` when the file recorded no usable range --
        an AnnData built by hand, or a ``$PnR`` that is missing, non-numeric or
        not positive.
    """
    if "pnr" not in adata.var:
        return None
    try:
        top = float(adata.var["pnr"].iloc[channel_index(adata, channel)])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(top) or top <= 0:
        return None
    return (0.0, top)


def axis_limits(
    adata: ad.AnnData,
    channel: str,
    layer: str | None,
    values: np.ndarray,
    *,
    robust: bool = False,
) -> tuple[float, float]:
    """Display-coordinate range for an axis. The same rule the viewer uses.

    A channel the layer never transformed keeps its raw instrument units, and
    those have a range the file already states: ``$PnR``. Using it means the
    scatter channels -- which share a ``$PnR`` -- get identical axes, the way a
    cytometrist expects an FSC/SSC plot to look, and that the ticks land on
    round numbers spanning the detector rather than on whatever this particular
    sample happened to reach.

    Anything else, including every transformed axis, falls back to the spread
    of the data. So does a linear channel whose values do not fit in ``[0,
    $PnR]``: compensation pushes events negative, and clipping those off the
    bottom of the plot would hide them.

    Parameters
    ----------
    adata
        The AnnData being plotted.
    channel
        Channel on the axis.
    layer
        Layer being plotted; ``None``, ``""`` and ``"X"`` all mean ``adata.X``.
    values
        The values to be shown on this axis, as stored in ``layer``.
    robust
        Clip the fallback range to the 0.1-99.9th percentile, so that a handful
        of extreme events cannot flatten the plot. Off by default: on a panel
        of this size 0.1% a side is thousands of events, and they are dropped
        from the plot and from every gate, not merely pushed out of frame.

    Returns
    -------
    tuple of float
        ``(lo, hi)`` in display coordinates, padded slightly at each end.
    """
    scale = axis_scale(adata, channel, layer)
    quantiles = (0.001, 0.999) if robust else (0.0, 1.0)
    fallback = scale.limits(values, quantiles=quantiles)
    if not isinstance(scale, LinearScale):
        return fallback
    span = instrument_range(adata, channel)
    if span is None:
        return fallback
    # Whether the detector's range can hold the data is a question about the
    # data, so it is asked of the unpadded extent. Asking it of `fallback`
    # would let AXIS_MARGIN decide: pad generously enough and every channel
    # spills past $PnR, and the scatter channels stop sharing an axis.
    lo, hi = _extent(values, quantiles)
    if lo < span[0] or hi > span[1]:
        return fallback
    return pad_range(*span)


def _extent(values: np.ndarray, quantiles: tuple[float, float]) -> tuple[float, float]:
    """The span of ``values`` between ``quantiles``, before any padding."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    lo = float(np.quantile(v, quantiles[0])) if quantiles[0] > 0 else float(v.min())
    hi = float(np.quantile(v, quantiles[1])) if quantiles[1] < 1 else float(v.max())
    return (lo, hi)


def _apply_ticks(axis, scale: Scale, lo: float, hi: float, which: str) -> None:
    ticks = scale.ticks(lo, hi)
    setter = axis.set_xticks if which == "x" else axis.set_yticks
    setter(ticks.major, ticks.labels)
    if len(ticks.minor):
        setter(ticks.minor, minor=True)


def plot_biaxial(
    adata: ad.AnnData,
    x: str,
    y: str,
    *,
    layer: str,
    cofactor: float | None = None,
    color_by: str = "density",
    subset: np.ndarray | str | None = None,
    gate: str | Sequence[str] | None = None,
    rectangles: Sequence[Sequence[float]] | None = None,
    bins: int = 512,
    smooth: float = 1.0,
    log: bool = True,
    cmap: str = "turbo",
    robust: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    seed: int = 0,
    ax=None,
):
    """A two-channel plot, coloured by event density.

    Parameters
    ----------
    adata
        Events x channels AnnData.
    x, y
        Channels for the two axes. Any alias :func:`~cytopy.channel_index`
        accepts.
    layer
        Matrix to plot, by name -- ``"raw"``, ``"comp"``, ``"asinh"``, or
        ``"X"`` for the working matrix. Required: a plot should say which
        matrix it is of. Values are plotted as stored;
        the axes are labelled in raw units when the layer records its
        transform, exactly as in the viewer.
    cofactor
        Arcsinh the two channels for display only, and put the ticks back in
        raw units. Nothing is written to ``adata`` -- this is for plotting raw
        counts on a biexponential axis without materialising a layer for it,
        which on a run of millions of events is worth avoiding.
    color_by
        ``"density"`` (default) shades a smoothed 2-D histogram, which is the
        only thing that stays readable at millions of events. Otherwise the
        name of a boolean ``obs`` column -- a gate, or the bead mask -- which
        draws that group as points over a grey density of everything else,
        so a report can show what a step removed.
    subset
        Boolean mask, or the name of a boolean ``obs`` column, limiting the
        events drawn. ``None`` draws all of them.
    gate
        Gate name, or names, whose outline to draw over the plot. Gates
        recorded on channels other than ``x`` and ``y`` are skipped, since
        their outline would not mean anything here.
    rectangles
        Outlines to draw as ``(x_lo, x_hi, y_lo, y_hi)``, in plotted
        coordinates -- for a gate held as ranges rather than vertices, such as
        a bead gate.
    bins
        Resolution of the density, per axis.
    smooth
        Gaussian smoothing of the density, in bins.
    log
        Shade ``log1p(count)``, so rare populations stay visible.
    cmap
        Colormap for the density.
    robust
        Clip the axes to the 0.1-99.9th percentile, so a few extreme events
        cannot flatten the plot. Ignored on whichever axis has an explicit
        ``xlim``/``ylim``.
    xlim, ylim
        Explicit display-coordinate limits, overriding the ones ``robust``
        would otherwise compute. For sharing one scale across several panels,
        e.g. the same detector's axis across every control in
        :func:`plot_compensation`.
    title
        Axes title. Defaults to the event count.
    seed
        Seed for thinning a highlighted population down to ``MAX_SCATTER``
        points, so the same events are drawn each time.
    ax
        Existing axes to draw on; one is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    KeyError
        If ``color_by`` or ``subset`` names a column that is not in ``obs``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.0))

    matrix = layer_matrix(adata, layer)
    xi, yi = channel_index(adata, x), channel_index(adata, y)
    x_name, y_name = str(adata.var_names[xi]), str(adata.var_names[yi])

    mask = _resolve_mask(adata, subset)
    xv = np.asarray(matrix[:, xi], dtype=np.float64).ravel()[mask]
    yv = np.asarray(matrix[:, yi], dtype=np.float64).ravel()[mask]

    if cofactor is not None:
        xv = np.arcsinh(xv / cofactor)
        yv = np.arcsinh(yv / cofactor)
        x_scale = y_scale = PretransformedScale(AsinhScale(cofactor=float(cofactor)))
    else:
        x_scale = axis_scale(adata, x_name, layer)
        y_scale = axis_scale(adata, y_name, layer)
    # A display-only cofactor puts both axes on a scale `adata` knows nothing
    # about, so the instrument range that `axis_limits` would reach for does
    # not apply; the transformed values are all there is to go on.
    quantiles = (0.001, 0.999) if robust else (0.0, 1.0)
    if xlim is not None:
        x_lo, x_hi = xlim
    elif xv.size == 0:
        x_lo, x_hi = 0.0, 1.0
    elif cofactor is not None:
        x_lo, x_hi = x_scale.limits(xv, quantiles=quantiles)
    else:
        x_lo, x_hi = axis_limits(adata, x_name, layer, xv, robust=robust)
    if ylim is not None:
        y_lo, y_hi = ylim
    elif yv.size == 0:
        y_lo, y_hi = 0.0, 1.0
    elif cofactor is not None:
        y_lo, y_hi = y_scale.limits(yv, quantiles=quantiles)
    else:
        y_lo, y_hi = axis_limits(adata, y_name, layer, yv, robust=robust)
    axes2d = Axes2D(x_lo, x_hi, y_lo, y_hi, bins=bins)

    highlight = None
    if color_by != "density":
        if color_by not in adata.obs:
            raise KeyError(f"no obs column {color_by!r} to colour by")
        highlight = np.asarray(adata.obs[color_by], dtype=bool)[mask]

    # The backdrop is every event in the subset; a highlight is drawn on top.
    image = density_image(
        xv if highlight is None else xv[~highlight],
        yv if highlight is None else yv[~highlight],
        axes2d,
        smooth=smooth,
        log=log,
    )
    # Empty bins are left blank rather than painted the colormap's darkest
    # colour, so a figure reads on a white page the way the canvas does.
    shaded = plt.get_cmap(cmap if highlight is None else "Greys").with_extremes(bad=(0, 0, 0, 0))
    ax.imshow(
        np.where(image > 0, image, np.nan),
        extent=(x_lo, x_hi, y_lo, y_hi),
        origin="upper",
        aspect="auto",
        cmap=shaded,
        interpolation="nearest",
    )
    if highlight is not None:
        hx, hy = xv[highlight], yv[highlight]
        take = subsample_indices(hx.size, MAX_SCATTER, rng=np.random.default_rng(seed))
        if take is not None:
            hx, hy = hx[take], hy[take]
        ax.scatter(hx, hy, s=1, c="#d62728", linewidths=0, alpha=0.5, label=color_by)
        ax.legend(loc="upper right", fontsize=7, frameon=False, markerscale=6)

    for name in _gate_names(gate):
        _draw_gate(ax, adata, name, x_name, y_name)
    for left, right, bottom, top in rectangles or []:
        ax.plot(
            [left, right, right, left, left],
            [bottom, bottom, top, top, bottom],
            color="#1f77b4",
            lw=1.2,
            ls="--",
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    _apply_ticks(ax, x_scale, x_lo, x_hi, "x")
    _apply_ticks(ax, y_scale, y_lo, y_hi, "y")
    ax.set_xlabel(x_name, fontsize=8)
    ax.set_ylabel(y_name, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title if title is not None else f"{int(xv.size):,} events", fontsize=9, loc="left")
    return ax


def _resolve_mask(adata: ad.AnnData, subset) -> np.ndarray:
    if subset is None:
        return np.ones(adata.n_obs, dtype=bool)
    if isinstance(subset, str):
        if subset not in adata.obs:
            raise KeyError(f"no obs column {subset!r} to subset on")
        return np.asarray(adata.obs[subset], dtype=bool)
    mask = np.asarray(subset, dtype=bool)
    if mask.shape[0] != adata.n_obs:
        raise ValueError(f"subset has {mask.shape[0]} entries, expected {adata.n_obs}")
    return mask


def _gate_names(gate) -> list[str]:
    if gate is None:
        return []
    return [gate] if isinstance(gate, str) else list(gate)


def _draw_gate(ax, adata: ad.AnnData, name: str, x_name: str, y_name: str) -> None:
    """Outline a recorded gate, if it was drawn on these two channels."""
    from .gating import gate_record

    if name not in adata.uns.get("cytopy", {}).get("gates", {}):
        return
    record = gate_record(adata, name)
    if record.x != x_name or record.y != y_name:
        return
    for shape in record.vertices:
        if shape.shape[0] < 2:
            continue
        closed = np.vstack([shape, shape[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="#1f77b4", lw=1.2, ls="--")
    ax.annotate(
        f"{name}: {record.n:,}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=7,
        color="#1f77b4",
    )


def plot_gate(adata: ad.AnnData, name: str, *, layer: str | None = None, ax=None, **kwargs):
    """A gate in the plane it was drawn in, with the events it kept picked out.

    Reads the channels, layer and outline back from
    ``adata.uns['cytopy']['gates'][name]``, so a gate drawn in the viewer
    weeks ago redraws exactly where it was.

    Parameters
    ----------
    adata
        AnnData carrying the gate as a boolean ``obs`` column.
    name
        The gate to draw.
    layer
        Override the layer the gate was recorded on.
    ax
        Existing axes to draw on.
    **kwargs
        Passed to :func:`plot_biaxial`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    KeyError
        If the gate was never recorded, or did not record its channels.
    """
    from .gating import gate_record

    record = gate_record(adata, name)
    if not record.x or not record.y:
        raise KeyError(f"gate {name!r} did not record which channels it was drawn on")
    if layer is None:
        layer = record.layer
    kwargs.setdefault("color_by", name)
    kwargs.setdefault("subset", record.parent or None)
    kwargs.setdefault("title", f"{name} ({record.n:,} events)")
    return plot_biaxial(adata, record.x, record.y, layer=layer, gate=name, ax=ax, **kwargs)


def plot_compensation(
    controls,
    spillover,
    *,
    unstained=None,
    cofactor: float = 150.0,
    positive_gate: str | None = None,
    bins: int = 256,
    max_events: int | None = 100_000,
    seed: int = 0,
    **kwargs,
):
    """Each single-stain control after compensation, one panel per detector.

    The picture every cytometrist checks a matrix against: the dye on the x
    axis, another detector on the y, and the positive population level with the
    negative one. A population that tips up is under-compensated and one that
    tips down is over-compensated, and the dashed line -- the negative
    population's level -- is what it should be sitting on.

    Parameters
    ----------
    controls
        Maps detector to its single-stain control, as
        :func:`~cytopy.read_controls` returns.
    spillover
        The matrix to try, in any form :func:`~cytopy.compensate` accepts.
    unstained
        Universal negative, drawn nowhere but used to place the line when a
        control has no negative events of its own.
    cofactor
        Arcsinh cofactor for the display. Display only -- compensation happens
        on the raw values.
    positive_gate
        ``obs`` column marking each control's positive events. Falls back to
        splitting the stained channel at its midpoint, which is only used to
        place the reference line.
    bins
        Resolution of each panel's density.
    max_events
        Events drawn per panel. ``None`` draws all of them; either way, axis
        ranges are always taken from every event, not just the ones drawn.
    seed
        Seed for that per-panel draw, so the same events are plotted each time.
    **kwargs
        Passed to :func:`plot_biaxial`. ``xlim``/``ylim`` are already set by
        this function -- to the row's own full range on x, and on y to the
        full range of that detector across every row -- and cannot be
        overridden here.

    Returns
    -------
    matplotlib.figure.Figure
        A row per control, a column per detector.

    Raises
    ------
    ValueError
        If ``controls`` is empty.
    """
    import matplotlib.pyplot as plt

    from .spillover import _as_anndata, compensate

    if not controls:
        raise ValueError("no controls to plot")
    compensated = {key: _as_anndata(value) for key, value in controls.items()}
    for control in compensated.values():
        compensate(control, spillover, key_added="_comp", inplace=True)
    if unstained is not None:
        unstained = compensate(unstained, spillover, key_added="_comp")
    names = list(compensated)
    first = next(iter(compensated.values()))
    detectors = [str(first.var_names[channel_index(first, name)]) for name in names]
    stained = [
        str(compensated[dye].var_names[channel_index(compensated[dye], dye)]) for dye in names
    ]

    # One axis range per row and per column, from every event rather than a
    # percentile, so a plot never lies about how far a population spreads --
    # and shared across a column so the same detector means the same thing
    # from row to row instead of each panel silently rescaling itself.
    row_xlim = [
        _channel_range(compensated[dye], stained[row], cofactor) for row, dye in enumerate(names)
    ]
    col_ylim = [
        _merge_ranges(
            _channel_range(compensated[dye], detector, cofactor, pad=False) for dye in names
        )
        for detector in detectors
    ]

    fig, axes = plt.subplots(
        len(names),
        len(detectors),
        figsize=(3.0 * len(detectors), 2.9 * len(names)),
        squeeze=False,
    )
    for row, dye in enumerate(names):
        adata = compensated[dye]
        positive = _above_median(adata, stained[row], positive_gate)
        for col, detector in enumerate(detectors):
            ax = axes[row][col]
            plot_biaxial(
                adata,
                stained[row],
                detector,
                layer="_comp",
                cofactor=cofactor,
                bins=bins,
                subset=_sample_of(adata, max_events, seed),
                xlim=row_xlim[row],
                ylim=col_ylim[col],
                title=f"{dye.split(' (')[0]} → {detector.split(' (')[0]}"
                if row != col
                else f"{dye.split(' (')[0]}",
                ax=ax,
                **kwargs,
            )
            if row == col:
                continue
            level = _negative_level(adata, detector, positive, unstained, cofactor)
            if level is not None:
                ax.axhline(level, color="#d62728", lw=1.0, ls="--")
    fig.tight_layout()
    return fig


def _above_median(adata, stained: str, positive_gate: str | None) -> np.ndarray:
    """Which events are the stained population, for placing the reference line."""
    if positive_gate and positive_gate in adata.obs:
        return np.asarray(adata.obs[positive_gate], dtype=bool)
    column = np.asarray(adata.layers["_comp"][:, channel_index(adata, stained)], dtype=np.float64)
    return column > np.median(column)


def _channel_range(
    adata, channel: str, cofactor: float | None, *, pad: bool = True
) -> tuple[float, float]:
    """The full display-coordinate span of ``channel``, from every event.

    Unlike :meth:`~cytopy.scales.Scale.limits` with a robust quantile, this
    never clips -- the point is to show exactly how far a population spreads,
    not to hide the tail.
    """
    values = np.asarray(
        adata.layers["_comp"][:, channel_index(adata, channel)], dtype=np.float64
    ).ravel()
    values = (
        np.arcsinh(values / cofactor)
        if cofactor is not None
        else axis_scale(adata, channel, "_comp").forward(values)
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    lo, hi = float(values.min()), float(values.max())
    return pad_range(lo, hi) if pad else (lo, hi)


def _merge_ranges(ranges) -> tuple[float, float]:
    """Combine several unpadded ``(lo, hi)`` spans into one, padded once."""
    los, his = zip(*ranges)
    return pad_range(min(los), max(his))


def _sample_of(adata, max_events: int | None, seed: int = 0):
    take = subsample_indices(adata.n_obs, max_events, rng=np.random.default_rng(seed))
    if take is None:
        return None
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[take] = True
    return mask


def _negative_level(adata, detector, positive, unstained, cofactor) -> float | None:
    """Where the positive population should sit: level with the negative one.

    The control's own negatives first -- same beads, same autofluorescence --
    falling back to a universal negative for a tube that has none of its own.
    """
    source, mask = adata, ~positive
    if int(mask.sum()) < 10:
        if unstained is None or "_comp" not in unstained.layers:
            return None
        source, mask = unstained, np.ones(unstained.n_obs, dtype=bool)
    values = np.asarray(
        source.layers["_comp"][:, channel_index(source, detector)], dtype=np.float64
    )[mask]
    return float(np.arcsinh(np.median(values) / cofactor))
