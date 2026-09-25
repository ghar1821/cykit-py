"""Viewer tests. Run headless with QT_QPA_PLATFORM=offscreen."""

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")


@pytest.fixture
def cv(demo, make_napari_viewer):
    import cytopy

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    from cytopy.viewer import CytoViewer

    return CytoViewer(
        demo,
        layer="asinh",
        x="CD3 (FITC-A)",
        y="CD19 (PE-A)",
        bins=128,
        viewer=make_napari_viewer(),
    )


def test_layers_exist(cv):
    assert set(cv.viewer.layers) >= set()
    names = [layer.name for layer in cv.viewer.layers]
    assert names == [
        "panel 1",
        "panel 1 curves",
        "axes",
        "axis labels",
        "y axis label",
        "applied gates",
        "gates",
    ]


def test_density_image_is_populated(cv):
    img = cv.density.data
    assert img.shape == (128, 128)
    assert img.max() > 0
    assert np.isfinite(img).all()


def _labels(cv):
    """Every label on the canvas, including the y axis on its own layer."""
    return list(cv.viewer.layers["axis labels"].text.string.array) + list(
        cv.viewer.layers["y axis label"].text.string.array
    )


def test_axis_labels_are_raw_units_for_a_transformed_layer(cv):
    text = _labels(cv)
    assert "CD3 (FITC-A)" in text and "CD19 (PE-A)" in text
    assert "0" in text
    assert any(t.startswith("10") for t in text)


def test_switching_channels_redraws(cv):
    before = cv.density.data.copy()
    cv.w_y.value = "CD8 (APC-A)"
    assert not np.array_equal(before, cv.density.data)


def test_transformed_ticks_label_the_stored_values(cv):
    from cytopy.scales import LinearScale, PretransformedScale

    assert isinstance(cv.x_scale, PretransformedScale)
    cv.w_ticks.value = "transformed"
    assert isinstance(cv.x_scale, LinearScale)
    # stored values are single-digit arcsinh units, not decades of raw signal
    assert not any(lbl.startswith("10") for lbl in _labels(cv))


def test_the_viewer_never_transforms_the_data(cv):
    """What is plotted must be exactly what is stored in the layer."""
    import cytopy

    mask = cv.selection_mask()
    j = cytopy.channel_index(cv.adata, "CD3 (FITC-A)")
    assert np.allclose(cv._column("CD3 (FITC-A)", mask), cv.adata.layers["asinh"][mask, j])
    # ... and switching the tick mode must not touch the values either
    cv.w_ticks.value = "transformed"
    assert np.allclose(cv._column("CD3 (FITC-A)", mask), cv.adata.layers["asinh"][mask, j])


def test_raw_X_gets_linear_ticks(cv):
    from cytopy.scales import LinearScale

    cv.w_layer.value = "X"
    assert isinstance(cv.x_scale, LinearScale)
    assert "records no transform" in cv.w_transform.value


def test_swap_axes(cv):
    x, y = cv.w_x.value, cv.w_y.value
    cv.w_swap.clicked()
    assert (cv.w_x.value, cv.w_y.value) == (y, x)


def test_axes_roundtrip_pixels_and_display(cv):
    disp = np.array([[0.2, 0.3], [0.8, 0.55]])
    rc = cv.axes.to_pixels(disp[:, 0], disp[:, 1])
    back = cv.axes.to_display(rc)
    assert np.allclose(back, disp, atol=1e-6)


def test_gate_from_a_rectangle(cv):
    """A rectangle over the top-left quadrant must select the CD3-/CD19+ cells."""
    b = cv.bins
    cv.viewer.layers["gates"].add_rectangles(
        np.array([[0, 0], [0, b / 2], [b / 2, b / 2], [b / 2, 0]])
    )
    cv.w_gate_name.value = "B cells"
    cv.w_gate_apply.clicked()

    mask = cv.adata.obs["B cells"].to_numpy(dtype=bool)
    assert 0 < mask.sum() < cv.adata.n_obs
    # top-left of the canvas is low x (CD3) and high y (CD19)
    import cytopy

    cd3 = cv.adata.layers["asinh"][:, cytopy.channel_index(cv.adata, "CD3")]
    cd19 = cv.adata.layers["asinh"][:, cytopy.channel_index(cv.adata, "CD19")]
    assert cd19[mask].mean() > cd19[~mask].mean()
    assert cd3[mask].mean() < cd3[~mask].mean()
    assert cv.adata.uns["cytopy"]["gates"]["B cells"]["x"] == "CD3 (FITC-A)"


def test_gate_hierarchy_intersects_with_parent(cv):
    b = cv.bins
    cv._robust = False  # so a full-canvas rectangle really does catch everything
    cv.refresh()
    gates = cv.viewer.layers["gates"]
    gates.add_rectangles(np.array([[0, 0], [0, b], [b, b], [b, 0]]))
    cv.w_gate_name.value = "all"
    cv.w_gate_apply.clicked()
    assert cv.adata.obs["all"].sum() == cv.adata.n_obs

    gates.data = []
    gates.add_rectangles(np.array([[0, 0], [0, b / 2], [b / 2, b / 2], [b / 2, 0]]))
    cv.w_parent.value = "all"
    cv.w_gate_name.value = "child"
    cv.w_gate_apply.clicked()
    child = cv.adata.obs["child"].to_numpy(dtype=bool)
    assert child.sum() > 0
    assert (child & ~cv.adata.obs["all"].to_numpy(dtype=bool)).sum() == 0
    assert cv.adata.uns["cytopy"]["gates"]["child"]["parent"] == "all"


def test_parent_gate_subsets_the_plotted_events(cv):
    b = cv.bins
    cv.viewer.layers["gates"].add_rectangles(
        np.array([[0, 0], [0, b / 2], [b / 2, b / 2], [b / 2, 0]])
    )
    cv.w_gate_name.value = "g"
    cv.w_gate_apply.clicked()
    total = cv.selection_mask().sum()
    cv.w_parent.value = "g"
    assert cv.selection_mask().sum() < total
    assert cv.selection_mask().sum() == cv.adata.obs["g"].sum()


def test_gate_without_shapes_is_a_no_op(cv):
    cv.w_gate_name.value = "empty"
    cv.w_gate_apply.clicked()
    assert "empty" not in cv.adata.obs
    assert "draw a shape" in cv.w_status.value


def test_the_axes_hold_everything_until_told_to_clip(cv):
    """Off by default: clipping drops events from the plot and from every gate."""
    full = cv.axes.x_hi - cv.axes.x_lo
    cv._robust = True
    cv.refresh()
    assert cv.axes.x_hi - cv.axes.x_lo < full


def test_the_clip_checkbox_drives_it(cv):
    full = cv.axes.x_hi - cv.axes.x_lo
    cv.w_clip.value = True
    assert cv._robust is True
    assert cv.axes.x_hi - cv.axes.x_lo < full
    cv.w_clip.value = False
    assert cv.axes.x_hi - cv.axes.x_lo == pytest.approx(full)


def test_channels_can_be_named_by_marker_or_detector(demo, make_napari_viewer):
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    cv = CytoViewer(demo, layer="asinh", x="CD3", y="PE-A", bins=64, viewer=make_napari_viewer())
    assert (cv.w_x.value, cv.w_y.value) == ("CD3 (FITC-A)", "CD19 (PE-A)")


# --------------------------------------------------------------------------
# canvas background
# --------------------------------------------------------------------------
def test_background_is_white_by_default(cv):
    from cytopy.viewer import DEFAULT_BACKGROUND

    assert DEFAULT_BACKGROUND == "white"
    assert cv.background == "white"
    assert np.allclose(cv.viewer.canvas.background_color_override, [1.0, 1.0, 1.0, 1.0])


def test_empty_bins_are_transparent(cv):
    """Otherwise the background is whatever the colormap maps zero counts to.

    turbo's zero is a dark blue-purple, and the density image covers the whole
    plot, so without this the canvas colour would never be visible.
    """
    colormap = cv.density.colormap
    assert colormap.map(np.array([0.0]))[0][3] == 0.0
    assert colormap.map(np.array([1.0]))[0][3] == 1.0
    # ... and the colours themselves are untouched.
    from napari.utils.colormaps import ensure_colormap

    assert np.allclose(
        colormap.map(np.array([0.5]))[0][:3], ensure_colormap("turbo").map(np.array([0.5]))[0][:3]
    )


def test_decorations_flip_with_the_background(demo, make_napari_viewer):
    from cytopy.viewer import CytoViewer

    cytopy = __import__("cytopy")
    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    cv = CytoViewer(demo, layer="asinh", bins=64, viewer=make_napari_viewer())

    def text_colour():
        # Per-point now, since gate labels get a colour of their own.
        return np.asarray(cv.labels.text.color.array)[0][:3]

    light_text = text_colour().copy()
    light_grid = np.asarray(cv.grid.edge_color[0])[:3].copy()

    cv.set_background("black")
    assert cv.background == "black"
    assert text_colour().mean() > light_text.mean()  # dark bg -> light text
    assert np.asarray(cv.grid.edge_color[0])[:3].mean() != light_grid.mean()


def test_background_can_be_changed_from_the_widget(cv):
    cv.w_background.value = "black"
    assert cv.background == "black"
    assert np.allclose(cv.viewer.canvas.background_color_override, [0.0, 0.0, 0.0, 1.0])
    assert "white" in list(cv.w_background.choices)


def test_a_custom_background_is_accepted_and_nonsense_is_not(demo, make_napari_viewer):
    from cytopy.viewer import CytoViewer

    __import__("cytopy").asinh_transform(demo, 150.0, layer="X", inplace=True)
    cv = CytoViewer(demo, layer="asinh", bins=64, background="#f5f5f5", viewer=make_napari_viewer())
    assert cv.background == "#f5f5f5"
    assert "#f5f5f5" in list(cv.w_background.choices)
    with pytest.raises(ValueError):
        cv.set_background("not a colour")
    assert cv.background == "#f5f5f5"  # unchanged after a bad one


def test_changing_the_colormap_keeps_the_fade(cv):
    cv.w_cmap.value = "viridis"
    assert cv.density.colormap.map(np.array([0.0]))[0][3] == 0.0
    assert "viridis" in cv.density.colormap.name


# --------------------------------------------------------------------------
# several samples in one window
# --------------------------------------------------------------------------
def test_as_one_anndata_leaves_a_single_object_alone(demo):
    from cytopy.viewer import as_one_anndata

    assert as_one_anndata(demo) is demo


def test_as_one_anndata_concatenates_a_list(demo, demo_path):
    import cytopy
    from cytopy.viewer import as_one_anndata

    second = cytopy.read_fcs(demo_path, sample_id="run2")
    out = as_one_anndata([demo, second])
    assert out.n_obs == demo.n_obs + second.n_obs
    assert sorted(out.obs["sample"].astype(str).unique()) == ["demo", "run2"]


def test_as_one_anndata_takes_names_from_a_mapping(demo, demo_path):
    import cytopy
    from cytopy.viewer import as_one_anndata

    second = cytopy.read_fcs(demo_path)
    out = as_one_anndata({"healthy": demo, "treated": second})
    assert sorted(out.obs["sample"].astype(str).unique()) == ["healthy", "treated"]

    out = as_one_anndata([demo, second], names=["a", "b"])
    assert sorted(out.obs["sample"].astype(str).unique()) == ["a", "b"]


def test_same_named_files_do_not_merge(demo, demo_path):
    """Two files both called 'demo' must stay two entries in the selector."""
    import cytopy
    from cytopy.viewer import as_one_anndata

    out = as_one_anndata([demo, cytopy.read_fcs(demo_path)])
    assert sorted(out.obs["sample"].astype(str).unique()) == ["demo", "demo.1"]
    assert out.n_obs == 2 * demo.n_obs


def test_as_one_anndata_reads_paths(demo_path, tmp_path):
    import cytopy
    from cytopy.viewer import as_one_anndata

    assert as_one_anndata(demo_path).n_obs == 60_000
    assert as_one_anndata([demo_path, demo_path]).n_obs == 120_000

    h5 = tmp_path / "demo.h5ad"
    cytopy.read_fcs(demo_path).write_h5ad(h5)
    assert as_one_anndata(h5).n_obs == 60_000

    import shutil

    shutil.copy(demo_path, tmp_path / "a.fcs")
    shutil.copy(demo_path, tmp_path / "b.fcs")
    assert as_one_anndata(tmp_path).n_obs == 120_000  # a directory of FCS


def test_as_one_anndata_rejects_nonsense(demo):
    from cytopy.viewer import as_one_anndata

    with pytest.raises(ValueError, match="nothing to view"):
        as_one_anndata([])
    with pytest.raises(ValueError, match="2 objects but 1 names"):
        as_one_anndata([demo, demo], names=["only one"])
    with pytest.raises(TypeError, match="expected an AnnData or a path"):
        as_one_anndata([demo, 42])


def test_viewer_opens_on_several_samples(demo, demo_path, make_napari_viewer):
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    second = cytopy.read_fcs(demo_path, sample_id="run2")
    cytopy.asinh_transform(second, 150.0, layer="X", inplace=True)

    cv = CytoViewer(
        [demo, second],
        layer="asinh",
        x="CD3",
        y="CD19",
        bins=64,
        viewer=make_napari_viewer(),
    )
    assert cv.adata.n_obs == 120_000
    assert list(cv.w_samples.choices) == ["demo", "run2"]
    assert cv.w_lock.value is True  # locked once there is more than one

    cv.w_samples.value = ["run2"]
    assert cv.panel.samples == ("run2",)
    assert int(cv.selection_mask().sum()) == 60_000
    assert int(cv.selection_mask(sample=False).sum()) == 120_000


def test_axes_stay_put_between_samples(demo, demo_path, make_napari_viewer):
    """Rescaling per sample is what makes two samples look alike when they are not."""
    import cytopy
    from cytopy.viewer import CytoViewer

    dim = cytopy.read_fcs(demo_path, sample_id="dim")
    dim.X = dim.X * np.float32(0.25)
    cv = CytoViewer([demo, dim], x="CD3", y="CD19", bins=64, viewer=make_napari_viewer())

    cv.w_samples.value = ["demo"]
    locked = (cv.axes.x_lo, cv.axes.x_hi)
    cv.w_samples.value = ["dim"]
    assert (cv.axes.x_lo, cv.axes.x_hi) == locked  # same axes, so they compare

    cv.w_lock.value = False
    assert cv.axes.x_hi < locked[1]  # unlocked, the dim sample fills the plot


def test_a_single_sample_is_not_locked_by_default(cv):
    assert list(cv.w_samples.choices) == ["demo"]
    assert cv.w_lock.value is False


# --------------------------------------------------------------------------
# gate hierarchies
# --------------------------------------------------------------------------
def _rect(cv, x_lo, x_hi, y_lo, y_hi):
    """A rectangle given in data coordinates, as canvas coordinates."""
    return cv.to_canvas(np.array([x_lo, x_lo, x_hi, x_hi]), np.array([y_lo, y_hi, y_hi, y_lo]))


def _truth(adata, layer, x, y, box):
    import cytopy

    values = np.asarray(adata.layers[layer])
    xi, yi = cytopy.channel_index(adata, x), cytopy.channel_index(adata, y)
    return (
        (values[:, xi] > box[0])
        & (values[:, xi] < box[1])
        & (values[:, yi] > box[2])
        & (values[:, yi] < box[3])
    )


def test_a_child_gate_counts_only_what_was_drawn_for_it(cv):
    """A leftover shape used to be OR-ed into the next gate, inflating it."""
    parent_box = (2.0, 9.0, -2.0, 9.0)
    cv.gates.add_polygons([_rect(cv, *parent_box)])
    cv.apply_gate("parent")
    assert len(cv.gates.data) == 0  # committed: it moves to the applied layer
    assert len(cv.applied.data) == 1

    child_box = (3.0, 5.0, 0.0, 2.0)
    cv.w_parent.value = "parent"
    cv.gates.add_polygons([_rect(cv, *child_box)])
    cv.apply_gate("child", parent="parent")

    expected = _truth(cv.adata, "asinh", "CD3", "CD19", parent_box) & _truth(
        cv.adata, "asinh", "CD3", "CD19", child_box
    )
    assert int(cv.adata.obs["child"].sum()) == int(expected.sum())
    assert np.array_equal(cv.adata.obs["child"].to_numpy(), expected)
    # and the child is a strict subset of its parent
    assert not (cv.adata.obs["child"] & ~cv.adata.obs["parent"]).any()


def test_two_shapes_still_make_one_gate(cv):
    """Clearing on apply must not break a gate deliberately drawn in two parts."""
    left, right = (2.0, 3.0, -2.0, 9.0), (5.0, 6.0, -2.0, 9.0)
    cv.gates.add_polygons([_rect(cv, *left)])
    cv.gates.add_polygons([_rect(cv, *right)])
    cv.apply_gate("both")
    expected = _truth(cv.adata, "asinh", "CD3", "CD19", left) | _truth(
        cv.adata, "asinh", "CD3", "CD19", right
    )
    assert int(cv.adata.obs["both"].sum()) == int(expected.sum())


@pytest.mark.parametrize(
    ("change", "value"),
    [("w_bins", 128), ("w_parent", "seed"), ("w_y", "CD8 (APC-A)")],
)
def test_a_shape_keeps_its_data_meaning_when_the_axes_move(cv, change, value):
    """The shape lives in pixels but means data; a rescale must not shift it."""
    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("seed")

    cv.gates.add_polygons([_rect(cv, 3.0, 5.0, 0.0, 2.0)])
    before = cv.axes.to_display(np.asarray(cv.gates.data[0], dtype=float))
    getattr(cv, change).value = value
    after = cv.axes.to_display(np.asarray(cv.gates.data[0], dtype=float))
    assert np.abs(after - before).max() < 1e-6


def test_shape_types_survive_being_reprojected(cv):
    cv.gates.add_rectangles([_rect(cv, 2.0, 4.0, 0.0, 2.0)])
    cv.gates.add_ellipses([_rect(cv, 5.0, 6.0, 1.0, 3.0)])
    kinds = [str(k) for k in cv.gates.shape_type]
    cv.w_bins.value = 128
    assert [str(k) for k in cv.gates.shape_type] == kinds
    assert len(cv.gates.data) == 2


def test_events_off_the_axes_are_reported_not_hidden(cv):
    """A narrowed axis draws some events nowhere, and the status line says so."""
    assert cv.off_axis_count() == 0  # the default range holds everything

    lo, hi = cv.axes.x_lo, cv.axes.x_hi
    narrow = cv.x_scale.to_raw(np.array([lo + 0.4 * (hi - lo), hi]))
    cv.set_limits(x=(float(narrow[0]), float(narrow[1])))
    off = cv.off_axis_count()
    assert off > 0

    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("g")
    assert f"{off:,} outside the axes" in cv.w_status.value

    cv.w_autoscale.changed.emit(None)
    assert cv.off_axis_count() == 0  # back to holding everything


def test_apply_gate_needs_a_shape(cv):
    with pytest.raises(ValueError, match="draw a shape"):
        cv.apply_gate("nothing")
    assert "nothing" not in cv.adata.obs


@pytest.fixture
def pair(demo, demo_path, make_napari_viewer):
    """Two samples in one viewer, the second a quarter as bright."""
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    dim = cytopy.read_fcs(demo_path, sample_id="dim")
    dim.X = dim.X * np.float32(0.4)
    cytopy.asinh_transform(dim, 150.0, layer="X", inplace=True)
    return CytoViewer(
        [demo, dim], layer="asinh", x="CD3", y="CD19", bins=128, viewer=make_napari_viewer()
    )


# --------------------------------------------------------------------------
# the colour bar
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# the 1-D histogram
# --------------------------------------------------------------------------
def test_histogram_swaps_the_density_out(pair):
    pair.w_plot.value = "histogram"
    assert pair.histogram is True
    assert pair.curves.visible is True
    assert pair.density.visible is False

    pair.w_plot.value = "density"
    assert pair.density.visible is True
    assert pair.curves.visible is False


def test_one_curve_per_sample(pair):
    pair.w_plot.value = "histogram"
    groups = pair.histogram_groups()
    assert [name for name, _ in groups] == ["demo", "dim"]
    assert all(int(mask.sum()) == 60_000 for _, mask in groups)
    # a fill and a line per curve, plus a legend rule for each
    assert len(pair.curves.data) == 6
    labels = [str(t) for t in pair.labels.text.string.array]
    assert "demo (60,000)" in labels and "dim (60,000)" in labels
    assert "2 curve(s)" in pair.w_status.value


def test_the_curves_follow_the_data(pair):
    """The dim sample is 0.4x, so its distribution sits to the left."""
    pair.w_plot.value = "histogram"

    def peak_x(index):
        path = np.asarray(pair.curves.data[index])
        column = path[np.argmin(path[:, 0]), 1]  # row 0 is the top of the plot
        return float(pair.axes.to_display(np.array([[0.0, column]]))[0, 0])

    assert peak_x(1) < peak_x(0)


def test_the_sample_list_narrows_to_one_curve(pair):
    pair.w_plot.value = "histogram"
    pair.w_samples.value = ["dim"]
    assert [name for name, _ in pair.histogram_groups()] == ["dim"]
    assert len(pair.curves.data) == 3  # one fill, one line, one legend rule


def test_the_parent_gate_applies_to_the_curves(cv):
    cv.gates.add_polygons([_rect(cv, 3.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("bright")
    n = int(cv.adata.obs["bright"].sum())

    cv.w_plot.value = "histogram"
    cv.w_parent.value = "bright"
    assert int(cv.histogram_groups()[0][1].sum()) == n


def test_the_y_axis_is_a_density_not_a_channel(pair):
    pair.w_plot.value = "histogram"
    labels = _labels(pair)
    assert "% of mode" in labels
    assert "CD19 (PE-A)" not in labels  # the y channel means nothing here
    assert "CD3 (FITC-A)" in labels  # ... but the x channel still does

    assert {"0", "50", "100"} <= set(_labels(pair))


def test_every_curve_peaks_at_a_hundred(three):
    """Per cent of mode, the flow convention: a rare population is still legible."""
    from cytopy.viewer import _MODE_TOP

    three.w_samples.value = ["demo", "c"]
    groups = three.histogram_groups()
    floor = three.bins - 1
    scale = (three.bins - 1) / (_MODE_TOP * 1.05)

    for index in range(len(groups)):
        line = np.asarray(three.curves.data[len(groups) + index])  # fills come first
        peak = (floor - line[:, 0].min()) / scale
        assert peak == pytest.approx(_MODE_TOP, abs=0.01)

    labels = _labels(three)
    assert "% of mode" in labels
    for tick in ("0", "25", "50", "75", "100"):
        assert tick in labels


def test_a_shape_gates_an_interval_on_a_histogram(cv):
    """There is no second channel to be inside of, so a box means an x range."""
    import cytopy

    cv.w_plot.value = "histogram"
    cv.gates.add_rectangles([_rect(cv, 3.0, 5.0, 0.0, 1.0)])
    mask = cv.current_gate_mask()

    values = np.asarray(cv.adata.layers["asinh"])[:, cytopy.channel_index(cv.adata, "CD3")]
    expected = (values >= 3.0) & (values <= 5.0)
    assert int(mask.sum()) == int(expected.sum())
    assert np.array_equal(mask, expected)


def test_settings_that_do_not_apply_are_greyed_out(pair):
    pair.w_plot.value = "histogram"
    assert pair.w_y.enabled is False

    pair.w_plot.value = "density"
    assert pair.w_y.enabled is True


def test_an_empty_selection_draws_nothing(cv):
    cv.gates.add_polygons([_rect(cv, 100.0, 101.0, 100.0, 101.0)])
    cv.apply_gate("empty")
    cv.w_plot.value = "histogram"
    cv.w_parent.value = "empty"
    assert cv.histogram_groups() == []
    assert len(cv.curves.data) == 0


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# revisiting a gate
# --------------------------------------------------------------------------
def _make_hierarchy(cv):
    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("parent")
    cv.w_parent.value = "parent"
    cv.gates.add_polygons([_rect(cv, 3.0, 6.0, 0.0, 4.0)])
    cv.apply_gate("child", parent="parent")
    return int(cv.adata.obs["parent"].sum()), int(cv.adata.obs["child"].sum())


def test_a_stored_gate_recomputes_to_exactly_what_was_drawn(cv):
    """The canvas maps data to pixels affinely, so the inside of a shape is the same."""
    import cytopy

    _make_hierarchy(cv)
    for name in ("parent", "child"):
        assert np.array_equal(cytopy.gate_mask(cv.adata, name), cv.adata.obs[name].to_numpy())


def test_a_gate_can_be_put_back_on_the_canvas(cv):
    _make_hierarchy(cv)
    cv.w_x.value = "CD8 (APC-A)"  # wander off somewhere else first
    cv.gates.data = []

    cv.load_gate("child")
    assert len(cv.gates.data) == 1
    assert cv.w_gate_name.value == "child"
    assert cv.w_x.value == "CD3 (FITC-A)"  # back in the plane it was drawn in
    assert cv.w_y.value == "CD19 (PE-A)"
    assert cv.w_parent.value == "parent"  # its own parent, not itself


def test_reapplying_an_untouched_gate_changes_nothing(cv):
    before = _make_hierarchy(cv)
    cv.load_gate("parent")
    cv.apply_gate("parent")
    after = (int(cv.adata.obs["parent"].sum()), int(cv.adata.obs["child"].sum()))
    assert after == before


def test_adjusting_a_gate_brings_its_children_along(cv):
    """A child was worked out against the old outline; it is stale until recomputed."""
    import cytopy

    _, child_before = _make_hierarchy(cv)
    cv.load_gate("parent")
    cv.gates.data = []
    cv.gates.add_polygons([_rect(cv, 4.0, 9.0, -2.0, 9.0)])  # smaller
    cv.apply_gate("parent")

    assert int(cv.adata.obs["child"].sum()) < child_before
    assert np.array_equal(cytopy.gate_mask(cv.adata, "child"), cv.adata.obs["child"].to_numpy())
    assert not (cv.adata.obs["child"] & ~cv.adata.obs["parent"]).any()
    assert "recomputed child" in cv.w_status.value


def test_a_grandchild_is_recomputed_too(cv):
    import cytopy

    _make_hierarchy(cv)
    cv.w_parent.value = "child"
    cv.gates.add_polygons([_rect(cv, 3.5, 5.5, 0.5, 3.5)])
    cv.apply_gate("grandchild", parent="child")

    cv.load_gate("parent")
    cv.gates.data = []
    cv.gates.add_polygons([_rect(cv, 4.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("parent")

    assert cytopy.gate_children(cv.adata, "child") == ["grandchild"]
    assert np.array_equal(
        cytopy.gate_mask(cv.adata, "grandchild"), cv.adata.obs["grandchild"].to_numpy()
    )
    assert not (cv.adata.obs["grandchild"] & ~cv.adata.obs["parent"]).any()


def test_deleting_a_gate_takes_its_children_with_it(cv):
    _make_hierarchy(cv)
    assert cv.delete_gate("parent") == ["parent", "child"]
    assert "parent" not in cv.adata.obs and "child" not in cv.adata.obs
    assert "parent" not in cv.adata.uns["cytopy"]["gates"]
    assert list(cv.w_gate_pick.choices) == ["<none>"]


def test_deleting_a_gate_a_panel_was_using_frees_the_panel(cv):
    _make_hierarchy(cv)
    cv.w_parent.value = "parent"
    cv.delete_gate("parent")
    assert cv.panel.parent == "<none>"


def test_a_gate_with_no_outline_cannot_be_adjusted(cv):
    import cytopy

    cytopy.add_gate(cv.adata, "from a mask", np.ones(cv.adata.n_obs, dtype=bool))
    with pytest.raises(KeyError, match="no outline"):
        cv.load_gate("from a mask")
    with pytest.raises(KeyError, match="no gate"):
        cv.load_gate("never drawn")


def test_a_gate_cannot_be_its_own_parent(cv):
    _make_hierarchy(cv)
    cv.load_gate("child")
    cv._updating = True
    cv.w_parent.value = "child"
    cv._updating = False
    cv.gates.add_polygons([_rect(cv, 3.0, 6.0, 0.0, 4.0)])
    cv._apply_gate()
    assert "own parent" in cv.w_status.value


# --------------------------------------------------------------------------
# applied gates stay on screen
# --------------------------------------------------------------------------
def test_an_applied_gate_stays_visible_and_labelled(cv):
    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("lymphs")

    assert len(cv.gates.data) == 0  # the editable layer is clear ...
    assert len(cv.applied.data) == 1  # ... and the gate is still on screen
    assert cv.applied.visible is True
    n = int(cv.adata.obs["lymphs"].sum())
    # The label goes on the text layer, which is the one that draws.
    assert f"lymphs  {100 * n / cv.adata.n_obs:.1f}%" in [
        str(t) for t in cv.labels.text.string.array
    ]


def test_the_label_is_a_share_of_its_own_parent(cv):
    """Not of whatever the panel shows, or the same gate would read differently."""
    _make_hierarchy(cv)
    parent_n = int(cv.adata.obs["parent"].sum())
    child_n = int(cv.adata.obs["child"].sum())
    labels = [str(t) for t in cv.labels.text.string.array if "%" in str(t) and "  " in str(t)]
    assert f"parent  {100 * parent_n / cv.adata.n_obs:.1f}%" in labels
    assert f"child  {100 * child_n / parent_n:.1f}%" in labels

    # switching the panel's parent gate must not move the numbers
    cv.w_parent.value = "parent"
    after = [str(t) for t in cv.labels.text.string.array if "%" in str(t) and "  " in str(t)]
    assert after == labels


def test_applied_gates_do_not_join_the_next_one(cv):
    """They are on their own layer, so the union bug cannot come back."""
    first = (2.0, 3.0, -2.0, 9.0)
    second = (6.0, 7.0, -2.0, 9.0)
    cv.gates.add_polygons([_rect(cv, *first)])
    cv.apply_gate("one")
    cv.gates.add_polygons([_rect(cv, *second)])
    cv.apply_gate("two")

    expected = _truth(cv.adata, "asinh", "CD3", "CD19", second)
    assert int(cv.adata.obs["two"].sum()) == int(expected.sum())
    assert len(cv.applied.data) == 2


def test_gates_belong_to_the_plane_they_were_drawn_in(cv):
    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("here")
    assert cv.applied.visible is True
    cv.w_y.value = "CD8 (APC-A)"
    assert len(cv.applied.data) == 0  # a different plane, so nothing to outline
    cv.w_y.value = "CD19 (PE-A)"
    assert len(cv.applied.data) == 1


# --------------------------------------------------------------------------
# widget choices survive napari resetting them
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# gating on a panel that is not at the origin
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# gate labels
# --------------------------------------------------------------------------
def test_gate_labels_are_drawn_on_the_text_layer(cv):
    """On the shapes layer they did not show; the text layer is the one that draws."""
    _make_hierarchy(cv)
    labels = [str(t) for t in cv.labels.text.string.array]
    parent_n = int(cv.adata.obs["parent"].sum())
    child_n = int(cv.adata.obs["child"].sum())
    assert f"parent  {100 * parent_n / cv.adata.n_obs:.1f}%" in labels
    assert f"child  {100 * child_n / parent_n:.1f}%" in labels


def test_gate_labels_stand_out_from_the_background(cv):
    """A pale blue label is invisible on white, which is why this follows it."""
    _make_hierarchy(cv)
    colours = np.asarray(cv.labels.text.color.array)
    assert len(np.unique(colours, axis=0)) == 2  # axis text and gate text differ

    on_white = colours[-1].copy()
    cv.set_background("black")
    on_black = np.asarray(cv.labels.text.color.array)[-1]
    assert on_white[:3].mean() < 0.5 < on_black[:3].mean()
    assert np.asarray(cv.applied.edge_color)[0][:3].mean() > 0.5


def test_a_label_sits_by_its_own_outline(cv):
    cv.gates.add_polygons([_rect(cv, 3.0, 6.0, 0.0, 4.0)])
    cv.apply_gate("g")
    (row, col, label) = cv._gate_labels[0]
    corners = np.asarray(cv.applied.data[0])
    assert label.startswith("g  ")
    assert abs(col - corners[:, 1].min()) < 1.0
    assert row < corners[:, 0].min()  # just above it


def test_the_y_label_reads_up_the_side(cv):
    from cytopy.viewer import Y_LABEL_ROTATION

    assert Y_LABEL_ROTATION == 270
    assert cv.ylabel.text.rotation == Y_LABEL_ROTATION
    assert list(cv.ylabel.text.string.array) == ["CD19 (PE-A)"]

    # it follows the plot, and stays rotated
    cv.w_y.value = "CD8 (APC-A)"
    assert list(cv.ylabel.text.string.array) == ["CD8 (APC-A)"]
    cv.w_plot.value = "histogram"
    assert list(cv.ylabel.text.string.array) == ["% of mode"]
    assert cv.ylabel.text.rotation == Y_LABEL_ROTATION


# --------------------------------------------------------------------------
# the widgets are a view of the plot, not the state itself
# --------------------------------------------------------------------------
def test_swap_actually_swaps_the_plot(cv):
    """Setting a widget with its callback suppressed used to move the view only."""
    before = np.asarray(cv.density.data).copy()
    cv.w_swap.clicked()

    assert (cv.panel.x, cv.panel.y) == ("CD19 (PE-A)", "CD3 (FITC-A)")
    assert (cv.w_x.value, cv.w_y.value) == (cv.panel.x, cv.panel.y)
    assert not np.array_equal(np.asarray(cv.density.data), before)  # the plot redrew

    labels = _labels(cv)
    assert "CD19 (PE-A)" in labels and "CD3 (FITC-A)" in labels
    assert list(cv.ylabel.text.string.array) == ["CD3 (FITC-A)"]
    cv.w_swap.clicked()
    assert (cv.panel.x, cv.panel.y) == ("CD3 (FITC-A)", "CD19 (PE-A)")


def test_set_plot_moves_state_and_widgets_together(cv):
    cv.set_plot(x="CD8 (APC-A)", kind="histogram")
    assert (cv.panel.x, cv.panel.kind) == ("CD8 (APC-A)", "histogram")
    assert (cv.w_x.value, cv.w_plot.value) == ("CD8 (APC-A)", "histogram")
    assert cv.curves.visible is True

    with pytest.raises(AttributeError, match="not a plot setting"):
        cv.set_plot(bins=64)


def test_a_gate_after_a_swap_uses_the_new_axes(cv):
    """The bug would have gated against the channels the plot no longer showed."""
    cv.w_swap.clicked()
    cv.gates.add_polygons([_rect(cv, 0.0, 4.0, 3.0, 9.0)])
    cv.apply_gate("swapped")

    # x is now CD19 and y is CD3
    expected = _truth(cv.adata, "asinh", "CD19", "CD3", (0.0, 4.0, 3.0, 9.0))
    assert int(cv.adata.obs["swapped"].sum()) == int(expected.sum())
    assert cv.adata.uns["cytopy"]["gates"]["swapped"]["x"] == "CD19 (PE-A)"


# --------------------------------------------------------------------------
# choosing which samples a histogram draws
# --------------------------------------------------------------------------
@pytest.fixture
def three(demo, demo_path, make_napari_viewer):
    """Three samples of decreasing brightness, in one viewer."""
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    parts = [demo]
    for name, factor in (("b", 0.6), ("c", 0.3)):
        other = cytopy.read_fcs(demo_path, sample_id=name)
        other.X = other.X * np.float32(factor)
        cytopy.asinh_transform(other, 150.0, layer="X", inplace=True)
        parts.append(other)
    cv = CytoViewer(parts, layer="asinh", x="CD3", y="CD19", bins=128, viewer=make_napari_viewer())
    cv.set_plot(kind="histogram")
    return cv


def test_picking_no_samples_draws_them_all(three):
    """So the plot is never accidentally empty."""
    assert list(three.w_samples.choices) == ["demo", "b", "c"]
    assert three.panel.samples == ()
    assert [name for name, _ in three.histogram_groups()] == ["demo", "b", "c"]


def test_samples_can_be_picked(three):
    three.w_samples.value = ["demo", "c"]
    assert three.panel.samples == ("demo", "c")
    assert [name for name, _ in three.histogram_groups()] == ["demo", "c"]
    assert [t for _, _, t in three.panel.legend] == ["demo (60,000)", "c (60,000)"]
    assert three.panel.title.startswith("demo, c")

    three.set_plot(samples=["b"])
    assert [name for name, _ in three.histogram_groups()] == ["b"]
    assert three.w_samples.value == ["b"]


def test_one_list_serves_both_plots(three):
    """Selecting one sample is how you show one; there is no separate field."""
    assert not hasattr(three, "w_sample")
    assert three.w_samples.enabled is True

    three.set_plot(kind="density", samples=["b"])
    assert int(three.selection_mask().sum()) == 60_000
    assert three.panel.title.startswith("b —")

    # a density pools whatever is picked ...
    three.set_plot(samples=["b", "c"])
    assert int(three.selection_mask().sum()) == 120_000
    # ... and a histogram draws one curve for each
    three.set_plot(kind="histogram")
    assert [name for name, _ in three.histogram_groups()] == ["b", "c"]
    assert three.w_samples.enabled is True


def test_a_parent_gate_still_narrows_the_curves(three):
    three.set_plot(kind="density")
    three.gates.add_polygons([_rect(three, 3.0, 9.0, -2.0, 9.0)])
    three.apply_gate("bright")
    three.set_plot(kind="histogram", parent="bright")

    for _, mask in three.histogram_groups():
        assert not (mask & ~three.adata.obs["bright"].to_numpy()).any()


# --------------------------------------------------------------------------
# the area under each curve
# --------------------------------------------------------------------------
def test_each_curve_is_filled_under_its_line(three):
    from cytopy.viewer import CURVE_FILL_ALPHA

    three.w_samples.value = ["demo", "b"]
    kinds = [str(k) for k in three.curves.shape_type]
    # two fills, two lines, two legend rules
    assert kinds == ["polygon", "polygon", "path", "path", "path", "path"]

    faces = np.asarray(three.curves.face_color)
    expected = int(CURVE_FILL_ALPHA, 16) / 255
    assert faces[0][3] == pytest.approx(expected, abs=0.01)
    assert 0 < faces[0][3] < 0.5  # translucent, so overlaps stay readable
    assert faces[2][3] == 0.0  # the lines themselves are not filled

    # a fill is its curve closed down to the baseline
    fill = np.asarray(three.curves.data[0])
    line = np.asarray(three.curves.data[2])
    assert len(fill) == len(line) + 2
    assert np.allclose(fill[: len(line)], line)
    assert fill[-1][0] == fill[-2][0] == pytest.approx(three.bins)  # just below the floor


def test_fill_and_line_share_a_colour(three):
    three.w_samples.value = ["demo", "b"]
    faces = np.asarray(three.curves.face_color)
    edges = np.asarray(three.curves.edge_color)
    for i in range(2):
        assert np.allclose(faces[i][:3], edges[i + 2][:3])
    assert not np.allclose(edges[2][:3], edges[3][:3])  # ... but differ between samples


# --------------------------------------------------------------------------
# gating across several planes, and coming back to it
# --------------------------------------------------------------------------
def test_gates_on_different_channel_pairs_in_one_session(cv):
    """A hierarchy is several plots, not one."""
    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, -2.0, 9.0)])
    cv.apply_gate("cells")

    cv.set_plot(x="CD8 (APC-A)", y="FSC-A", parent="cells")
    cv.gates.add_polygons([_rect(cv, 2.0, 9.0, 0.0, 9e4)])
    cv.apply_gate("cd8", parent="cells")

    gates = cv.adata.uns["cytopy"]["gates"]
    assert (gates["cells"]["x"], gates["cells"]["y"]) == ("CD3 (FITC-A)", "CD19 (PE-A)")
    assert (gates["cd8"]["x"], gates["cd8"]["y"]) == ("CD8 (APC-A)", "FSC-A")
    assert not (cv.adata.obs["cd8"] & ~cv.adata.obs["cells"]).any()


def test_gates_are_loaded_when_the_data_is_opened_again(demo, make_napari_viewer):
    """Gating is something you come back to, not do in one sitting."""
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    first = CytoViewer(demo, layer="asinh", x="CD3", y="CD19", bins=64, viewer=make_napari_viewer())
    first.gates.add_polygons([_rect(first, 2.0, 9.0, -2.0, 9.0)])
    first.apply_gate("cells")
    n = int(demo.obs["cells"].sum())

    again = CytoViewer(demo, layer="asinh", x="CD3", y="CD19", bins=64, viewer=make_napari_viewer())
    assert len(again.applied.data) == 1  # outlined where it was drawn
    assert "cells" in list(again.w_gate_pick.choices)
    assert "cells" in list(again.w_parent.choices)

    again.load_gate("cells")
    again.apply_gate("cells")
    assert int(demo.obs["cells"].sum()) == n  # unchanged by the round trip


@pytest.fixture
def on_fixture_window(make_napari_viewer, monkeypatch):
    """Let ``open_napari`` build on the test window instead of its own.

    It normally calls ``napari.Viewer()`` itself, which re-registers the
    plugin actions and trips over napari's own fixture. Everything else about
    the function runs for real.
    """
    import napari

    import cytopy.viewer as viewer_module

    window = make_napari_viewer()
    monkeypatch.setattr(napari, "Viewer", lambda **kwargs: window)
    yield window
    viewer_module._CURRENT = None


def test_open_napari_returns_the_data_not_the_window(demo, on_fixture_window, capsys):
    import cytopy

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    cytopy.add_gate(demo, "earlier", np.ones(demo.n_obs, dtype=bool))

    out = cytopy.open_napari(demo, "asinh", block=False)
    assert out is demo  # the data, not the viewer
    assert cytopy.current_viewer() is not None
    assert cytopy.current_viewer().adata is demo
    assert capsys.readouterr().out == ""  # silent unless asked


def test_open_napari_reports_gates_when_verbose(demo, on_fixture_window, capsys):
    import cytopy

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    cytopy.add_gate(demo, "earlier", np.ones(demo.n_obs, dtype=bool))
    cytopy.open_napari(demo, "asinh", block=False, verbose=True)
    printed = capsys.readouterr().out
    assert "loaded 1 gate(s): earlier" in printed
    assert "gated: nothing new" in printed


def test_open_napari_takes_names_for_several_objects(demo, on_fixture_window):
    """The names bug: gate() used to resolve them twice and raise."""
    import cytopy

    other = demo.copy()
    out = cytopy.open_napari([demo, other], "raw", names=["A", "B"], block=False)
    assert sorted(out.obs["sample"].astype(str).unique()) == ["A", "B"]
    assert out.n_obs == demo.n_obs * 2


# --------------------------------------------------------------------------
# gating several files in one window
# --------------------------------------------------------------------------
@pytest.fixture
def pooled(demo, demo_path, make_napari_viewer):
    """Two files in one viewer, as a dict of samples."""
    import cytopy
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    other = cytopy.read_fcs(demo_path, sample_id="b")
    cytopy.asinh_transform(other, 150.0, layer="X", inplace=True)
    return CytoViewer(
        {"a": demo, "b": other},
        layer="asinh",
        x="CD3",
        y="CD19",
        bins=128,
        viewer=make_napari_viewer(),
    )


def test_a_gate_on_all_samples_covers_them_all(pooled):
    pooled.gates.add_polygons([_rect(pooled, 2.0, 9.0, -2.0, 9.0)])
    pooled.apply_gate("singlets")
    labels = pooled.adata.obs["sample"].astype(str).to_numpy()
    for name in ("a", "b"):
        assert int(pooled.adata.obs["singlets"][labels == name].sum()) > 0


def test_gating_one_sample_leaves_the_others_alone(pooled):
    """The shared gates once, the per-file ones one at a time, same window."""
    labels = pooled.adata.obs["sample"].astype(str).to_numpy()

    pooled.set_plot(samples=["a"])
    pooled.gates.add_polygons([_rect(pooled, 3.0, 9.0, 0.0, 4.0)])
    pooled.apply_gate("positive")
    first = int(pooled.adata.obs["positive"][labels == "a"].sum())
    assert first > 0
    assert int(pooled.adata.obs["positive"][labels == "b"].sum()) == 0

    pooled.set_plot(samples=["b"])
    pooled.gates.add_polygons([_rect(pooled, 3.0, 9.0, 0.0, 4.0)])
    pooled.apply_gate("positive")
    # b is gated now, and a still is -- the second apply used to wipe the first
    assert int(pooled.adata.obs["positive"][labels == "a"].sum()) == first
    assert int(pooled.adata.obs["positive"][labels == "b"].sum()) > 0


def test_regating_a_sample_can_still_turn_events_off(pooled):
    pooled.set_plot(samples=["a"])
    pooled.gates.add_polygons([_rect(pooled, 2.0, 9.0, -2.0, 9.0)])
    pooled.apply_gate("p")
    wide = int(pooled.adata.obs["p"].sum())

    pooled.gates.add_polygons([_rect(pooled, 6.0, 9.0, -2.0, 9.0)])
    pooled.apply_gate("p")
    assert int(pooled.adata.obs["p"].sum()) < wide  # narrowing really narrows


def test_splitting_back_out_keeps_the_gates(pooled):
    import cytopy

    pooled.gates.add_polygons([_rect(pooled, 2.0, 9.0, -2.0, 9.0)])
    pooled.apply_gate("singlets")
    for name in ("a", "b"):
        pooled.set_plot(samples=[name], parent="singlets")
        pooled.gates.add_polygons([_rect(pooled, 3.0, 9.0, 0.0, 4.0)])
        pooled.apply_gate("positive", parent="singlets")

    out = cytopy.split_samples(pooled.adata)
    assert sorted(out) == ["a", "b"]
    for name, part in out.items():
        assert part.n_obs == 60_000
        assert int(part.obs["positive"].sum()) > 0
        assert not (part.obs["positive"] & ~part.obs["singlets"]).any()
        assert sorted(part.uns["cytopy"]["gates"]) == ["positive", "singlets"]


def test_open_napari_can_start_on_chosen_channels(demo, on_fixture_window):
    """x/y are a starting point; the window owns them afterwards."""
    import cytopy

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    cytopy.open_napari(demo, "asinh", x="CD3", y="CD19", block=False)

    cv = cytopy.current_viewer()
    assert (cv.panel.x, cv.panel.y) == ("CD3 (FITC-A)", "CD19 (PE-A)")
    assert (cv.w_x.value, cv.w_y.value) == ("CD3 (FITC-A)", "CD19 (PE-A)")


def test_open_napari_can_start_on_one_sample(demo, on_fixture_window):
    import cytopy

    other = demo.copy()
    cytopy.open_napari([demo, other], "raw", names=["A", "B"], samples=["B"], block=False)

    cv = cytopy.current_viewer()
    assert cv.panel.samples == ("B",)
    assert list(cv.w_samples.value) == ["B"]
    # only that sample's events are in scope
    labels = cv.adata.obs["sample"].astype(str).to_numpy()
    assert int(cv.selection_mask().sum()) == int((labels == "B").sum())


def test_open_napari_defaults_show_everything(demo, on_fixture_window):
    import cytopy

    other = demo.copy()
    cytopy.open_napari([demo, other], "raw", names=["A", "B"], block=False)

    cv = cytopy.current_viewer()
    assert cv.panel.samples == ()
    assert int(cv.selection_mask().sum()) == cv.adata.n_obs


# --------------------------------------------------------------------------
# axis range fields
# --------------------------------------------------------------------------
def test_each_end_of_each_axis_has_its_own_slider(cv):
    """Four sliders, four values: two handles on one track cannot be separated."""
    sliders = (cv.w_xmin, cv.w_xmax, cv.w_ymin, cv.w_ymax)
    assert len({id(w) for w in sliders}) == 4
    assert [w.label.split()[:2] for w in sliders] == [
        ["x", "min"],
        ["x", "max"],
        ["y", "min"],
        ["y", "max"],
    ]


def test_the_sliders_read_back_the_range_on_screen(cv):
    """They are a readout as well as an input, in display coordinates."""
    axes = cv.axes
    assert float(cv.w_xmin.value) == pytest.approx(axes.x_lo, rel=1e-6)
    assert float(cv.w_xmax.value) == pytest.approx(axes.x_hi, rel=1e-6)
    assert float(cv.w_ymin.value) == pytest.approx(axes.y_lo, rel=1e-6)
    assert float(cv.w_ymax.value) == pytest.approx(axes.y_hi, rel=1e-6)


def test_each_slider_label_carries_its_own_raw_value(cv):
    """The ticks are labelled in raw units, so the sliders say them too."""
    raw_lo, raw_hi = cv.x_scale.to_raw(np.array([cv.axes.x_lo, cv.axes.x_hi]))
    assert cv.w_xmin.label.startswith("x min")
    assert cv.w_xmax.label.startswith("x max")
    assert f"{raw_hi:,.0f}" in cv.w_xmax.label or f"{raw_hi:.3g}" in cv.w_xmax.label
    label = cv.w_xmin.label
    assert str(int(abs(raw_lo)))[:2] in label or f"{raw_lo:.2f}".rstrip("0") in label


def test_the_sliders_can_be_dragged_past_the_data(cv):
    """Framing an outlier needs room beyond where the axis currently sits."""
    lo, hi = cv.axes.x_lo, cv.axes.x_hi
    for widget in (cv.w_xmin, cv.w_xmax):
        assert widget.min < lo
        assert widget.max > hi


def test_both_ends_of_an_axis_share_one_travel(cv):
    """Either end can be dragged anywhere the other can."""
    assert (cv.w_xmin.min, cv.w_xmin.max) == (cv.w_xmax.min, cv.w_xmax.max)
    assert (cv.w_ymin.min, cv.w_ymin.max) == (cv.w_ymax.min, cv.w_ymax.max)


def test_dragging_a_slider_pins_that_end_of_the_axis(cv):
    """In display coordinates, which is what the sliders travel in."""
    lo, hi = float(cv.axes.x_lo), float(cv.axes.x_hi)
    cv.w_xmin.value = lo + 0.1
    assert cv.panel.x_lim == pytest.approx((lo + 0.1, hi))
    assert (cv.axes.x_lo, cv.axes.x_hi) == pytest.approx((lo + 0.1, hi))
    cv.w_xmax.value = hi - 0.1
    assert (cv.axes.x_lo, cv.axes.x_hi) == pytest.approx((lo + 0.1, hi - 0.1))


def test_the_y_sliders_move_the_y_axis_alone(cv):
    lo, hi = float(cv.axes.y_lo), float(cv.axes.y_hi)
    x_before = (cv.axes.x_lo, cv.axes.x_hi)
    cv.w_ymax.value = hi - 0.2
    assert (cv.axes.y_lo, cv.axes.y_hi) == pytest.approx((lo, hi - 0.2))
    assert (cv.axes.x_lo, cv.axes.x_hi) == pytest.approx(x_before)


def test_moving_one_end_leaves_the_other_exactly_where_it_was(cv):
    """A slider holds a rounded copy of its end; reading it back drifts."""
    lo = float(cv.axes.x_lo)
    cv.w_xmax.value = float(cv.axes.x_hi) - 0.37
    assert float(cv.axes.x_lo) == lo
    hi = float(cv.axes.x_hi)
    cv.w_xmin.value = lo + 0.21
    assert float(cv.axes.x_hi) == hi


def test_moving_an_end_twice_does_not_drift_the_other(cv):
    """Rounding is per step, so it only shows up after a few drags."""
    lo = float(cv.axes.x_lo)
    for offset in (0.4, 0.31, 0.22):
        cv.w_xmax.value = float(cv.axes.x_hi) - offset
        assert float(cv.axes.x_lo) == lo


def test_the_travel_holds_still_across_a_redraw(cv):
    """Bounds that moved with the view sent the handles back to the middle."""
    before = (cv.w_xmin.min, cv.w_xmin.max)
    cv.w_xmin.value = float(cv.axes.x_lo) + 0.2
    assert (cv.w_xmin.min, cv.w_xmin.max) == pytest.approx(before)
    assert float(cv.w_xmin.value) == pytest.approx(float(cv.axes.x_lo))


def test_the_travel_does_not_narrow_under_the_other_handle(cv):
    """Pinning wide widens it; dragging back in must not move it home again."""
    cv.set_limits(x=(-1000.0, 1_000_000.0))
    wide = (cv.w_xmin.min, cv.w_xmin.max)
    cv.w_xmax.value = float(cv.axes.x_hi) - 1.0
    assert (cv.w_xmin.min, cv.w_xmin.max) == pytest.approx(wide)
    assert (cv.w_xmax.min, cv.w_xmax.max) == pytest.approx(wide)


def test_fit_to_data_re_measures_the_travel(cv):
    """A travel a pinned range widened is not one the data asked for."""
    cv.set_limits(x=(-1000.0, 1_000_000.0))
    wide = float(cv.w_xmax.max)
    cv.w_autoscale.changed.emit(None)
    assert float(cv.w_xmax.max) < wide


def test_a_wide_channel_does_not_break_the_next_narrow_one(cv):
    """The sliders have to survive magicgui rescaling them behind our back.

    magicgui holds a float slider as an integer one times a precision factor,
    and coarsens the factor to fit a wide range. Setting a step finer than the
    factor makes it ten times finer again, without bringing the bounds along:
    the slider is then left refusing the value the axis is sitting at. Time
    spans its detector, a logicle channel spans about one, so going from one
    to the other is all it takes.
    """
    cv.adata.var.loc["Time", "pnr"] = 1e8
    cv.w_x.value = "Time"
    cv.w_x.value = "CD3 (FITC-A)"
    assert cv.w_xmin.min <= cv.axes.x_lo
    assert cv.w_xmax.max >= cv.axes.x_hi
    assert float(cv.w_xmin.value) == pytest.approx(cv.axes.x_lo, abs=0.01)
    assert float(cv.w_xmax.value) == pytest.approx(cv.axes.x_hi, abs=0.01)


def test_changing_channel_re_measures_the_travel(cv):
    """A travel measured on one channel is unusable on the next."""
    before = (cv.w_xmin.min, cv.w_xmin.max)
    cv.w_x.value = "FSC-A"
    assert (cv.w_xmin.min, cv.w_xmin.max) != pytest.approx(before)


def test_dragging_min_past_max_pushes_max_along(cv):
    """An inverted range would read as "fit the data" and lose the view."""
    cv.w_xmax.value = float(cv.axes.x_lo) + 0.5
    top = float(cv.w_xmax.value)
    cv.w_xmin.value = top + 0.3
    lo, hi = cv.axes.x_lo, cv.axes.x_hi
    assert hi > lo
    assert cv.panel.x_lim is not None
    assert float(cv.w_xmax.value) > float(cv.w_xmin.value)


def test_dragging_max_past_min_pushes_min_along(cv):
    cv.w_xmin.value = float(cv.axes.x_lo) + 0.5
    bottom = float(cv.w_xmin.value)
    cv.w_xmax.value = bottom - 0.3
    assert cv.axes.x_hi > cv.axes.x_lo
    assert float(cv.w_xmin.value) < float(cv.w_xmax.value)


def test_a_pinned_range_survives_a_redraw(cv):
    """Changing something unrelated must not quietly refit the axes."""
    cv.set_limits(x=(-100.0, 10_000.0))
    before = (cv.axes.x_lo, cv.axes.x_hi)
    cv.w_smooth.value = 2.0
    assert (cv.axes.x_lo, cv.axes.x_hi) == before


def test_fit_to_data_hands_the_axes_back(cv):
    cv.set_limits(x=(-100.0, 10_000.0), y=(-100.0, 10_000.0))
    pinned = (cv.axes.x_lo, cv.axes.x_hi)
    cv.w_autoscale.changed.emit(None)
    assert cv.panel.x_lim is None and cv.panel.y_lim is None
    assert (cv.axes.x_lo, cv.axes.x_hi) != pinned


def test_an_inverted_range_means_automatic(cv):
    """A cleared field should not strand the plot on a sliver."""
    cv.set_limits(x=(-100.0, 10_000.0))
    cv.set_limits(x=(0.0, 0.0))
    assert cv.panel.x_lim is None


def test_changing_channel_drops_a_range_typed_for_the_old_one(cv):
    cv.set_limits(x=(-100.0, 10_000.0))
    cv.w_x.value = "CD8 (APC-A)"
    assert cv.panel.x_lim is None


def test_a_histogram_ignores_the_y_range(cv):
    """Its vertical axis is per cent of mode, so the fields are off."""
    cv.set_plot(kind="histogram")
    assert not cv.w_ymin.enabled and not cv.w_ymax.enabled
    cv.set_limits(y=(-100.0, 10_000.0))
    assert (cv.axes.y_lo, cv.axes.y_hi) == (0.0, 1.0)


def test_limits_on_an_untransformed_channel_are_plain_values(cv):
    """No transform, so what is typed is what the axis gets."""
    cv.set_plot(x="FSC-A", y="SSC-A")
    cv.set_limits(x=(0.0, 200_000.0))
    assert (cv.axes.x_lo, cv.axes.x_hi) == (0.0, 200_000.0)
