"""The cofactor window: it transforms for display, and writes nothing back."""

import ast

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

RANGE = (1.0, 10_000.0)


@pytest.fixture
def cw(demo, make_napari_viewer):
    from cytopy.cofactors import CofactorWindow

    return CofactorWindow(demo, "raw", cofactor_range=RANGE, bins=64, viewer=make_napari_viewer())


def _snapshot(adata):
    return {
        "X": np.array(adata.X, copy=True),
        "layers": {k: np.array(v, copy=True) for k, v in adata.layers.items()},
        "var": adata.var.copy(deep=True),
        "obs": adata.obs.copy(deep=True),
        "uns": repr(adata.uns.get("cytopy", {})),
    }


def _assert_same(before, adata):
    assert np.array_equal(before["X"], np.asarray(adata.X))
    assert set(before["layers"]) == set(adata.layers)
    for key, matrix in before["layers"].items():
        assert np.array_equal(matrix, np.asarray(adata.layers[key])), key
    assert list(before["var"].columns) == list(adata.var.columns)
    assert list(before["obs"].columns) == list(adata.obs.columns)
    assert before["uns"] == repr(adata.uns.get("cytopy", {}))


# --------------------------------------------------------------------- contract
def test_the_transform_window_writes_nothing_to_the_adata(cw):
    before = _snapshot(cw.adata)
    cw.set_cofactor(cw.x, 12.0)
    cw.step(1)
    cw.set_cofactor(cw.x, 4000.0)
    cw.set_channels(y="CD8 (APC-A)")
    cw.set_cofactor(cw.y, 900.0)
    _assert_same(before, cw.adata)
    assert "cofactor" not in cw.adata.var
    assert "asinh_layer" not in cw.adata.uns.get("cytopy", {})
    assert "asinh" not in cw.adata.layers


def test_the_displayed_values_are_the_stored_values_arcsinh_the_current_cofactor(cw, demo):
    import cytopy

    cw.set_cofactor(cw.x, 37.0)
    j = cytopy.find_channel_name(demo, cw.x)
    raw = np.asarray(demo.layers["raw"][:, j], dtype=np.float32).ravel()
    shown = cw._channel_data(cw.x).display(37.0)
    assert np.allclose(shown, np.arcsinh(raw / 37.0), atol=1e-6)


# ---------------------------------------------------------------------- seeding
def test_every_channel_has_a_cofactor_before_anything_is_touched(cw):
    assert set(cw.cofactors) == set(cw.channels)
    assert cw.cofactors.n_adjusted == 0
    assert sorted(cw.cofactors.untouched) == sorted(cw.channels)


def test_the_seed_is_the_geometric_midpoint_of_the_range_when_none_is_given(cw):
    midpoint = float(np.sqrt(RANGE[0] * RANGE[1]))
    assert all(v == pytest.approx(midpoint) for v in cw.cofactors.values())


def test_a_mapping_seed_is_read_the_way_asinh_transform_reads_one(demo, make_napari_viewer):
    from cytopy.cofactors import CofactorWindow

    window = CofactorWindow(
        demo,
        "raw",
        cofactor_range=RANGE,
        bins=64,
        viewer=make_napari_viewer(),
        cofactor={"CD3": 40.0, "PE-A": 250.0, "default": 900.0},
    )
    assert window.cofactors["CD3 (FITC-A)"] == 40.0  # by marker
    assert window.cofactors["CD19 (PE-A)"] == 250.0  # by detector
    assert window.cofactors["CD8 (APC-A)"] == 900.0  # by the default


def test_a_seed_outside_the_range_is_refused_rather_than_clamped(demo, make_napari_viewer):
    from cytopy.cofactors import CofactorWindow

    with pytest.raises(ValueError, match="outside cofactor_range"):
        CofactorWindow(
            demo,
            "raw",
            cofactor_range=(1.0, 100.0),
            bins=64,
            viewer=make_napari_viewer(),
            cofactor=5000.0,
        )


@pytest.mark.parametrize("bad", [(0.0, 10.0), (10.0, 1.0), (1.0, np.inf), (5.0, 5.0), 7.0])
def test_the_range_must_be_positive_and_increasing(demo, make_napari_viewer, bad):
    from cytopy.cofactors import CofactorWindow

    with pytest.raises(ValueError, match="cofactor_range"):
        CofactorWindow(demo, "raw", cofactor_range=bad, bins=64, viewer=make_napari_viewer())


# -------------------------------------------------------------------- traversal
def test_next_channel_advances_in_file_order_and_wraps(cw):
    assert cw.x == cw.channels[0]
    for expected in cw.channels[1:] + cw.channels[:1]:
        cw.step(1)
        assert cw.x == expected
    cw.step(-1)
    assert cw.x == cw.channels[-1]


def test_coming_back_to_a_channel_restores_its_slider(cw):
    first = cw.x
    cw.set_cofactor(first, 500.0)
    cw.step(1)
    assert cw.w_cx.value != pytest.approx(500.0)
    cw.step(-1)
    assert cw.x == first
    # The dict is the state and keeps the value exactly; the slider is a view of
    # it and lands on the nearest of its thousand log-spaced positions.
    assert cw.cofactors[first] == 500.0
    assert cw.w_cx.value == pytest.approx(500.0, rel=0.01)


def test_an_adjusted_channel_is_marked_in_the_channel_list(cw):
    labels = [label for label, _ in cw._channel_choices()]
    assert all(label.startswith("○") for label in labels)
    cw.set_cofactor(cw.x, 300.0)
    marked = {value: label for label, value in cw._channel_choices()}
    assert marked[cw.x].startswith("●")
    assert all(marked[c].startswith("○") for c in cw.channels if c != cw.x)


def test_setting_every_channel_overwrites_decisions_and_clears_the_marks(cw):
    decided, *rest = cw.channels
    cw.set_cofactor(decided, 42.0)
    cw.set_all(7000.0)
    assert all(cw.cofactors[c] == pytest.approx(7000.0) for c in [decided, *rest])
    assert cw.cofactors.n_adjusted == 0
    assert sorted(cw.cofactors.untouched) == sorted(cw.channels)


def test_the_bulk_box_applies_on_enter_and_echoes_what_it_applied(cw):
    cw.w_all.value = "7,000"
    cw._all_entered()
    assert all(v == pytest.approx(7000.0) for v in cw.cofactors.values())
    assert cw.w_all.value == "7000"


def test_the_bulk_box_clamps_into_the_range_and_says_so(cw):
    cw.w_all.value = str(RANGE[1] * 10)
    cw._all_entered()
    assert all(v == pytest.approx(RANGE[1]) for v in cw.cofactors.values())
    assert "clamped" in cw.w_status.value


def test_the_bulk_box_refuses_a_non_number_and_changes_nothing(cw):
    before = dict(cw.cofactors)
    cw.w_all.value = "three thousand"
    cw._all_entered()
    assert dict(cw.cofactors) == before
    assert "not a number" in cw.w_status.value


def test_the_bulk_box_ignores_an_empty_entry(cw):
    before = dict(cw.cofactors)
    cw.set_cofactor(cw.x, 42.0)
    cw.w_all.value = "   "
    cw._all_entered()
    assert cw.cofactors[cw.x] == pytest.approx(42.0)
    assert set(cw.cofactors) == set(before)


def test_the_axis_label_follows_the_chosen_x_channel(cw):
    """The frame labels its axes from the panel, which has to be re-pointed."""
    assert cw.p_x.x == cw.x and cw.p_density.x == cw.x
    cw.step(1)
    assert cw.p_x.x == cw.x
    assert cw.p_density.x == cw.x
    cw.set_channels(x=cw.channels[-1])
    assert cw.p_x.x == cw.channels[-1]
    assert cw.p_density.x == cw.channels[-1]


def test_the_y_panel_label_follows_the_chosen_y_channel(cw):
    cw.set_channels(y="CD8 (APC-A)")
    assert cw.p_y.x == "CD8 (APC-A)"
    assert cw.p_density.y == "CD8 (APC-A)"


# ---------------------------------------------------------------- the slider
def test_every_slider_position_survives_a_redraw(cw):
    """magicgui truncates value -> position, which walks the handle leftwards.

    Every redraw writes the cofactor back to the widget, so a position that
    does not survive that round trip is one the handle drifts away from while
    the number beside it stays put.
    """
    from cytopy.cofactors import SLIDER_STEPS

    q = cw.w_cx._widget._qwidget
    moved = []
    for pos in range(0, SLIDER_STEPS + 1, 7):
        q.setValue(pos)
        cw.w_cx.value = cw.w_cx.value  # what _bind does on every redraw
        if q.value() != pos:
            moved.append(pos)
    assert not moved


def test_both_ends_of_the_range_are_reachable(cw):
    from cytopy.cofactors import SLIDER_STEPS

    q = cw.w_cx._widget._qwidget
    for value, want in ((RANGE[0], 0), (RANGE[1], SLIDER_STEPS)):
        cw.w_cx.value = value
        assert q.value() == want
        assert cw.w_cx.value == pytest.approx(value)


def test_the_slider_rounds_rather_than_truncates(cw):
    """Truncation biases every cofactor downwards; rounding is symmetric."""
    errors = []
    for value in np.geomspace(RANGE[0], RANGE[1], 200):
        cw.w_cx.value = float(value)
        errors.append((cw.w_cx.value - value) / value)
    errors = np.asarray(errors)
    assert errors.min() < 0 < errors.max()
    assert np.abs(errors).max() < 0.005


# ------------------------------------------------------- typed cofactor boxes
def test_the_box_stores_the_typed_value_exactly(cw):
    """The slider quantises; what it rounds to must not become the answer."""
    cw.w_cx_box.value = "3000"
    cw._box_entered(cw.w_cx_box, "x")
    assert cw.cofactors[cw.x] == 3000.0
    assert cw.x in cw.cofactors.adjusted
    # the slider moved to the nearest step it has, which is not exactly 3000
    assert cw.w_cx.value == pytest.approx(3000.0, rel=0.01)


def test_the_box_follows_the_slider(cw):
    cw.set_cofactor(cw.x, 42.0)
    assert cw.w_cx_box.value == "42"
    cw.step(1)
    assert cw.w_cx_box.value == f"{cw.cofactors[cw.x]:g}"


def test_the_box_accepts_a_grouped_number(cw):
    cw.w_cx_box.value = "1,250"
    cw._box_entered(cw.w_cx_box, "x")
    assert cw.cofactors[cw.x] == 1250.0


def test_the_box_clamps_into_the_range_and_says_so(cw):
    cw.w_cx_box.value = str(RANGE[1] * 10)
    cw._box_entered(cw.w_cx_box, "x")
    assert cw.cofactors[cw.x] == pytest.approx(RANGE[1])
    assert "clamped" in cw.w_status.value


def test_the_box_refuses_a_non_number_and_puts_the_value_back(cw):
    cw.set_cofactor(cw.x, 42.0)
    cw.w_cx_box.value = "nope"
    cw._box_entered(cw.w_cx_box, "x")
    assert cw.cofactors[cw.x] == pytest.approx(42.0)
    assert "not a number" in cw.w_status.value
    assert cw.w_cx_box.value == "42"


def test_the_y_box_is_dead_while_y_is_a_scatter_channel(cw):
    assert not cw._y_tuned
    assert not cw.w_cy_box.enabled
    assert cw.w_cy_box.value == ""
    cw._box_entered(cw.w_cy_box, "y")  # must not raise, and must change nothing
    assert cw.y not in cw.cofactors


def test_the_y_box_comes_alive_on_a_fluorescence_channel(cw):
    cw.set_channels(y="CD8 (APC-A)")
    assert cw.w_cy_box.enabled
    cw.w_cy_box.value = "900"
    cw._box_entered(cw.w_cy_box, "y")
    assert cw.cofactors["CD8 (APC-A)"] == 900.0


# ----------------------------------------------------------------- axis margin
def test_the_default_margin_keeps_the_axis_near_the_data(cw):
    """The axis must not run decades past the last event it has to show."""
    import cytopy
    from cytopy.scales import AXIS_MARGIN

    assert cw.margin == AXIS_MARGIN
    cw.set_cofactor(cw.x, 500.0)
    j = cytopy.find_channel_name(cw.adata, cw.x)
    raw = np.asarray(cw.adata.layers["raw"][:, j]).ravel()
    top = float(np.sinh(cw.p_x.axes.x_hi) * 500.0)
    assert top < 2.0 * raw.max()


def test_a_wider_margin_widens_the_axis_and_leaves_the_data_alone(cw):
    cw.set_cofactor(cw.x, 500.0)
    before = (cw.p_x.axes.x_lo, cw.p_x.axes.x_hi)
    cw.set_margin(0.4)
    after = (cw.p_x.axes.x_lo, cw.p_x.axes.x_hi)
    assert after[0] < before[0] and after[1] > before[1]
    assert cw.cofactors[cw.x] == pytest.approx(500.0)


def test_a_zero_margin_pins_the_axis_to_the_quantiles(cw):
    cw.set_margin(0.0)
    d = cw._channel_data(cw.x)
    c = cw.cofactors[cw.x]
    assert cw.p_x.axes.x_lo == pytest.approx(float(np.arcsinh(d.q_lo / c)))
    assert cw.p_x.axes.x_hi == pytest.approx(float(np.arcsinh(d.q_hi / c)))


def test_the_margin_slider_drives_it_in_per_cent(cw):
    cw.w_margin.value = 25
    cw._margin_changed()
    assert cw.margin == pytest.approx(0.25)
    cw.set_margin(0.05)
    assert cw.w_margin.value == 5


@pytest.mark.parametrize("bad", [-0.1, 1.0, 10, float("nan")])
def test_an_out_of_range_margin_is_refused(cw, bad):
    with pytest.raises(ValueError, match="margin must be a proportion"):
        cw.set_margin(bad)


# ------------------------------------------------------------------ axis ticks
def test_untransformed_ticks_label_the_original_units(cw):
    from cytopy.scales import PretransformedScale

    assert cw.w_ticks.value == "untransformed"
    cw.set_cofactor(cw.x, 500.0)
    assert isinstance(cw.p_x.x_scale, PretransformedScale)
    labels = cw.p_x.x_scale.ticks(cw.p_x.axes.x_lo, cw.p_x.axes.x_hi).labels
    assert any(lbl.lstrip("-").startswith("10") for lbl in labels)


def test_transformed_ticks_label_the_arcsinh_values(cw):
    from cytopy.scales import LinearScale

    cw.w_ticks.value = "transformed"
    assert isinstance(cw.p_x.x_scale, LinearScale)
    labels = cw.p_x.x_scale.ticks(cw.p_x.axes.x_lo, cw.p_x.axes.x_hi).labels
    # single-digit arcsinh units, not decades of raw signal
    assert not any(lbl.lstrip("-").startswith("10") for lbl in labels)


def test_the_tick_mode_moves_nothing(cw):
    cw.set_cofactor(cw.x, 500.0)
    before = (cw.p_x.axes.x_lo, cw.p_x.axes.x_hi, np.array(cw.p_density.image.data, copy=True))
    cw.w_ticks.value = "transformed"
    assert (cw.p_x.axes.x_lo, cw.p_x.axes.x_hi) == before[:2]
    assert np.array_equal(cw.p_density.image.data, before[2])


def test_an_unknown_tick_mode_is_refused(demo, make_napari_viewer):
    from cytopy.cofactors import CofactorWindow

    with pytest.raises(ValueError, match="ticks must be one of"):
        CofactorWindow(
            demo,
            "raw",
            cofactor_range=RANGE,
            bins=64,
            viewer=make_napari_viewer(),
            ticks="raw units",
        )


def test_relabelling_the_channel_list_does_not_change_the_selected_channel(cw):
    cw.step(1)
    chosen = cw.x
    cw._bind()
    assert cw.w_x.value == chosen
    assert cw.x == chosen


# ----------------------------------------------------------------------- y axis
def test_y_defaults_to_a_scatter_channel(cw):
    assert cw.y == "SSC-A"
    assert cw.y not in cw.cofactors


def test_the_y_slider_is_greyed_out_while_y_is_scatter(cw):
    assert not cw.w_cy.enabled
    cw.set_channels(y="CD19 (PE-A)")
    assert cw.w_cy.enabled
    cw.set_cofactor(cw.y, 640.0)
    assert cw.cofactors["CD19 (PE-A)"] == pytest.approx(640.0)


def test_y_falls_back_to_another_channel_when_there_is_no_scatter(demo, make_napari_viewer):
    from cytopy.cofactors import CofactorWindow

    fluor = [c for c in demo.var_names if demo.var["kind"].loc[c] == "fluor"]
    window = CofactorWindow(
        demo[:, fluor].copy(),
        "raw",
        cofactor_range=RANGE,
        bins=64,
        viewer=make_napari_viewer(),
    )
    assert window.y in fluor
    assert window.y != window.x


def test_moving_the_x_slider_leaves_the_y_histogram_alone(cw):
    before = [np.array(s, copy=True) for s in cw.p_y.curves.data]
    fill = np.array(cw.p_y.image.data, copy=True)
    cw.set_cofactor(cw.x, 3000.0)
    after = cw.p_y.curves.data
    assert len(before) == len(after)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert np.array_equal(fill, cw.p_y.image.data)


# ------------------------------------------------------------ redraw and caching
def test_all_three_panels_share_one_canvas(cw):
    assert [layer.name for layer in cw.viewer.layers] == [
        "panel 1",
        "panel 1 curves",
        "panel 2",
        "panel 2 curves",
        "panel 3",
        "panel 3 curves",
        "guides",
        "axes",
        "axis labels",
        "y axis labels",
    ]
    assert cw.p_density.image.visible and not cw.p_density.curves.visible
    # A histogram panel uses both: the image is the shaded area under the curve,
    # the shape is its outline.
    for panel in (cw.p_x, cw.p_y):
        assert panel.image.visible and panel.curves.visible
        assert len(panel.curves.data) == 1


def test_the_density_and_the_x_histogram_redraw_when_the_x_slider_moves(cw):
    density = np.array(cw.p_density.image.data, copy=True)
    outline = np.array(cw.p_x.curves.data[0], copy=True)
    fill = np.array(cw.p_x.image.data, copy=True)
    cw.set_cofactor(cw.x, 4.0)
    assert not np.array_equal(density, cw.p_density.image.data)
    assert not np.array_equal(outline, cw.p_x.curves.data[0])
    assert not np.array_equal(fill, cw.p_x.image.data)


def test_the_panels_do_not_overlap(cw):
    b = cw.bins
    assert cw.p_x.col >= cw.p_density.col + b
    assert cw.p_y.row >= cw.p_x.row + b
    assert cw.p_y.col == cw.p_x.col


def test_a_larger_cofactor_compresses_the_axis(cw):
    cw.set_cofactor(cw.x, 10.0)
    narrow = cw.p_density.axes.x_hi
    cw.set_cofactor(cw.x, 1000.0)
    assert cw.p_density.axes.x_hi < narrow


def test_the_axis_limits_come_from_raw_quantiles_transformed_not_re_percentiled(cw):
    from cytopy.scales import AsinhScale, PretransformedScale

    data = cw._channel_data(cw.x)
    for cofactor in (5.0, 150.0, 3000.0):
        honest = PretransformedScale(AsinhScale(cofactor=cofactor)).limits(
            np.arcsinh(np.asarray(data.raw, dtype=float) / cofactor), quantiles=(0.001, 0.999)
        )
        assert data.limits(cofactor) == pytest.approx(honest, rel=1e-6, abs=1e-9)


def test_a_channels_raw_quantiles_are_computed_once_per_channel(cw, monkeypatch):
    calls = []
    real = np.quantile
    monkeypatch.setattr(np, "quantile", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    cw._channel_data(cw.x)  # already cached by the first draw
    before = len(calls)
    for cofactor in (10.0, 100.0, 1000.0, 2000.0, 5000.0):
        cw.set_cofactor(cw.x, cofactor)
    assert len(calls) == before


def test_the_raw_column_cache_is_bounded(demo, make_napari_viewer):
    from cytopy.cofactors import CACHE_SIZE, CofactorWindow

    window = CofactorWindow(demo, "raw", cofactor_range=RANGE, bins=64, viewer=make_napari_viewer())
    for name in list(demo.var_names) * 3:
        window._channel_data(str(name))
    assert len(window._cache) <= CACHE_SIZE


def test_the_statistics_are_computed_on_every_event_not_the_subsample(demo, make_napari_viewer):
    import cytopy
    from cytopy.cofactors import CofactorWindow

    window = CofactorWindow(
        demo,
        "raw",
        cofactor_range=RANGE,
        bins=64,
        max_events=1_000,
        viewer=make_napari_viewer(),
    )
    assert window.n_plotted == 1_000
    j = cytopy.find_channel_name(demo, window.x)
    column = np.asarray(demo.layers["raw"][:, j], dtype=np.float32).ravel()
    assert window.statistics()["n_negative"] == float((column < 0).sum())


def test_the_same_seed_draws_the_same_events(demo, make_napari_viewer):
    from cytopy.cofactors import CofactorWindow

    made = [
        CofactorWindow(
            demo,
            "raw",
            cofactor_range=RANGE,
            bins=64,
            max_events=5_000,
            seed=3,
            viewer=make_napari_viewer(),
        )
        for _ in range(2)
    ]
    assert np.array_equal(made[0].p_density.image.data, made[1].p_density.image.data)


# --------------------------------------------------------------------- overlays
def test_the_knee_line_sits_at_asinh_of_one_whatever_the_cofactor(cw):
    from cytopy.cofactors import KNEE

    assert KNEE == pytest.approx(float(np.arcsinh(1.0)))
    seen = []
    for cofactor in (30.0, 3000.0):
        cw.set_cofactor(cw.x, cofactor)
        axes = cw.p_x.axes
        column = cw.p_x.col + axes.to_pixels(np.array([KNEE]), np.array([axes.y_lo]))[0, 1]
        seen.append(column)
        assert any(np.isclose(np.asarray(shape)[:, 1], column).all() for shape in cw.guides.data), (
            cofactor
        )
    # The data slides under it, so the line itself barely moves.
    assert seen[0] == pytest.approx(seen[1], rel=0.2)


def test_the_negative_band_widens_as_the_cofactor_falls(cw):
    def band_width():
        widths = [
            float(np.asarray(s)[:, 1].max() - np.asarray(s)[:, 1].min())
            for s in cw.guides.data
            if len(s) == 4
        ]
        return max(widths, default=0.0)

    cw.set_cofactor(cw.x, 3000.0)
    narrow = band_width()
    cw.set_cofactor(cw.x, 30.0)
    assert band_width() > narrow


def test_a_channel_with_no_negative_population_says_so(demo, make_napari_viewer):
    import cytopy
    from cytopy.cofactors import CofactorWindow

    positive = demo.copy()
    j = cytopy.find_channel_name(positive, "CD3 (FITC-A)")
    raw = np.asarray(positive.layers["raw"], dtype=np.float32)
    raw[:, j] = np.abs(raw[:, j]) + 1.0
    positive.layers["raw"] = raw

    window = CofactorWindow(
        positive,
        "raw",
        "CD3 (FITC-A)",
        cofactor_range=RANGE,
        bins=64,
        viewer=make_napari_viewer(),
    )
    assert "nothing to sit on" in window.w_status.value
    assert np.isnan(window.statistics()["spread"])


# ----------------------------------------------------------------------- result
def test_the_repr_is_a_python_dict_literal_that_round_trips(cw):
    cw.set_cofactor(cw.x, 812.0)
    assert ast.literal_eval(repr(cw.cofactors)) == dict(cw.cofactors)
    assert cw.cofactors.to_source("COFACTORS").startswith("COFACTORS = {")


def test_the_cofactors_go_straight_into_asinh_transform(cw, demo):
    import cytopy

    cw.set_cofactor(cw.x, 220.0)
    cw.step(1)
    cw.set_cofactor(cw.x, 1750.0)

    out = cytopy.asinh_transform(demo, cw.cofactors, layer="raw")
    for name, cofactor in cw.cofactors.items():
        j = cytopy.find_channel_name(out, name)
        assert float(out.var["cofactor"].iloc[j]) == pytest.approx(cofactor)
        raw = np.asarray(out.layers["raw"][:, j], dtype=float)
        assert np.allclose(
            np.asarray(out.layers["asinh"][:, j], dtype=float),
            np.arcsinh(raw / cofactor),
            atol=1e-5,
        )


def test_the_dict_it_returns_can_seed_the_next_session(cw, demo, make_napari_viewer):
    from cytopy.cofactors import CofactorWindow

    cw.set_cofactor(cw.x, 640.0)
    again = CofactorWindow(
        demo,
        "raw",
        cofactor_range=RANGE,
        bins=64,
        viewer=make_napari_viewer(),
        cofactor=dict(cw.cofactors),
    )
    assert dict(again.cofactors) == dict(cw.cofactors)


@pytest.fixture
def on_fixture_window(make_napari_viewer, monkeypatch):
    """Let ``open_napari_transform`` build on the test window instead of its own.

    It normally calls ``napari.Viewer()`` itself, which re-registers the plugin
    actions and trips over napari's own fixture. Everything else runs for real.
    """
    import napari

    import cytopy.cofactors as module

    window = make_napari_viewer()
    monkeypatch.setattr(napari, "Viewer", lambda **kwargs: window)
    yield window
    module._CURRENT_TRANSFORM = None


def test_open_napari_transform_returns_the_live_dict_and_leaves_the_data_alone(
    demo, on_fixture_window, capsys
):
    import cytopy
    from cytopy.cofactors import Cofactors, current_transform_window

    before = _snapshot(demo)
    result = cytopy.open_napari_transform(demo, "raw", cofactor_range=RANGE, bins=64, block=False)
    assert isinstance(result, Cofactors)
    assert set(result) == set(cytopy.get_fluor_channels(demo))
    assert capsys.readouterr().out == ""  # silent unless asked

    window = current_transform_window()
    assert window is not None
    window.set_cofactor(window.x, 333.0)
    assert result[window.x] == pytest.approx(333.0)  # the same live object
    _assert_same(before, demo)


def test_open_napari_transform_prints_the_literal_when_verbose(demo, on_fixture_window, capsys):
    import cytopy

    cytopy.open_napari_transform(
        demo, "raw", cofactor_range=RANGE, bins=64, block=False, verbose=True
    )
    printed = capsys.readouterr().out
    assert printed.startswith("COFACTORS = {")
    assert "CD3 (FITC-A)" in printed


def test_open_napari_transform_can_start_on_chosen_channels(demo, on_fixture_window):
    import cytopy
    from cytopy.cofactors import current_transform_window

    cytopy.open_napari_transform(
        demo,
        "raw",
        "CD8 (APC-A)",
        "CD19 (PE-A)",
        cofactor_range=RANGE,
        bins=64,
        block=False,
    )
    window = current_transform_window()
    assert (window.x, window.y) == ("CD8 (APC-A)", "CD19 (PE-A)")
    assert window.w_cy.enabled  # y is a tunable channel here
