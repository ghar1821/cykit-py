"""The compensation window: gates on the controls, a matrix tuned by eye."""

import numpy as np
import pandas as pd
import pytest

import cytopy

pytest.importorskip("napari")
pytest.importorskip("qtpy")

FLUOR = ["CD3 (FITC-A)", "CD19 (PE-A)", "CD8 (APC-A)"]
COFACTOR = 150.0
#: asinh(1000 / 150) and asinh(200 / 150): the same split as raw > 1000, < 200.
POSITIVE = (float(np.arcsinh(1000 / COFACTOR)), 50.0)
NEGATIVE = (-50.0, float(np.arcsinh(200 / COFACTOR)))


@pytest.fixture
def tubes(controls):
    stained, unstained = controls
    for adata in (*stained.values(), unstained):
        cytopy.asinh_transform(adata, COFACTOR, layer="raw", inplace=True)
    return stained, unstained


def _window(tubes, make_napari_viewer, **kwargs):
    from cytopy.compensation import CompensationWindow

    stained, unstained = tubes
    return CompensationWindow(
        stained, unstained, "asinh", bins=64, viewer=make_napari_viewer(), **kwargs
    )


def _gate_everything(win):
    for detector in win.detectors:
        win.set_control(detector)
        win.apply_gate("positive", POSITIVE)
        win.apply_gate("negative", NEGATIVE)


def _with_file_matrix(stained, matrix):
    for control in stained.values():
        control.uns["spillover"] = pd.DataFrame(matrix, index=FLUOR, columns=FLUOR)


# ------------------------------------------------------------------- gating
def test_without_a_matrix_it_opens_on_the_gating_step_with_the_identity(tubes, make_napari_viewer):
    win = _window(tubes, make_napari_viewer)
    assert win.step == "gate controls"
    assert win.source == "identity"
    assert np.array_equal(win.spillover.to_numpy(), np.eye(3))
    assert win.missing_gates() == FLUOR


def test_gates_are_written_on_the_control_and_never_on_the_unstained(tubes, make_napari_viewer):
    stained, unstained = tubes
    before = list(unstained.obs.columns), repr(unstained.uns.get("cytopy", {}).get("gates"))
    win = _window(tubes, make_napari_viewer)
    mask = win.apply_gate("positive", POSITIVE)

    control = stained[FLUOR[0]]
    raw = np.asarray(control.layers["raw"][:, cytopy.find_channel_name(control, FLUOR[0])])
    assert np.array_equal(mask, raw > 1000.0)
    assert np.array_equal(control.obs["positive"].to_numpy(), mask)
    record = cytopy.gate_record(control, "positive")
    assert record.kind == "histogram" and record.x == FLUOR[0] and record.layer == "asinh"
    # Recorded so that the generic machinery recomputes the same events.
    assert np.array_equal(cytopy.gate_mask(control, "positive"), mask)
    assert (
        list(unstained.obs.columns),
        repr(unstained.uns.get("cytopy", {}).get("gates")),
    ) == before


def test_a_box_drawn_over_both_curves_gates_only_the_control(tubes, make_napari_viewer):
    stained, unstained = tubes
    win = _window(tubes, make_napari_viewer)
    ax = win.panel.axes
    lo, hi = POSITIVE[0], ax.x_hi
    corners = ax.to_pixels(np.array([lo, hi]), np.array([ax.y_lo, ax.y_hi]))
    (r0, c0), (r1, c1) = corners
    win.gates.add(np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]]), shape_type="rectangle")
    assert win.drawn_intervals()[0] == pytest.approx((lo, hi), abs=1e-6)

    mask = win.apply_gate("positive")
    assert mask.shape == (stained[FLUOR[0]].n_obs,)
    assert "positive" not in unstained.obs
    assert len(win.gates.data) == 0


def test_overlapping_gates_and_empty_intervals_are_refused(tubes, make_napari_viewer):
    win = _window(tubes, make_napari_viewer)
    win.apply_gate("positive", POSITIVE)
    with pytest.raises(ValueError, match="overlaps"):
        win.apply_gate("negative", (-50.0, 50.0))
    with pytest.raises(ValueError, match="no events"):
        win.apply_gate("negative", (-1e6, -1e5))
    with pytest.raises(ValueError, match="draw a box"):
        win.apply_gate("negative")


# ------------------------------------------------------------------ compute
def test_compute_waits_for_every_control_then_matches_the_function(tubes, make_napari_viewer):
    stained, _ = tubes
    win = _window(tubes, make_napari_viewer)
    live = win.spillover
    win.apply_gate("positive", POSITIVE)
    assert win.compute() is None
    assert "gate these first" in win.w_status.value

    _gate_everything(win)
    assert win.compute() is live
    assert win.source == "computed" and win.step == "check compensation"
    # The unstained plays no part: the same call without it gives the same matrix.
    expected = cytopy.compute_spillover_matrix(
        stained, positive_gate="positive", negative_gate="negative"
    )
    assert np.allclose(live.to_numpy(), expected.to_numpy())


def test_compute_over_hand_edits_asks_first(tubes, make_napari_viewer):
    win = _window(tubes, make_napari_viewer)
    _gate_everything(win)
    win.compute()
    win.set_coefficient(FLUOR[0], FLUOR[1], 0.5)
    assert win.compute() is None
    assert win.spillover.loc[FLUOR[0], FLUOR[1]] == 0.5
    assert win.compute() is win.spillover
    assert win.spillover.loc[FLUOR[0], FLUOR[1]] != 0.5


# -------------------------------------------------------------- the matrix
def test_a_file_matrix_is_loaded_and_opens_ready_to_tune(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    assert win.source == "file $SPILLOVER"
    assert win.step == "check compensation"
    assert np.allclose(win.spillover.loc[FLUOR, FLUOR].to_numpy(), true_spillover)
    cell = win.table.item(0, 1)
    assert float(cell.text()) == pytest.approx(true_spillover[0, 1], abs=1e-4)


def test_the_check_plots_show_the_controls_compensated_by_the_table(
    tubes, true_spillover, make_napari_viewer
):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    control = stained[FLUOR[0]]
    inverse = np.linalg.inv(win.spillover.to_numpy())
    shown = win.compensated(control, FLUOR[1], inverse)

    by_hand = cytopy.compensate(control, win.spillover)
    j = cytopy.find_channel_name(control, FLUOR[1])
    expected = np.arcsinh(np.asarray(by_hand.layers["comp"][:, j], dtype=float) / COFACTOR)
    rows = win._rows[id(control)]
    expected = expected if rows is None else expected[rows]
    assert np.allclose(shown, expected, atol=1e-3)


def test_editing_a_cell_redraws_and_reset_brings_the_file_back(
    tubes, true_spillover, make_napari_viewer
):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    before = np.array(win.panel.image.data, copy=True)
    r0, c0 = (int(v) for v in win.tile_origin(0, 1))
    t = win.tile

    win.set_coefficient(FLUOR[0], FLUOR[1], 0.0)
    assert win.edited
    after = win.panel.image.data
    assert not np.array_equal(before[r0 : r0 + t, c0 : c0 + t], after[r0 : r0 + t, c0 : c0 + t])
    win.table.item(0, 1).setText("0.3")  # the table goes through the same path
    assert win.spillover.loc[FLUOR[0], FLUOR[1]] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="diagonal"):
        win.set_coefficient(FLUOR[0], FLUOR[0], 0.9)

    win.reset()
    assert not win.edited
    assert np.allclose(win.spillover.loc[FLUOR, FLUOR].to_numpy(), true_spillover)
    assert np.array_equal(before, win.panel.image.data)


def test_a_matrix_that_cannot_be_trusted_is_an_error(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    stained[FLUOR[1]].uns["spillover"] = pd.DataFrame(np.eye(3), index=FLUOR, columns=FLUOR)
    with pytest.raises(ValueError, match="different \\$SPILLOVER"):
        _window(tubes, make_napari_viewer)

    _with_file_matrix(stained, true_spillover * 100)
    with pytest.raises(ValueError, match="diagonal"):
        _window(tubes, make_napari_viewer)


def test_the_starting_matrix_can_be_given_or_refused(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer, spillover=None)
    assert win.source == "identity" and win.step == "gate controls"

    given = pd.DataFrame(np.eye(3), index=FLUOR, columns=FLUOR)
    given.loc[FLUOR[1], FLUOR[2]] = 0.2
    win = _window(tubes, make_napari_viewer, spillover=given)
    assert win.source == "argument"
    assert win.spillover.loc[FLUOR[1], FLUOR[2]] == 0.2


def test_the_window_writes_nothing_but_gates(tubes, true_spillover, make_napari_viewer):
    stained, unstained = tubes
    _with_file_matrix(stained, true_spillover)
    layers = {
        id(a): {k: np.array(v, copy=True) for k, v in a.layers.items()}
        for a in (*stained.values(), unstained)
    }
    win = _window(tubes, make_napari_viewer)
    win.set_coefficient(FLUOR[0], FLUOR[1], 0.2)
    win.set_step("gate controls")
    _gate_everything(win)
    win.compute(overwrite_edits=True)
    for adata in (*stained.values(), unstained):
        assert set(adata.layers) == set(layers[id(adata)])
        for key, matrix in layers[id(adata)].items():
            assert np.array_equal(matrix, np.asarray(adata.layers[key])), key


def test_to_source_round_trips(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    namespace = {"pd": pd}
    exec(win.to_source(), namespace)  # noqa: S102 - our own generated source
    assert np.allclose(namespace["SPILLOVER"].to_numpy(), win.spillover.to_numpy(), atol=1e-6)
    assert list(namespace["SPILLOVER"].index) == FLUOR


# ------------------------------------------------------------- the N x N grid
def test_the_check_grid_has_a_tile_per_control_and_detector(
    tubes, true_spillover, make_napari_viewer
):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer, tile_bins=32)
    step = 32 + 4
    margin = round(1.8 * 32)  # room for the row labels
    assert win.scatter == "SSC-A"
    assert win.tile_origin(2, 1) == (2 * step, margin + step)
    left = margin + 3 * step + 16
    assert win.tile_origin(2, 1, scatter=True) == (2 * step, left + step)
    assert win.panel.image.data.shape == (2 * step + 32, left + 2 * step + 32)
    fluor = {k[:2] for k in win._tile_axes if not k[2]}
    assert fluor == {(r, c) for r in FLUOR for c in FLUOR if r != c}
    # The scatter grid keeps its diagonal: the control's own split.
    assert {k[:2] for k in win._tile_axes if k[2]} == {(r, c) for r in FLUOR for c in FLUOR}
    # Tile (i, j) plots control i's own detector against detector j ...
    ax = win._tile_axes[(FLUOR[0], FLUOR[1], False)]
    assert (ax.x_lo, ax.x_hi) == win._limits[(FLUOR[0], FLUOR[0])]
    assert (ax.y_lo, ax.y_hi) == win._limits[(FLUOR[0], FLUOR[1])]
    # ... and, in the scatter grid, detector j against scatter.
    ax = win._tile_axes[(FLUOR[0], FLUOR[1], True)]
    assert (ax.x_lo, ax.x_hi) == win._limits[(FLUOR[0], FLUOR[1])]
    assert (ax.y_lo, ax.y_hi) == win._limits[(FLUOR[0], "SSC-A")]


def test_a_tile_is_found_from_the_canvas_and_the_table(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    r0, c0 = win.tile_origin(2, 0)
    assert win.tile_at(r0 + 3, c0 + 3) == (FLUOR[2], FLUOR[0])
    assert win.tile_at(*win.tile_origin(1, 1)) is None  # the diagonal
    assert win.tile_at(-5.0, 3.0) is None
    # The scatter grid's tile is the same coefficient.
    r0, c0 = win.tile_origin(2, 0, scatter=True)
    assert win.tile_at(r0 + 3, c0 + 3) == (FLUOR[2], FLUOR[0])
    assert win.tile_at(*win.tile_origin(1, 1, scatter=True)) is None

    win.table.setCurrentCell(0, 1)
    assert win.focus == (FLUOR[0], FLUOR[1])
    assert win.w_status.value.startswith(f"{FLUOR[0]} → {FLUOR[1]}: 0.1200")
    assert len(win.bands.data) == 2  # outlined in both grids

    win.set_focus(FLUOR[1], FLUOR[2])
    assert (win.table.currentRow(), win.table.currentColumn()) == (1, 2)


def test_with_both_gates_a_tile_reports_the_gap_left_behind(tubes, make_napari_viewer):
    win = _window(tubes, make_napari_viewer)
    _gate_everything(win)
    win.compute()
    win.set_focus(FLUOR[0], FLUOR[1])
    assert "pos − neg" in win.w_status.value
    gap_right = abs(float(win.w_status.value.split("pos − neg")[1].replace(",", "")))
    win.set_coefficient(FLUOR[0], FLUOR[1], 0.0)
    win.set_focus(FLUOR[0], FLUOR[1])
    gap_wrong = abs(float(win.w_status.value.split("pos − neg")[1].replace(",", "")))
    assert gap_wrong > 10 * max(gap_right, 1.0)
    # Up to two median lines across each off-diagonal tile, in both grids.
    assert 12 <= len(win.guides.data) <= 2 * 2 * 3 * 2


def test_the_scatter_grid_can_be_changed_or_hidden(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer, tile_bins=32)
    wide = win.panel.image.data.shape[1]
    win.set_scatter("FSC-A")
    assert win.scatter == "FSC-A" and win.w_scatter.value == "FSC-A"
    ax = win._tile_axes[(FLUOR[0], FLUOR[1], True)]
    assert (ax.y_lo, ax.y_hi) == win._limits[(FLUOR[0], "FSC-A")]
    win.set_scatter(None)
    assert not any(k[2] for k in win._tile_axes)
    assert win.panel.image.data.shape[1] < wide
    assert win.tile_at(5.0, wide - 5.0) is None


def test_two_plots_show_what_was_picked_and_follow_the_matrix(
    tubes, true_spillover, make_napari_viewer
):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    win.set_view("two plots")
    assert not win.pair_box.native.isHidden() and win.grid_box.native.isHidden()
    b = win.bins
    assert win.panel.image.data.shape == (b, 2 * b + round(b * 0.45))

    win.set_pair(0, control=FLUOR[0])
    assert win.pair[0] == {"control": FLUOR[0], "x": FLUOR[0], "y": FLUOR[1]}
    win.set_pair(1, control=FLUOR[1], x="FSC-A", y="CD3")
    assert win.pair[1] == {"control": FLUOR[1], "x": "FSC-A", "y": FLUOR[0]}

    before = np.array(win.panel.image.data, copy=True)
    win.set_coefficient(FLUOR[0], FLUOR[1], 0.0)
    after = win.panel.image.data
    assert not np.array_equal(before[:, :b], after[:, :b])  # CD3 against CD19 moved
    assert win.w_status.value.startswith(f"{FLUOR[0]} → {FLUOR[1]}")

    win.set_view("grid")
    assert any(k[2] for k in win._tile_axes)


def test_the_unstained_can_be_hidden_without_moving_the_axis(tubes, make_napari_viewer):
    from cytopy.compensation import _NEGATIVE

    win = _window(tubes, make_napari_viewer)
    win.apply_gate("negative", NEGATIVE)
    axis = (win.panel.axes.x_lo, win.panel.axes.x_hi)
    red = [int(_NEGATIVE[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    assert np.allclose(win.bands.face_color[0][:3], red, atol=0.01)  # the band is red
    with_unstained = len(win.panel.curves.data)

    win.set_show_unstained(False)
    assert win.w_unstained.value is False
    assert len(win.panel.curves.data) == with_unstained - 2  # its fill and its line
    assert (win.panel.axes.x_lo, win.panel.axes.x_hi) == axis
    assert "Unstained" not in " ".join(win.labels.text.values)
    assert "negative" in " ".join(win.labels.text.values)  # the gate share stays

    win.w_unstained.value = True  # the checkbox goes through the same path
    assert win.show_unstained
    assert len(win.panel.curves.data) == with_unstained


# ------------------------------------------------------- adjusting the gates
def test_a_gate_can_be_put_back_adjusted_and_replaced(tubes, make_napari_viewer):
    stained, _ = tubes
    win = _window(tubes, make_napari_viewer)
    win.apply_gate("positive", POSITIVE)
    assert len(win.bands.data) == 1

    win.adjust_gate("positive")
    assert win.editing == "positive"
    assert len(win.bands.data) == 0  # only the editable outline is on screen
    assert len(win.gates.data) == 1
    lo, hi = win.drawn_intervals()[0]
    ax = win.panel.axes
    assert lo == pytest.approx(POSITIVE[0], abs=1e-6)
    assert hi == pytest.approx(min(POSITIVE[1], ax.x_hi), abs=1e-6)

    # Narrow it, as dragging the box's left edge would.
    narrower = (POSITIVE[0] + 1.0, hi)
    corners = ax.to_pixels(np.array(narrower), np.array([ax.y_lo, ax.y_hi]))
    (r0, c0), (r1, c1) = corners
    win.gates.data = [np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])]
    mask = win.apply_gate("positive")
    assert win.editing is None
    assert win.w_status.value.startswith("replaced")
    control = stained[FLUOR[0]]
    shown = np.asarray(control.layers["asinh"][:, cytopy.find_channel_name(control, FLUOR[0])])
    assert np.array_equal(mask, (shown >= narrower[0]) & (shown <= narrower[1]))
    assert np.array_equal(control.obs["positive"].to_numpy(), mask)


def test_moving_away_drops_an_edit_and_leaves_the_gate(tubes, make_napari_viewer):
    stained, _ = tubes
    win = _window(tubes, make_napari_viewer)
    win.apply_gate("positive", POSITIVE)
    before = stained[FLUOR[0]].obs["positive"].to_numpy().copy()
    win.adjust_gate("positive")
    win.step_control(1)
    assert win.editing is None and len(win.gates.data) == 0
    win.step_control(-1)
    assert len(win.bands.data) == 1
    assert np.array_equal(stained[FLUOR[0]].obs["positive"].to_numpy(), before)


def test_a_gate_can_be_deleted(tubes, make_napari_viewer):
    stained, _ = tubes
    win = _window(tubes, make_napari_viewer)
    win.apply_gate("positive", POSITIVE)
    win.apply_gate("negative", NEGATIVE)
    win.delete_gate("negative")
    control = stained[FLUOR[0]]
    assert "negative" not in control.obs
    assert "negative" not in control.uns["cytopy"]["gates"]
    assert "positive" in control.obs
    assert win.gate_state(FLUOR[0]) == (True, False)
    assert len(win.bands.data) == 1
    with pytest.raises(ValueError, match="no negative gate"):
        win.delete_gate("negative")
    with pytest.raises(ValueError, match="no negative gate"):
        win.adjust_gate("negative")


def test_changing_gates_after_computing_is_flagged(tubes, make_napari_viewer):
    win = _window(tubes, make_napari_viewer)
    _gate_everything(win)
    win.compute()
    assert "gates changed" not in win.w_head.value
    win.set_step("gate controls")
    win.delete_gate("positive")
    assert "gates changed since, compute again" in win.w_head.value
    win.apply_gate("positive", POSITIVE)
    win.compute(overwrite_edits=True)
    assert "gates changed" not in win.w_head.value


# ------------------------------------------------- the unstained as negative
def test_a_control_can_take_its_negative_from_the_unstained(tubes, make_napari_viewer):
    stained, unstained = tubes
    win = _window(tubes, make_napari_viewer)
    _gate_everything(win)
    win.set_control(FLUOR[1])
    win.delete_gate("negative")
    assert win.gate_state(FLUOR[1]) == (True, False)

    win.w_use_unstained.value = True  # the checkbox goes through set_use_unstained
    assert win.uses_unstained(FLUOR[1])
    assert win.gate_state(FLUOR[1]) == (True, True)
    assert stained[FLUOR[1]].uns["cytopy"]["use_unstained"] is True  # kept for next time
    assert "negative: unstained" in " ".join(win.labels.text.values)
    assert "unstained" not in unstained.obs.columns  # nothing is written on the tube

    assert win.compute() is win.spillover
    expected = cytopy.compute_spillover_matrix(
        stained,
        unstained=unstained,
        positive_gate="positive",
        negative_gate="negative",
        use_unstained=[FLUOR[1]],
    )
    assert np.allclose(win.spillover.to_numpy(), expected.to_numpy())
    assert win.spillover.attrs["cytopy"]["controls"][FLUOR[1]]["negative"] == "unstained (chosen)"

    win.set_focus(FLUOR[1], FLUOR[2])
    assert "pos − neg" in win.w_status.value


def test_unticking_puts_the_controls_own_gate_back(tubes, make_napari_viewer):
    win = _window(tubes, make_napari_viewer)
    win.apply_gate("positive", POSITIVE)
    win.apply_gate("negative", NEGATIVE)
    win.set_use_unstained(True)
    assert len(win.bands.data) == 1  # the unused negative gate is not drawn
    assert "negative" in win.controls[FLUOR[0]].obs  # ... but it is kept
    win.set_use_unstained(False)
    assert not win.uses_unstained(FLUOR[0])
    assert "use_unstained" not in win.controls[FLUOR[0]].uns["cytopy"]
    assert len(win.bands.data) == 2
    assert win.w_use_unstained.value is False


def test_the_choice_is_per_control_and_survives_reopening(tubes, make_napari_viewer):
    win = _window(tubes, make_napari_viewer)
    win.set_use_unstained(True, FLUOR[2])
    win.step_control(2)
    assert win.control == FLUOR[2] and win.w_use_unstained.value is True
    win.step_control(1)
    assert win.w_use_unstained.value is False
    again = _window(tubes, make_napari_viewer)
    assert again.uses_unstained(FLUOR[2]) and not again.uses_unstained(FLUOR[0])


def test_an_unstained_with_no_raw_layer_cannot_be_used(tubes, make_napari_viewer):
    _, unstained = tubes
    del unstained.layers["raw"]
    win = _window(tubes, make_napari_viewer)  # displaying it needs no raw layer
    win.w_use_unstained.value = True
    assert win.w_use_unstained.value is False
    assert "no 'raw' layer" in win.w_status.value
    assert not win.uses_unstained(FLUOR[0])


# --------------------------------------------------- medians in the gate plot
def _median_columns(win):
    """Display values the median lines sit at, read back off the canvas."""
    ax = win.panel.axes
    cols = np.array([v[0][1] for v in win.guides.data]) - win.panel.col
    return sorted(ax.to_display(np.column_stack([np.zeros_like(cols), cols]))[:, 0])


def _shown(adata, detector):
    return np.asarray(adata.layers["asinh"][:, cytopy.find_channel_name(adata, detector)], float)


def test_each_gate_shows_its_median(tubes, make_napari_viewer):
    stained, _ = tubes
    win = _window(tubes, make_napari_viewer)
    assert len(win.guides.data) == 0
    win.apply_gate("positive", POSITIVE)
    win.apply_gate("negative", NEGATIVE)
    assert len(win.guides.data) == 2
    control = stained[FLUOR[0]]
    shown = _shown(control, FLUOR[0])
    expected = sorted(np.median(shown[control.obs[g].to_numpy()]) for g in ("negative", "positive"))
    step = (win.panel.axes.x_hi - win.panel.axes.x_lo) / win.bins
    assert np.allclose(_median_columns(win), expected, atol=step)
    assert not any(str(t).startswith("median") for t in win.labels.text.values)

    win.adjust_gate("positive")  # the gate being moved has no median yet
    assert len(win.guides.data) == 1


def test_using_the_unstained_draws_its_median_and_nothing_to_gate(tubes, make_napari_viewer):
    _, unstained = tubes
    win = _window(tubes, make_napari_viewer)
    win.apply_gate("positive", POSITIVE)
    win.set_use_unstained(True)
    assert len(win.bands.data) == 1  # the positive band, and no negative box
    assert len(win.guides.data) == 2  # positive median, and the unstained's
    step = (win.panel.axes.x_hi - win.panel.axes.x_lo) / win.bins
    assert _median_columns(win)[0] == pytest.approx(
        np.median(_shown(unstained, FLUOR[0])), abs=step
    )

    win.apply_gate("negative", NEGATIVE)
    assert "not used while 'use unstained as negative' is ticked" in win.w_status.value
    assert len(win.bands.data) == 1


# ------------------------------------------------- ranges of the two plots
def test_the_two_plots_take_typed_axis_ranges(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    win.set_view("two plots")
    auto = win.pair_axes[0]
    lo_box, hi_box = win.w_range[0]["x"]
    # The boxes read back the automatic range, in raw units.
    assert float(lo_box.value.replace(",", "")) == pytest.approx(
        np.sinh(auto.x_lo) * COFACTOR, rel=0.01, abs=0.01
    )

    win.set_pair_range(0, "x", -500.0, 20_000.0)
    ax = win.pair_axes[0]
    assert (ax.x_lo, ax.x_hi) == pytest.approx(np.arcsinh(np.array([-500, 20_000]) / COFACTOR))
    assert (ax.y_lo, ax.y_hi) == (auto.y_lo, auto.y_hi)  # the other axis is untouched
    assert hi_box.value == "20,000"

    # Typed into the boxes: one end, the other left automatic.
    win.w_range[0]["y"][1].value = "5000"
    win._range_entered(0, "y")
    assert win.pair_axes[0].y_hi == pytest.approx(np.arcsinh(5000 / COFACTOR))
    assert win.pair_axes[0].y_lo == pytest.approx(auto.y_lo)
    # Clearing it puts it back.
    win.w_range[0]["y"][1].value = ""
    win._range_entered(0, "y")
    assert win.pair_axes[0].y_hi == pytest.approx(auto.y_hi)


def test_bad_ranges_are_refused_and_moves_reset_them(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    win.set_view("two plots")
    with pytest.raises(ValueError, match="below max"):
        win.set_pair_range(0, "x", 100.0, 10.0)
    win.w_range[0]["x"][0].value = "abc"
    win._range_entered(0, "x")
    assert "not a number" in win.w_status.value

    win.set_pair_range(0, "x", -500.0, 20_000.0)
    win.set_pair_range(0, "y", -500.0, 20_000.0)
    win.set_pair(0, y="CD8")  # a new y channel: its range goes, x keeps its own
    assert win.ranges[0] == {"x": [-500.0, 20_000.0], "y": [None, None]}
    win.set_pair(0, control=FLUOR[1])
    assert win.ranges[0] == {"x": [None, None], "y": [None, None]}
    win.set_pair_range(1, "x", -500.0, 20_000.0)
    win._fit_clicked()
    assert win.ranges[1]["x"] == [None, None]


def test_each_plot_has_its_own_bins_and_keeps_its_size(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    win.set_view("two plots")
    shape = win.panel.image.data.shape
    right_before = np.array(win.panel.image.data[:, -win.bins :], copy=True)

    win.w_bins[0].value = 32  # through the spin box
    assert win.pair_bins == [32, win.bins]
    image = win.panel.image.data
    assert image.shape == shape
    left = image[:, : win.bins]
    # 32 bins drawn on a 64-pixel plot: every bin is a 2 x 2 block.
    assert len(np.unique(left[::2, ::2])) == len(np.unique(left))
    assert np.array_equal(image[:, -win.bins :], right_before)  # the other plot is untouched
    with pytest.raises(ValueError, match="bins must be between"):
        win.set_pair_bins(1, 4)
    with pytest.raises(ValueError, match="bins must be between"):
        win.set_pair_bins(1, 4096)


def test_the_plots_grow_to_the_finer_bins(tubes, true_spillover, make_napari_viewer):
    """No size to choose up front: more bins than the window opened with just works."""
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)  # bins=64
    win.set_view("two plots")
    win.set_pair_bins(1, 256)
    assert win._pair_size == 256
    assert win.panel.image.data.shape == (256, 2 * 256 + round(256 * 0.45))
    left = win.panel.image.data[:, :256]
    # The coarser plot is scaled up to match: its 64 bins are 4 x 4 blocks.
    assert np.array_equal(left, np.repeat(np.repeat(left[::4, ::4], 4, 0), 4, 1))
    # Its axes are still its own, in display units, whatever the pixel size.
    assert win.pair_axes[0].bins == 256
    win.set_pair_bins(1, 32)
    assert win._pair_size == 64


def test_highlighted_cells_have_readable_text(tubes, true_spillover, make_napari_viewer):
    from qtpy.QtGui import QColor

    from cytopy.compensation import _CELL_TEXT

    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    win.set_coefficient(FLUOR[0], FLUOR[1], 0.2)
    for cell in (win.table.item(0, 1), win.table.item(0, 0)):  # edited, and diagonal
        assert cell.foreground().color() == QColor(_CELL_TEXT)


def test_each_cell_has_its_own_minus_and_plus(tubes, true_spillover, make_napari_viewer):
    stained, _ = tubes
    _with_file_matrix(stained, true_spillover)
    win = _window(tubes, make_napari_viewer)
    assert set(win._cells) == {(i, j) for i in range(3) for j in range(3) if i != j}
    minus, box, plus = win._cells[(0, 1)]
    before = float(win.spillover.loc[FLUOR[0], FLUOR[1]])
    assert box.text() == f"{before:.4f}"

    plus.click()
    plus.click()
    assert win.spillover.loc[FLUOR[0], FLUOR[1]] == pytest.approx(before + 0.002)
    assert box.text() == f"{before + 0.002:.4f}"
    assert win.focus == (FLUOR[0], FLUOR[1])  # pressing it picks out its tile
    assert "background" in box.styleSheet()  # marked as edited
    win.w_step_size.setValue(0.01)
    minus.click()
    assert win.spillover.loc[FLUOR[0], FLUOR[1]] == pytest.approx(before - 0.008)

    # Typed into the box, committed on Enter.
    box.setText("0.25")
    box.editingFinished.emit()
    assert win.spillover.loc[FLUOR[0], FLUOR[1]] == pytest.approx(0.25)
    box.setText("abc")
    box.editingFinished.emit()
    assert "not a number" in win.w_status.value
    assert box.text() == "0.2500"

    win.reset()
    assert box.styleSheet() == ""
    assert win.nudge(-3, FLUOR[2], FLUOR[0]) == pytest.approx(-0.03)
    assert (1, 1) not in win._cells  # the diagonal has no buttons


def test_the_matrix_is_square_over_exactly_the_controls(tubes, true_spillover, make_napari_viewer):
    """A file matrix covers the whole panel; a detector nobody stained just drops out."""
    from cytopy.compensation import CompensationWindow

    stained, unstained = tubes
    _with_file_matrix(stained, true_spillover)
    partial = {k: stained[k] for k in FLUOR[:2]}
    win = CompensationWindow(partial, unstained, "asinh", tile_bins=32, viewer=make_napari_viewer())
    assert list(win.spillover.index) == list(win.spillover.columns) == FLUOR[:2]
    assert np.allclose(win.spillover.to_numpy(), true_spillover[:2, :2])  # the file's own values
    assert win.table.rowCount() == win.table.columnCount() == 2
    assert {k[:2] for k in win._tile_axes if not k[2]} == {
        (FLUOR[0], FLUOR[1]),
        (FLUOR[1], FLUOR[0]),
    }
    with pytest.raises(KeyError, match="not in the matrix"):
        win.set_focus(FLUOR[0], FLUOR[2])

    _gate_everything(win)
    win.compute()
    assert list(win.spillover.index) == FLUOR[:2]
    expected = cytopy.compute_spillover_matrix(
        partial, positive_gate="positive", negative_gate="negative"
    )
    assert np.allclose(win.spillover.to_numpy(), expected.to_numpy())

    # Applied, the dropped channel is left exactly as acquired.
    control = stained[FLUOR[2]]
    compensated = cytopy.compensate(control, win.spillover)
    j = cytopy.find_channel_name(control, FLUOR[2])
    assert np.array_equal(compensated.layers["comp"][:, j], control.layers["raw"][:, j])


def test_a_dropped_detectors_diagonal_does_not_matter(tubes, true_spillover, make_napari_viewer):
    from cytopy.compensation import CompensationWindow

    stained, unstained = tubes
    odd = np.array(true_spillover, copy=True)
    odd[2, 2] = 100.0  # nonsense, but only on the detector nobody stained
    _with_file_matrix(stained, odd)
    partial = {k: stained[k] for k in FLUOR[:2]}
    win = CompensationWindow(partial, unstained, "asinh", viewer=make_napari_viewer())
    assert win.spillover.shape == (2, 2)
