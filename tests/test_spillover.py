"""Getting a spillover matrix: from controls, from the file, from someone else's CSV."""

import numpy as np
import pandas as pd
import pytest

import cytopy

FLUOR = ["CD3 (FITC-A)", "CD19 (PE-A)", "CD8 (APC-A)"]


def _matrix(values, names=FLUOR):
    return pd.DataFrame(np.asarray(values, dtype=float), index=names, columns=names)


# --------------------------------------------------------------------------
# deriving from single-stain controls
# --------------------------------------------------------------------------
def test_controls_are_a_mapping_the_caller_builds(controls):
    """Which file stains which detector is the caller's to state, not ours to guess."""
    stained, unstained = controls
    assert sorted(stained) == sorted(FLUOR)
    assert unstained is not None
    assert (unstained.obs["sample"] == "Unstained").all()
    assert stained["CD3 (FITC-A)"].obs["sample"].iloc[0] == "Compensation Controls_FITC-A"
    # The controls carry no matrix of their own; that is the point of them.
    assert "spillover" not in stained["CD3 (FITC-A)"].uns


def test_compute_spillover_matrix_recovers_the_true_matrix(controls, true_spillover):
    stained, _ = controls
    spill = cytopy.compute_spillover_matrix(stained)
    assert list(spill.index) == FLUOR and list(spill.columns) == FLUOR
    assert np.allclose(np.diag(spill), 1.0)
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)


def test_compute_spillover_matrix_against_an_unstained_reference(controls, true_spillover):
    stained, unstained = controls
    spill = cytopy.compute_spillover_matrix(stained, unstained=unstained)
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)
    assert spill.attrs["cytopy"]["negative"] == "unstained"


def test_compute_spillover_matrix_with_means(controls, true_spillover):
    stained, unstained = controls
    spill = cytopy.compute_spillover_matrix(stained, unstained=unstained, statistic="mean")
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.005)
    with pytest.raises(ValueError, match="median.*mean"):
        cytopy.compute_spillover_matrix(stained, statistic="mode")


def test_compute_spillover_matrix_accepts_markers_and_paths(controls, controls_dir, true_spillover):
    stained, _ = controls
    by_marker = {"CD3": stained[FLUOR[0]], "CD19": stained[FLUOR[1]], "APC-A": stained[FLUOR[2]]}
    assert np.allclose(
        cytopy.compute_spillover_matrix(by_marker).to_numpy(), true_spillover, atol=0.01
    )
    by_path = {"FITC-A": controls_dir / "Compensation Controls_FITC-A.fcs"}
    assert cytopy.compute_spillover_matrix(by_path).shape == (1, 1)


def test_compute_spillover_matrix_uses_a_gate_when_given_one(controls, true_spillover):
    stained, _ = controls
    for name, control in stained.items():
        j = cytopy.find_channel_name(control, name)
        control.obs["P1"] = np.asarray(control.X[:, j]) > 1000.0
    spill = cytopy.compute_spillover_matrix(stained, positive_gate="P1")
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)
    counts = spill.attrs["cytopy"]["controls"][FLUOR[0]]
    assert counts["positive_events"] == int((stained[FLUOR[0]].obs["P1"]).sum())


def test_compute_spillover_matrix_takes_explicit_thresholds(controls, true_spillover):
    stained, _ = controls
    spill = cytopy.compute_spillover_matrix(stained, thresholds=dict.fromkeys(stained, 1000.0))
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)


def test_an_all_positive_bead_control_needs_an_unstained_tube(controls, true_spillover):
    """A tube with no negatives of its own: -inf takes every event as positive."""
    stained, unstained = controls
    beads = {}
    for name, control in stained.items():
        column = np.asarray(control.X[:, cytopy.find_channel_name(control, name)])
        beads[name] = control[column > 1000].copy()

    spill = cytopy.compute_spillover_matrix(
        beads, unstained=unstained, thresholds=dict.fromkeys(beads, -np.inf)
    )
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)
    with pytest.raises(ValueError, match="negative events"):
        cytopy.compute_spillover_matrix(beads, thresholds=dict.fromkeys(beads, -np.inf))


def test_a_detector_without_a_control_only_appears_if_asked_for(controls):
    stained, _ = controls
    partial = {k: v for k, v in stained.items() if k != FLUOR[1]}
    assert list(cytopy.compute_spillover_matrix(partial).columns) == [FLUOR[0], FLUOR[2]]

    spill = cytopy.compute_spillover_matrix(partial, channels=FLUOR)
    assert list(spill.columns) == FLUOR
    # The unconstrained detector gets an identity row: it spills into nothing.
    assert np.allclose(spill.loc[FLUOR[1]], [0.0, 1.0, 0.0])
    # ... but the spill of the other dyes *into* it is still measured.
    assert spill.loc[FLUOR[0], FLUOR[1]] == pytest.approx(0.12, abs=0.01)


def test_compute_spillover_matrix_rejects_nonsense(controls):
    stained, unstained = controls
    with pytest.raises(ValueError, match="no controls"):
        cytopy.compute_spillover_matrix({})
    with pytest.raises(ValueError, match="does not separate into two populations"):
        cytopy.compute_spillover_matrix({"CD3": unstained}, min_separation=5.0)
    # A gate that picks the dim events instead of the bright ones.
    control = stained[FLUOR[0]].copy()
    column = np.asarray(control.X[:, cytopy.find_channel_name(control, FLUOR[0])])
    control.obs["inverted"] = column < np.median(column)
    with pytest.raises(ValueError, match="not brighter than its negatives"):
        cytopy.compute_spillover_matrix({FLUOR[0]: control}, positive_gate="inverted")
    with pytest.raises(ValueError, match="omits detectors"):
        cytopy.compute_spillover_matrix(stained, channels=[FLUOR[0]])
    with pytest.raises(ValueError, match="same detector"):
        cytopy.compute_spillover_matrix({"CD3": stained[FLUOR[0]], "FITC-A": stained[FLUOR[0]]})
    with pytest.raises(ValueError, match="positive events"):
        cytopy.compute_spillover_matrix(stained, min_events=10**9)


def test_a_derived_matrix_compensates_like_the_files_own(demo, controls):
    stained, unstained = controls
    spill = cytopy.compute_spillover_matrix(stained, unstained=unstained, statistic="mean")
    cytopy.compensate(demo, spill, key_added="derived", inplace=True)
    cytopy.compensate(demo, key_added="from_file", inplace=True)
    j = [cytopy.find_channel_name(demo, c) for c in FLUOR]
    scale = np.ptp(demo.layers["from_file"][:, j])
    assert (
        np.abs(demo.layers["derived"][:, j] - demo.layers["from_file"][:, j]).max() < 0.005 * scale
    )


# --------------------------------------------------------------------------
# writing one out, and bringing someone else's back
# --------------------------------------------------------------------------
def test_write_spillover_roundtrips_through_pandas(demo, tmp_path):
    """No reader of our own: a CSV of ours is a CSV pandas already reads."""
    path = cytopy.write_spillover(demo.uns["spillover"], tmp_path / "spill.csv")
    back = pd.read_csv(path, index_col=0)
    assert list(back.columns) == ["FITC-A", "PE-A", "APC-A"]
    assert list(back.index) == list(back.columns)
    assert np.allclose(back.to_numpy(), demo.uns["spillover"].to_numpy())


def test_a_matrix_read_by_hand_can_be_applied(demo, tmp_path, true_spillover):
    """The documented route for a foreign CSV: read it yourself, pass the frame."""
    names = ["FITC-A", "PE-A", "APC-A"]
    pd.DataFrame(true_spillover, index=names, columns=names).to_csv(tmp_path / "s.csv")

    spill = pd.read_csv(tmp_path / "s.csv", index_col=0)
    out = cytopy.compensate(demo, spill)
    assert "comp" in out.layers
    assert out.uns["cytopy"]["spillover_source"] == "argument"


def test_compensate_no_longer_takes_a_path(demo, tmp_path):
    """A path is not a matrix; nothing here guesses how to read one."""
    (tmp_path / "s.csv").write_text("d,A,B\nA,1,0.12\nB,0.05,1\n")
    with pytest.raises((TypeError, ValueError)):
        cytopy.compensate(demo, str(tmp_path / "s.csv"))


# --------------------------------------------------------------------------
# applying it
# --------------------------------------------------------------------------
def test_compensate_records_where_the_matrix_came_from(demo):
    cytopy.compensate(demo, inplace=True)
    assert demo.uns["cytopy"]["spillover_source"] == "uns"
    cytopy.compensate(demo, demo.uns["spillover"], inplace=True)
    assert demo.uns["cytopy"]["spillover_source"] == "argument"


def test_compensate_aligns_rows_to_columns(demo):
    """Rows in a different order than the columns must not silently transpose the fix."""
    spill = _matrix([[1.0, 0.2, 0.0], [0.05, 1.0, 0.1], [0.0, 0.03, 1.0]])
    cytopy.compensate(demo, spill, key_added="ordered", inplace=True)
    cytopy.compensate(demo, spill.iloc[[2, 0, 1]], key_added="scrambled", inplace=True)
    assert np.allclose(demo.layers["ordered"], demo.layers["scrambled"])


def test_compensate_rejects_a_matrix_it_cannot_use(demo):
    with pytest.raises(ValueError, match="square"):
        cytopy.compensate(demo, pd.DataFrame(np.ones((2, 3))))
    with pytest.raises(ValueError, match="do not name the same detectors"):
        cytopy.compensate(demo, _matrix(np.eye(3)).rename(columns={FLUOR[0]: "FSC-A"}))
    with pytest.raises(ValueError, match="singular"):
        cytopy.compensate(demo, _matrix(np.ones((3, 3))))
    with pytest.raises(ValueError, match="fluorescence channels"):
        cytopy.compensate(demo, np.eye(2), inplace=True)
    demo.uns.pop("spillover")
    with pytest.raises(ValueError, match="no spillover matrix given"):
        cytopy.compensate(demo, inplace=True)


def test_cli_applies_the_files_own_matrix(demo_path, monkeypatch):
    """--compensate is the only matrix the CLI can reach; the rest is Python."""
    from cytopy import __main__

    seen = {}
    monkeypatch.setattr(
        "cytopy.open_napari",
        lambda adata, layer="raw", **kw: seen.update(adata=adata, layer=layer, **kw),
    )

    assert __main__.main([str(demo_path)]) == 0
    assert seen["layer"] == "raw"

    assert __main__.main([str(demo_path), "--compensate"]) == 0
    assert seen["layer"] == "comp"
    assert seen["adata"].uns["cytopy"]["spillover_source"] == "uns"


# --------------------------------------------------------------------------
# gating the controls by hand
# --------------------------------------------------------------------------
def test_a_matrix_built_from_hand_drawn_gates(controls, true_spillover, make_napari_viewer):
    """The whole route: read, gate each control, compute from those gates."""
    stained, unstained = controls

    from cytopy.viewer import CytoViewer

    for key, control in stained.items():
        cytopy.asinh_transform(control, 150.0, layer="X", inplace=True)
        cv = CytoViewer(control, layer="asinh", x=key, viewer=make_napari_viewer())
        cv.set_plot(kind="histogram", x=key)
        # drag an interval over the positive peak
        box = cv.to_canvas(np.array([2.5, 2.5, 9.0, 9.0]), np.array([0, 1, 1, 0]))
        cv.gates.add_rectangles([box])
        cv.apply_gate("positive")

    # each control keeps its own gate, because each is its own object
    assert all("positive" in c.obs for c in stained.values())
    assert all(int(c.obs["positive"].sum()) > 1000 for c in stained.values())

    spill = cytopy.compute_spillover_matrix(
        stained, unstained=unstained, positive_gate="positive", statistic="mean"
    )
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.005)


def test_a_control_without_a_gate_falls_back_to_the_split(controls, true_spillover):
    """One gated by hand, the rest automatic, in the same call."""
    stained, unstained = controls
    first = next(iter(stained))
    control = stained[first]
    column = np.asarray(control.X[:, cytopy.find_channel_name(control, first)])
    control.obs["positive"] = column > 1000.0

    spill = cytopy.compute_spillover_matrix(
        stained, unstained=unstained, positive_gate="positive", statistic="mean"
    )
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)


# --------------------------------------------------------------------------
# trying a matrix out on the controls
# --------------------------------------------------------------------------
def test_compensating_controls_is_just_compensate_in_a_loop(controls, true_spillover):
    """No wrapper: the same function that compensates a sample compensates a control."""
    stained, _ = controls
    spill = _matrix(true_spillover, list(stained))

    for adata in stained.values():
        cytopy.compensate(adata, spill, inplace=True)

    for adata in stained.values():
        assert "comp" in adata.layers
        assert not np.allclose(adata.layers["comp"], adata.X)
    assert not hasattr(cytopy, "compensate_controls")


def test_a_hand_edited_matrix_round_trips_through_compensate(controls, demo):
    stained, unstained = controls
    spill = cytopy.compute_spillover_matrix(stained, unstained=unstained, statistic="mean")
    spill.loc["CD3 (FITC-A)", "CD19 (PE-A)"] = 0.15  # tweak one coefficient by hand

    cytopy.compensate(demo, key_added="from_file", inplace=True)
    cytopy.compensate(demo, spill, key_added="tweaked", inplace=True)
    assert not np.allclose(demo.layers["tweaked"], demo.layers["from_file"])
    assert demo.uns["cytopy"]["spillover_source"] == "argument"


def test_the_separation_guard_is_off_by_default(controls):
    """It cannot tell a dim control from an empty tube, so it must not be a default.

    On real single-stain controls a genuine but dim tube scores about 3.5 in
    negative-MAD units while an unstained tube scores 6.7, so any cutoff that
    passes the real one passes the empty one too. The synthetic controls here
    are cleaner than that, which is exactly how the measure looked trustworthy
    until it met real data.
    """
    stained, unstained = controls
    # an unstained tube sails through the automatic split ...
    spill = cytopy.compute_spillover_matrix({"CD3": unstained})
    assert spill.shape == (1, 1)
    # ... so the check that does the real work is this one
    control = stained[FLUOR[0]].copy()
    column = np.asarray(control.X[:, cytopy.find_channel_name(control, FLUOR[0])])
    control.obs["inverted"] = column < np.median(column)
    with pytest.raises(ValueError, match="not brighter than its negatives"):
        cytopy.compute_spillover_matrix({FLUOR[0]: control}, positive_gate="inverted")


# --------------------------------------------------------------------------
# gating the controls in two passes: cells, then positives
# --------------------------------------------------------------------------
def test_gating_controls_is_just_open_napari_in_a_loop(controls):
    """No wrapper: the same function that opens a sample opens a control."""
    stained, _ = controls
    assert not hasattr(cytopy, "gate_controls")
    assert not hasattr(cytopy, "gate")  # nor a separate gating entry point

    control = next(iter(stained.values()))
    cytopy.asinh_transform(control, 150.0, layer="X", inplace=True)
    assert "asinh" in control.layers  # ready to open; the window itself is
    # covered in tests/test_viewer.py, which has napari's fixture to hand.


def test_subset_controls_keeps_only_the_gated_events(controls):
    stained, _ = controls
    for control in stained.values():
        fsc = np.asarray(control.X[:, cytopy.find_channel_name(control, "FSC-A")], dtype=float)
        control.obs["cells"] = fsc > np.median(fsc)

    gated = cytopy.subset_controls(stained, "cells")
    for key, control in gated.items():
        assert control.n_obs == int(stained[key].obs["cells"].sum())
        assert control.n_obs < stained[key].n_obs
        assert control.obs["cells"].all()
    assert gated is not stained


def test_subset_controls_says_when_a_control_was_never_gated(controls):
    stained, _ = controls
    with pytest.raises(KeyError, match="has no gate 'cells'"):
        cytopy.subset_controls(stained, "cells")

    for control in stained.values():
        control.obs["cells"] = np.zeros(control.n_obs, dtype=bool)
    with pytest.raises(ValueError, match="holds no events"):
        cytopy.subset_controls(stained, "cells")


def test_a_matrix_from_scatter_gated_controls(controls, true_spillover):
    """The whole two-pass route: keep the cells, then find the positives."""
    stained, unstained = controls
    everything = {**stained, "unstained": unstained}
    for control in everything.values():
        fsc = np.asarray(control.X[:, cytopy.find_channel_name(control, "FSC-A")], dtype=float)
        control.obs["cells"] = fsc > np.quantile(fsc, 0.1)

    gated = cytopy.subset_controls(everything, "cells")
    comp_adatas = {ch: gated[ch] for ch in stained}

    spill = cytopy.compute_spillover_matrix(
        comp_adatas, unstained=gated["unstained"], statistic="mean"
    )
    assert np.allclose(spill.to_numpy(), true_spillover, atol=0.01)
    # and the controls really were cut down
    assert all(comp_adatas[ch].n_obs < stained[ch].n_obs for ch in stained)
