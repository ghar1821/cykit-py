"""Invariants pinned before the consolidation refactor.

Each test here exists because two copies of some logic are about to be merged
into one. They assert that the copies agree *now*, so that if they have
already drifted the failure lands before the merge rather than after it, and
so that the merge itself cannot change behaviour unnoticed.
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------
# axis scales: plotting.axis_scale vs CytoViewer._axis_scale
#
# The two are ~90% identical and are about to become one function. They
# currently disagree about how "no layer" is spelled -- plotting takes None,
# the viewer takes "X" -- so the comparison maps between them.
# --------------------------------------------------------------------------
def _scale_fingerprint(scale, lo: float, hi: float):
    """What a scale actually does to an axis: its type and its ticks."""
    ticks = scale.ticks(lo, hi)
    return (
        type(scale).__name__,
        np.asarray(ticks.major, dtype=float).round(9).tolist(),
        list(ticks.labels),
        np.asarray(ticks.minor, dtype=float).round(9).tolist(),
    )


def test_viewer_and_plotting_agree_on_axis_scale(demo, make_napari_viewer):
    """The window and the static figures must label an axis identically."""
    pytest.importorskip("napari")
    import cytopy
    from cytopy.plotting import axis_scale
    from cytopy.viewer import CytoViewer

    x, y = "CD3 (FITC-A)", "CD19 (PE-A)"
    cytopy.asinh_transform(
        demo, 150.0, layer="X", inplace=True
    )  # -> layers["asinh"], records cofactors
    cytopy.logicle_transform(demo, layer="X", inplace=True)  # -> layers["logicle"], records params
    demo.layers["untouched"] = demo.X.copy()  # a layer with no recorded transform

    cv = CytoViewer(demo, layer="asinh", x=x, y=y, bins=128, viewer=make_napari_viewer())
    assert cv.w_ticks.value != "transformed"  # otherwise the viewer short-circuits

    for layer in ("asinh", "logicle", "untouched", "X"):
        for channel in (x, y):
            plotting_layer = None if layer == "X" else layer
            want = axis_scale(demo, channel, plotting_layer)
            got = cv._axis_scale(channel, layer)
            where = f"layer={layer!r} channel={channel!r}"
            assert type(got) is type(want), where
            assert _scale_fingerprint(got, 0.0, 5.0) == _scale_fingerprint(want, 0.0, 5.0), where


def test_viewer_transformed_ticks_override_is_viewer_only(demo, make_napari_viewer):
    """`axis ticks: transformed` is the viewer's own guard, not part of the rule.

    If this guard ever migrates into the shared function, every static
    figure silently loses its raw-unit ticks.
    """
    pytest.importorskip("napari")
    import cytopy
    from cytopy.plotting import axis_scale
    from cytopy.scales import LinearScale
    from cytopy.viewer import CytoViewer

    x = "CD3 (FITC-A)"
    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    cv = CytoViewer(
        demo, layer="asinh", x=x, y="CD19 (PE-A)", bins=128, viewer=make_napari_viewer()
    )

    cv.w_ticks.value = "transformed"
    assert isinstance(cv._axis_scale(x, "asinh"), LinearScale)
    # The shared rule is unaffected by the widget.
    assert not isinstance(axis_scale(demo, x, "asinh"), LinearScale)


# --------------------------------------------------------------------------
# subsampling: one draw is about to be shared by four callers
#
# Any change to the rng call order changes *which* events get kept, and so
# which events get plotted. These literals are the tripwire.
# --------------------------------------------------------------------------
def test_subsample_is_reproducible(demo):
    import cytopy

    out = cytopy.subsample(demo, 100, seed=0)
    assert out.n_obs == 100
    names = out.obs_names.tolist()
    assert names[:5] == [
        "demo_164",
        "demo_320",
        "demo_496",
        "demo_990",
        "demo_1324",
    ]
    assert names[-3:] == ["demo_58162", "demo_58806", "demo_59786"]
    # Same seed, same draw; a different seed, a different draw.
    assert cytopy.subsample(demo, 100, seed=0).obs_names.tolist() == names
    assert cytopy.subsample(demo, 100, seed=1).obs_names.tolist() != names


def test_subsample_per_sample_is_reproducible(demo):
    """`per_sample` draws within each sample, so the rng is consumed per group."""
    import cytopy

    two = cytopy.concat_samples([demo.copy(), demo.copy()])
    two.obs["sample"] = ["a"] * demo.n_obs + ["b"] * demo.n_obs
    out = cytopy.subsample(two, 50, per_sample=True, seed=0)
    assert out.n_obs == 100
    assert out.obs["sample"].value_counts().to_dict() == {"a": 50, "b": 50}
    assert cytopy.subsample(two, 50, per_sample=True, seed=0).obs_names.tolist() == (
        out.obs_names.tolist()
    )


# --------------------------------------------------------------------------
# gating across samples
#
# Already correct, but untested: add_gate's `within` leaves events outside the
# drawn scope with whatever they had, and a fresh gate starts at all-False.
# Phase 9 folds sample_scope into selection_mask, so pin both halves first.
# --------------------------------------------------------------------------
@pytest.fixture
def two_samples(demo):
    import cytopy

    a, b = demo.copy(), demo.copy()
    both = cytopy.concat_samples([a, b])
    both.obs["sample"] = ["a"] * a.n_obs + ["b"] * b.n_obs
    return both


def test_gate_on_one_sample_leaves_other_samples_false(two_samples):
    """A gate drawn on sample A must not claim events in sample B."""
    from cytopy.gating import add_gate

    labels = two_samples.obs["sample"].astype(str).to_numpy()
    in_a = labels == "a"

    # Everything passes the shape, but only sample A was on screen.
    everything = np.ones(two_samples.n_obs, dtype=bool)
    stored = add_gate(two_samples, "g", everything, within=in_a)

    assert stored[in_a].all()
    assert not stored[~in_a].any()
    assert two_samples.obs["g"].to_numpy(dtype=bool).tolist() == stored.tolist()
    assert two_samples.uns["cytopy"]["gates"]["g"]["n"] == int(in_a.sum())


def test_regating_a_second_sample_keeps_the_first(two_samples):
    """The same gate name drawn on B afterwards adds B and keeps A."""
    from cytopy.gating import add_gate

    labels = two_samples.obs["sample"].astype(str).to_numpy()
    in_a, in_b = labels == "a", labels == "b"
    everything = np.ones(two_samples.n_obs, dtype=bool)

    add_gate(two_samples, "g", everything, within=in_a)
    stored = add_gate(two_samples, "g", everything, within=in_b)

    assert stored.all(), "A should have been kept while B was added"
    assert two_samples.uns["cytopy"]["gates"]["g"]["n"] == two_samples.n_obs


def test_gate_without_within_rewrites_the_whole_column(two_samples):
    """No `within` means the gate owns the column outright."""
    from cytopy.gating import add_gate

    labels = two_samples.obs["sample"].astype(str).to_numpy()
    in_a = labels == "a"

    add_gate(two_samples, "g", np.ones(two_samples.n_obs, dtype=bool), within=in_a)
    stored = add_gate(two_samples, "g", in_a)  # no `within`

    assert stored[in_a].all()
    assert not stored[~in_a].any()


# --------------------------------------------------------------------------
# gate records after a round trip through h5ad
#
# uns comes back holding numpy arrays where it was given lists, so `if
# record["vertices"]` raises rather than answering. Every reader now goes
# through gate_record, which is where that is dealt with.
# --------------------------------------------------------------------------
def test_gate_records_survive_h5ad_as_arrays(demo, tmp_path):
    import anndata

    import cytopy

    verts = [[500.0, -200.0], [500.0, 2000.0], [20000.0, 2000.0], [20000.0, -200.0]]
    x = np.asarray(demo.X[:, cytopy.find_channel_name(demo, "CD3")], dtype=float)
    y = np.asarray(demo.X[:, cytopy.find_channel_name(demo, "CD19")], dtype=float)
    mask = cytopy.polygon_mask(np.column_stack([x, y]), np.asarray(verts))
    cytopy.add_gate(
        demo,
        "lymphs",
        mask,
        meta={
            "x": "CD3 (FITC-A)",
            "y": "CD19 (PE-A)",
            "layer": "X",
            "vertices": [verts],
            "shape_types": ["polygon"],
        },
    )

    path = tmp_path / "run.h5ad"
    demo.write_h5ad(path)
    back = anndata.read_h5ad(path)

    # This is the shape that used to break the callers.
    assert isinstance(back.uns["cytopy"]["gates"]["lymphs"]["vertices"], np.ndarray)

    record = cytopy.gate_record(back, "lymphs")
    assert record.has_outline()
    assert record.layer == "X"
    assert record.shape_types == ["polygon"]
    assert len(record.vertices) == 1
    assert record.vertices[0].shape == (4, 2)

    # And the things that read a record all cope.
    assert cytopy.gate_mask(back, "lymphs").sum() == mask.sum()
    cytopy.plot_gate(back, "lymphs")
    cytopy.gating_pdf(back, tmp_path / "gates.pdf")


def test_gate_order_is_parents_before_children(demo):
    import cytopy

    everything = np.ones(demo.n_obs, dtype=bool)
    cytopy.add_gate(demo, "cells", everything)
    cytopy.add_gate(demo, "singlets", everything, parent="cells")
    cytopy.add_gate(demo, "live", everything, parent="singlets")
    cytopy.add_gate(demo, "other", everything)

    order = cytopy.gate_order(demo)
    assert order.index("cells") < order.index("singlets") < order.index("live")
    assert set(order) == {"cells", "singlets", "live", "other"}


# --------------------------------------------------------------------------
# inplace=False is the default, and it really does leave the original alone
#
# The flag was called `copy` and defaulted to modifying in place. A call whose
# result was never assigned used to change the data underneath the caller;
# now it cannot.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("call", "layer_added"),
    [
        (lambda a: __import__("cytopy").compensate(a), "comp"),
        (lambda a: __import__("cytopy").asinh_transform(a, 150.0, layer="raw"), "asinh"),
        (lambda a: __import__("cytopy").logicle_transform(a, layer="raw"), "logicle"),
    ],
    ids=["compensate", "asinh_transform", "logicle_transform"],
)
def test_the_default_returns_a_copy_and_leaves_the_original_alone(demo, call, layer_added):
    before = list(demo.layers)
    out = call(demo)

    assert out is not demo, "the default must not hand back the same object"
    assert layer_added in out.layers
    assert layer_added not in demo.layers, "the original was modified anyway"
    assert list(demo.layers) == before


@pytest.mark.parametrize(
    ("call", "layer_added"),
    [
        (lambda a: __import__("cytopy").compensate(a, inplace=True), "comp"),
        (
            lambda a: __import__("cytopy").asinh_transform(a, 150.0, layer="raw", inplace=True),
            "asinh",
        ),
        (lambda a: __import__("cytopy").logicle_transform(a, layer="raw", inplace=True), "logicle"),
    ],
    ids=["compensate", "asinh_transform", "logicle_transform"],
)
def test_inplace_true_modifies_and_hands_the_same_object_back(demo, call, layer_added):
    out = call(demo)

    assert out is demo, "inplace=True should hand back the object it modified"
    assert layer_added in demo.layers


def test_copy_is_gone_as_a_keyword(demo):
    """The old name must fail loudly rather than being silently ignored."""
    import cytopy

    for call in (
        lambda: cytopy.compensate(demo, copy=True),
        lambda: cytopy.asinh_transform(demo, 150.0, layer="raw", copy=True),
        lambda: cytopy.logicle_transform(demo, layer="raw", copy=True),
    ):
        with pytest.raises(TypeError, match="copy"):
            call()


# --------------------------------------------------------------------------
# axis limits: plotting.axis_limits vs CytoViewer._limits
# --------------------------------------------------------------------------
def test_viewer_and_plotting_agree_on_axis_limits(demo, make_napari_viewer):
    """Same rule, one function: an FSC/SSC panel is square in both."""
    pytest.importorskip("napari")
    import cytopy
    from cytopy.plotting import axis_limits
    from cytopy.viewer import CytoViewer

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    cv = CytoViewer(demo, layer="asinh", x="FSC-A", y="SSC-A", bins=64, viewer=make_napari_viewer())
    axes = cv.panel.axes
    values = np.asarray(demo[:, "FSC-A"].layers["asinh"]).ravel()
    assert (axes.x_lo, axes.x_hi) == axis_limits(demo, "FSC-A", "asinh", values)
    # Untransformed scatter: both axes end up on the shared instrument range.
    assert (axes.x_lo, axes.x_hi) == (axes.y_lo, axes.y_hi)
