"""Static figures: biaxial plots, gate overlays, and the gating PDF."""

import numpy as np
import pytest

import cytopy


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib

    matplotlib.use("Agg", force=True)


# --------------------------------------------------------------------------
# biaxial plots
# --------------------------------------------------------------------------
def test_biaxial_is_a_density_by_default(demo):
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", layer="X")
    assert len(ax.images) == 1 and not ax.collections
    image = ax.images[0].get_array()
    assert image.shape == (512, 512) and image.max() > 0
    assert ax.get_xlabel() == "CD3 (FITC-A)" and ax.get_ylabel() == "CD19 (PE-A)"
    assert "60,000 events" in ax.get_title(loc="left")


def test_biaxial_colours_by_a_boolean_column(demo):
    demo.obs["big"] = np.asarray(demo.X[:, 0]) > np.median(demo.X[:, 0])
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", layer="X", color_by="big")
    assert len(ax.images) == 1 and len(ax.collections) == 1
    assert ax.images[0].get_cmap().name == "Greys"  # the highlight carries the colour
    assert ax.get_legend() is not None
    with pytest.raises(KeyError, match="to colour by"):
        cytopy.plot_biaxial(demo, "CD3", "CD19", layer="X", color_by="nope")


def test_biaxial_subsets(demo):
    demo.obs["half"] = np.arange(demo.n_obs) < demo.n_obs // 2
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", layer="X", subset="half")
    assert "30,000 events" in ax.get_title(loc="left")
    ax = cytopy.plot_biaxial(
        demo, "CD3", "CD19", layer="X", subset=np.zeros(demo.n_obs, dtype=bool)
    )
    assert "0 events" in ax.get_title(loc="left")
    with pytest.raises(ValueError, match="expected 60000"):
        cytopy.plot_biaxial(demo, "CD3", "CD19", layer="X", subset=np.ones(3, dtype=bool))


def test_biaxial_axes_read_in_raw_units(demo):
    """The viewer's rule: a layer that records its transform gets decade ticks."""
    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    plain = cytopy.plot_biaxial(demo, "CD3", "CD19", layer="X")
    labelled = cytopy.plot_biaxial(demo, "CD3", "CD19", layer="asinh")
    assert not any(
        "¹⁰" in t.get_text() or "10" in t.get_text() for t in plain.get_xticklabels()[:1]
    )
    assert any(t.get_text() == "0" for t in labelled.get_xticklabels())
    assert any("10" in t.get_text() for t in labelled.get_xticklabels())


def test_biaxial_cofactor_transforms_for_display_only(demo):
    before = np.asarray(demo.X).copy()
    ax = cytopy.plot_biaxial(demo, "CD3", "CD19", layer="X", cofactor=5.0)
    assert np.array_equal(np.asarray(demo.X), before)  # nothing written back
    assert "asinh" not in demo.layers
    assert ax.get_xlim()[1] < 20  # plotted on the compressed scale
    assert any("10" in t.get_text() for t in ax.get_xticklabels())


def test_biaxial_draws_rectangles(demo):
    ax = cytopy.plot_biaxial(
        demo, "CD3", "CD19", layer="X", rectangles=[(100.0, 500.0, 100.0, 500.0)]
    )
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == 1
    # The outline must not drag the axes with it.
    assert ax.get_xlim()[0] < 100.0


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def _draw_gate(adata, name="lymphs"):
    x = np.asarray(adata.X[:, cytopy.find_channel_name(adata, "CD3")], dtype=float)
    y = np.asarray(adata.X[:, cytopy.find_channel_name(adata, "CD19")], dtype=float)
    verts = [[500.0, -200.0], [500.0, 2000.0], [20000.0, 2000.0], [20000.0, -200.0]]
    mask = cytopy.polygon_mask(np.column_stack([x, y]), np.asarray(verts))
    cytopy.add_gate(
        adata,
        name,
        mask,
        meta={"x": "CD3 (FITC-A)", "y": "CD19 (PE-A)", "layer": "X", "vertices": [verts]},
    )
    return name


def test_plot_gate_redraws_where_it_was_drawn(demo):
    name = _draw_gate(demo)
    ax = cytopy.plot_gate(demo, name)
    assert ax.get_xlabel() == "CD3 (FITC-A)"
    assert any(ln.get_linestyle() == "--" for ln in ax.lines)
    assert len(ax.collections) == 1  # the gated events, picked out
    assert name in ax.get_title(loc="left")


def test_plot_gate_needs_a_recorded_gate(demo):
    with pytest.raises(KeyError, match="no gate"):
        cytopy.plot_gate(demo, "never drawn")
    cytopy.add_gate(demo, "bare", np.ones(demo.n_obs, dtype=bool))
    with pytest.raises(KeyError, match="did not record which channels"):
        cytopy.plot_gate(demo, "bare")


def test_a_gate_outline_is_skipped_on_the_wrong_axes(demo):
    name = _draw_gate(demo)
    ax = cytopy.plot_biaxial(demo, "CD3", "CD8", layer="X", gate=name)
    assert not [ln for ln in ax.lines if ln.get_linestyle() == "--"]


# --------------------------------------------------------------------------
# what survives a round trip
# --------------------------------------------------------------------------
def test_gates_survive_write_h5ad(demo, tmp_path):
    """A whole processed run has to save and come back -- gates and all."""
    import anndata

    name = _draw_gate(demo)

    path = tmp_path / "run.h5ad"
    demo.write_h5ad(path)
    back = anndata.read_h5ad(path)

    assert back.n_obs == demo.n_obs
    assert list(back.obs[name]) == list(demo.obs[name])
    record = back.uns["cytopy"]["gates"][name]
    assert record["x"] == "CD3 (FITC-A)" and record["y"] == "CD19 (PE-A)"
    # uns lists come back as arrays, hence the list() on both sides.
    assert np.asarray(record["vertices"]).shape == (1, 4, 2)

    # The restored object is enough to draw the gate from.
    ax = cytopy.plot_gate(back, name)
    assert any(ln.get_linestyle() == "--" for ln in ax.lines)


# --------------------------------------------------------------------------
# the gating hierarchy as a PDF
# --------------------------------------------------------------------------
def _hierarchy(adata):
    """Three nested gates, the last drawn on a different pair of channels."""
    import cytopy

    cytopy.asinh_transform(adata, 150.0, layer="X", inplace=True)
    values = np.asarray(adata.layers["asinh"])

    def gate(name, x, y, box, parent=None):
        xi, yi = cytopy.find_channel_name(adata, x), cytopy.find_channel_name(adata, y)
        mask = (
            (values[:, xi] > box[0])
            & (values[:, xi] < box[1])
            & (values[:, yi] > box[2])
            & (values[:, yi] < box[3])
        )
        cytopy.add_gate(
            adata,
            name,
            mask,
            parent=parent,
            meta={
                "x": str(adata.var_names[xi]),
                "y": str(adata.var_names[yi]),
                "layer": "asinh",
                "vertices": [
                    [[box[0], box[2]], [box[0], box[3]], [box[1], box[3]], [box[1], box[2]]]
                ],
                "shape_types": ["polygon"],
            },
        )

    gate("lymphocytes", "CD3", "CD19", (2.0, 9.0, -2.0, 9.0))
    gate("T cells", "CD3", "CD19", (3.0, 9.0, -2.0, 4.0), parent="lymphocytes")
    gate("CD8+", "CD8", "CD3", (3.0, 9.0, 3.0, 9.0), parent="T cells")
    return adata


def test_gating_pdf_writes_a_page_per_gate(demo, tmp_path):
    _hierarchy(demo)
    path = cytopy.gating_pdf(demo, tmp_path / "gating.pdf", ncols=3, per_page=3)
    assert path.exists() and path.stat().st_size > 10_000
    assert path.read_bytes().startswith(b"%PDF")
    # a contents page plus one page of three plots
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401

    assert path.read_bytes().count(b"/Page") >= 2


def test_the_hierarchy_comes_out_parents_first(demo):
    _hierarchy(demo)
    assert cytopy.gate_order(demo) == ["lymphocytes", "T cells", "CD8+"]


def test_each_gate_is_drawn_on_the_plane_it_was_drawn_in(demo):
    import matplotlib.pyplot as plt

    from cytopy.plotting import _draw_gate_page, plot_gate

    _hierarchy(demo)
    _, axes = plt.subplots(1, 3)
    for ax, name in zip(axes, ["lymphocytes", "T cells", "CD8+"]):
        _draw_gate_page(demo, demo, name, ax, plot_gate, {})

    # the last gate was drawn on a different pair, and follows it
    assert axes[0].get_xlabel() == "CD3 (FITC-A)"
    assert axes[2].get_xlabel() == "CD8 (APC-A)"
    assert axes[2].get_ylabel() == "CD3 (FITC-A)"
    # each title carries its lineage and its share of its own parent
    n = int(demo.obs["T cells"].sum())
    total = int(demo.obs["lymphocytes"].sum())
    title = axes[1].get_title(loc="left")
    assert "lymphocytes > T cells" in title
    assert f"{n:,} of {total:,}" in title and f"{100 * n / total:.1f}%" in title


def test_the_contents_page_lists_the_tree(demo):
    from cytopy.plotting import _hierarchy_page

    _hierarchy(demo)
    gates = demo.uns["cytopy"]["gates"]
    fig = _hierarchy_page(demo, gates, cytopy.gate_order(demo), "run 1")
    text = [t.get_text() for t in fig.axes[0].texts]
    assert "run 1" in text
    assert any("60,000 events" in t for t in text)

    cells = [c.get_text().get_text() for c in fig.axes[0].tables[0].get_celld().values()]
    assert any(c.strip() == "lymphocytes" for c in cells)
    assert any(c.startswith("    ") and c.strip() == "T cells" for c in cells)  # indented
    assert f"{int(demo.obs['CD8+'].sum()):,}" in cells


def test_gating_pdf_needs_something_to_draw(demo, tmp_path):
    with pytest.raises(ValueError, match="no gates with an outline"):
        cytopy.gating_pdf(demo, tmp_path / "empty.pdf")

    cytopy.add_gate(demo, "from a mask", np.ones(demo.n_obs, dtype=bool))
    with pytest.raises(ValueError, match="no gates with an outline"):
        cytopy.gating_pdf(demo, tmp_path / "empty.pdf")


# --------------------------------------------------------------------------
# axis_limits: an untransformed channel spans its detector, not this sample
# --------------------------------------------------------------------------
def test_untransformed_channels_span_the_instrument_range(demo):
    """FSC and SSC share a $PnR, so the scatter plot comes out square."""
    import numpy as np

    from cytopy.plotting import axis_limits

    x = axis_limits(demo, "FSC-A", "X", np.asarray(demo[:, "FSC-A"].X).ravel())
    y = axis_limits(demo, "SSC-A", "X", np.asarray(demo[:, "SSC-A"].X).ravel())
    assert x == y
    top = float(demo.var["pnr"].iloc[0])
    assert x[0] < 0.0 < top < x[1]  # the padded [0, $PnR]


def test_transformed_channels_still_span_the_data(demo):
    """The instrument range means nothing once a channel has been arcsinh'd."""
    import numpy as np

    import cytopy
    from cytopy.plotting import axis_limits

    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    values = np.asarray(demo[:, "CD3 (FITC-A)"].layers["asinh"]).ravel()
    _, hi = axis_limits(demo, "CD3 (FITC-A)", "asinh", values)
    assert hi < 20.0  # arcsinh units, nowhere near $PnR


def test_data_outside_the_instrument_range_is_not_clipped(demo):
    """Compensation pushes events negative; hiding them would be the worse bug."""
    import numpy as np

    from cytopy.plotting import axis_limits

    values = np.asarray(demo[:, "FSC-A"].X).ravel().copy()
    values[:1000] = -50_000.0
    lo, _ = axis_limits(demo, "FSC-A", "X", values)
    assert lo < -50_000.0


def test_an_anndata_without_pnr_falls_back_to_the_data(demo):
    """Nothing here requires the object to have come from an FCS file."""
    import numpy as np

    from cytopy.plotting import axis_limits
    from cytopy.scales import pad_range

    del demo.var["pnr"]
    values = np.asarray(demo[:, "FSC-A"].X).ravel()
    limits = axis_limits(demo, "FSC-A", "X", values)
    assert limits == pad_range(float(values.min()), float(values.max()))
