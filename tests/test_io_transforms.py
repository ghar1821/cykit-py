import numpy as np
import pytest

import cykit


def test_read_fcs(demo_path):
    a = cykit.read_fcs(demo_path)
    assert a.shape == (60000, 6)
    assert list(a.var["channel"]) == ["FSC-A", "SSC-A", "FITC-A", "PE-A", "APC-A", "Time"]
    assert a.var.loc["CD3 (FITC-A)", "marker"] == "CD3"
    assert list(a.var["kind"]) == ["scatter", "scatter", "fluor", "fluor", "fluor", "time"]
    assert a.uns["spillover"].shape == (3, 3)
    assert (a.obs["sample"] == "demo").all()


def test_raw_layer_is_an_untouched_copy(demo):
    assert np.array_equal(demo.layers["raw"], demo.X)
    assert demo.layers["raw"] is not demo.X
    before = demo.layers["raw"].copy()
    cykit.asinh_transform(demo, 150.0, layer="X", key_added=None, inplace=True)  # overwrites .X
    assert np.array_equal(demo.layers["raw"], before)
    assert not np.array_equal(demo.X[:, 2], before[:, 2])


def test_store_raw_can_be_turned_off(demo_path):
    a = cykit.read_fcs(demo_path, store_raw=False)
    assert "raw" not in [k for k in a.layers if k is not None]


def test_gain_is_opt_in(demo_path):
    """$PnG is not applied unless asked for, so the two agree on a gain-1 file."""
    default = cykit.read_fcs(demo_path)
    gained = cykit.read_fcs(demo_path, apply_gain=True)
    assert (default.var["png"] == 1.0).all()
    assert np.array_equal(default.X, gained.X)


def test_fluor_channels_excludes_scatter_and_time(demo):
    assert cykit.get_fluor_channels(demo) == ["CD3 (FITC-A)", "CD19 (PE-A)", "CD8 (APC-A)"]


def test_channel_index_accepts_marker_and_channel(demo):
    assert cykit.find_channel_name(demo, "CD3") == 2
    assert cykit.find_channel_name(demo, "FITC-A") == 2
    assert cykit.find_channel_name(demo, "CD3 (FITC-A)") == 2
    with pytest.raises(KeyError):
        cykit.find_channel_name(demo, "CD4")


def test_compensate_recovers_the_uncontaminated_signal(demo):
    fluor = [2, 3, 4]
    truth = demo.X[:, fluor].astype(np.float64).copy()
    spill = np.array([[1.0, 0.2, 0.0], [0.05, 1.0, 0.1], [0.0, 0.03, 1.0]])
    demo.X[:, fluor] = (truth @ spill).astype(demo.X.dtype)

    cykit.compensate(
        demo,
        demo.uns["spillover"].__class__(
            spill, index=cykit.get_fluor_channels(demo), columns=cykit.get_fluor_channels(demo)
        ),
        layer="X",  # the contamination above was written to .X, not to layers["raw"]
        inplace=True,
    )
    assert np.allclose(demo.layers["comp"][:, fluor], truth, rtol=1e-3, atol=1e-2)
    # Channels outside the matrix pass straight through.
    assert np.allclose(demo.layers["comp"][:, 0], demo.X[:, 0])


def test_compensate_uses_the_spillover_from_the_file(demo):
    cykit.compensate(demo, inplace=True)
    assert "comp" in demo.layers
    assert demo.uns["cykit"]["compensated_layer"] == "comp"


def test_asinh_transform_scalar_cofactor(demo):
    cykit.asinh_transform(demo, 150.0, layer="X", inplace=True)
    out = demo.layers["asinh"]
    assert np.allclose(out[:, 2], np.arcsinh(demo.X[:, 2] / 150.0), atol=1e-5)
    # scatter and time are copied through untransformed
    assert np.allclose(out[:, 0], demo.X[:, 0])
    assert np.isnan(demo.var["cofactor"].iloc[0])
    assert demo.var["cofactor"].iloc[2] == 150.0


def test_asinh_transform_per_channel_cofactor(demo):
    cykit.asinh_transform(demo, {"CD3": 10.0, "default": 500.0}, layer="X", inplace=True)
    assert demo.var["cofactor"].iloc[2] == 10.0
    assert demo.var["cofactor"].iloc[3] == 500.0


def test_asinh_transform_chains_off_a_layer(demo):
    cykit.compensate(demo, inplace=True)
    cykit.asinh_transform(demo, 150.0, layer="comp", inplace=True)
    assert np.allclose(
        demo.layers["asinh"][:, 2], np.arcsinh(demo.layers["comp"][:, 2] / 150.0), atol=1e-5
    )


def test_asinh_in_place_overwrites_X(demo):
    before = demo.X.copy()
    cykit.asinh_transform(demo, 150.0, layer="X", key_added=None, inplace=True)
    assert not np.allclose(demo.X[:, 2], before[:, 2])
    assert [k for k in demo.layers if k is not None] == ["raw"]


def test_logicle_transform_maps_into_unit_interval(demo):
    cykit.logicle_transform(demo, layer="X", inplace=True)
    out = demo.layers["logicle"][:, 2]
    assert out.min() >= -0.21 and out.max() <= 1.21
    assert demo.uns["cykit"]["logicle_params"]["CD3 (FITC-A)"]["M"] == 4.5


def test_subsample(demo):
    small = cykit.subsample(demo, 1000)
    assert small.n_obs == 1000
    assert cykit.subsample(demo, 10**9).n_obs == demo.n_obs


def test_h5ad_roundtrip(demo, tmp_path):
    cykit.asinh_transform(demo, 150.0, layer="X", inplace=True)
    demo.uns.pop("spillover")
    p = tmp_path / "demo.h5ad"
    demo.write_h5ad(p)
    import anndata

    back = anndata.read_h5ad(p)
    assert np.allclose(back.layers["asinh"], demo.layers["asinh"])


def test_transforming_twice_keeps_both_layers_labelled(demo):
    """Raw and compensated, say. One slot meant the first lost its raw-unit ticks."""
    cykit.compensate(demo, inplace=True)
    cykit.logicle_transform(demo, layer="raw", key_added="logicle", inplace=True)
    cykit.logicle_transform(demo, layer="comp", key_added="comp_logicle", inplace=True)

    info = demo.uns["cykit"]
    assert set(info["logicle_layers"]) == {"logicle", "comp_logicle"}
    for layer in ("logicle", "comp_logicle"):
        assert "CD3 (FITC-A)" in info["logicle_layers"][layer]

    cykit.asinh_transform(demo, 150.0, layer="X", key_added="a1", inplace=True)
    cykit.asinh_transform(demo, 5.0, layer="X", key_added="a2", inplace=True)
    assert info["asinh_layers"]["a1"]["CD3 (FITC-A)"] == 150.0
    assert info["asinh_layers"]["a2"]["CD3 (FITC-A)"] == 5.0


def test_both_layers_get_raw_unit_ticks(demo):
    from cykit.plotting import axis_scale
    from cykit.scales import PretransformedScale

    cykit.compensate(demo, inplace=True)
    cykit.logicle_transform(demo, layer="raw", key_added="logicle", inplace=True)
    cykit.logicle_transform(demo, layer="comp", key_added="comp_logicle", inplace=True)
    for layer in ("logicle", "comp_logicle"):
        assert isinstance(axis_scale(demo, "CD3", layer), PretransformedScale)


def test_split_samples_is_the_other_end_of_concat(demo, demo_path):
    other = cykit.read_fcs(demo_path, sample_id="run2")
    pooled = cykit.concat_samples([demo, other])
    parts = cykit.split_samples(pooled)

    assert list(parts) == ["demo", "run2"]
    assert all(p.n_obs == 60_000 for p in parts.values())
    assert np.array_equal(parts["demo"].X, demo.X)
    with pytest.raises(KeyError, match="nothing to split on"):
        cykit.split_samples(demo, key="absent")


def test_slash_in_channel_name_survives_h5ad_roundtrip(tmp_path):
    """Imaging panels name parameters things like ``Delta CoM (SSC/FSC)``.

    HDF5 reads ``/`` as a path separator, and cykit keys ``uns`` entries by
    channel name, so an unsanitised name silently nests those entries under a
    group nothing reads back.
    """
    import flowio

    rng = np.random.default_rng(0)
    channels = ["FSC-A", "Delta CoM (SSC (Imaging)/FSC)", "PE-A"]
    events = rng.lognormal(6.0, 1.0, size=(500, len(channels))).astype(np.float32)
    path = tmp_path / "imaging.fcs"
    with open(path, "wb") as fh:
        flowio.create_fcs(fh, events.flatten().tolist(), channels, channels)

    a = cykit.read_fcs(path)
    # The index is safe to write, the original is still on hand to read.
    assert "Delta CoM (SSC (Imaging)_FSC)" in a.var_names
    assert not any("/" in n for n in a.var_names)
    assert a.var["channel"].tolist() == channels

    # Looking the channel up by the name the file used keeps working.
    from cykit.transforms import find_channel_name

    assert find_channel_name(a, "Delta CoM (SSC (Imaging)/FSC)") == 1

    cykit.logicle_transform(a, layer="raw", key_added="logicle", inplace=True)
    assert not any("/" in k for k in a.uns["cykit"]["logicle_params"])

    out = tmp_path / "imaging.h5ad"
    a.write_h5ad(out)

    import anndata as ad

    back = ad.read_h5ad(out)
    assert list(back.var_names) == list(a.var_names)
    assert set(back.uns["cykit"]["logicle_params"]) == set(a.uns["cykit"]["logicle_params"])


# --------------------------------------------------------------------------
# $SPILLOVER: read as written, checked, never silently reinterpreted
# --------------------------------------------------------------------------
def _refcs(path, tmp_path, keyword, name="respill.fcs"):
    """Rewrite the demo file with a different ``$SPILLOVER`` payload."""
    import flowio

    import cykit

    a = cykit.read_fcs(path)
    out = tmp_path / name
    with open(out, "wb") as fh:
        flowio.create_fcs(
            fh,
            np.asarray(a.layers["raw"]).flatten().tolist(),
            list(a.var["channel"]),
            list(a.var["marker"]),
            {"$SPILLOVER": keyword} if keyword is not None else {},
        )
    return out


def test_spillover_is_read_as_written(demo_path):
    """Detector names come back exactly as the file wrote them, unrenamed."""
    a = cykit.read_fcs(demo_path)
    spill = a.uns["spillover"]
    assert list(spill.index) == ["FITC-A", "PE-A", "APC-A"]
    assert list(spill.columns) == list(spill.index)
    assert np.allclose(np.diag(spill.to_numpy()), 1.0)


def test_spillover_percentages_are_converted_only_when_asked(demo_path, tmp_path):
    """A percentage matrix is nothing but a diagonal of 100 until you say so."""
    payload = "2,FITC-A,PE-A,100,12,5,100"
    path = _refcs(demo_path, tmp_path, payload)

    with pytest.warns(UserWarning, match="not 1"):
        plain = cykit.read_fcs(path)
    assert np.allclose(np.diag(plain.uns["spillover"].to_numpy()), 100.0)

    converted = cykit.read_fcs(path, convert_spillover=True)
    assert np.allclose(np.diag(converted.uns["spillover"].to_numpy()), 1.0)
    assert converted.uns["spillover"].loc["FITC-A", "PE-A"] == pytest.approx(0.12)


def test_a_compensation_matrix_is_stored_not_inverted(demo_path, tmp_path):
    """Inverting is `compensate`'s job; reading keeps what the file holds."""
    payload = "2,FITC-A,PE-A,1.0,-0.12,-0.05,1.0"
    path = _refcs(demo_path, tmp_path, payload)

    a = cykit.read_fcs(path)
    # negative off-diagonals are the giveaway, and they survive untouched
    assert a.uns["spillover"].loc["FITC-A", "PE-A"] == pytest.approx(-0.12)


def test_spillover_names_are_checked_against_the_channels(demo_path, tmp_path):
    path = _refcs(demo_path, tmp_path, "2,Comp-FITC-A,PE-A :: CD19,1,0.12,0.05,1")
    with pytest.warns(UserWarning, match="not channels of this file"):
        a = cykit.read_fcs(path)
    # stored as written -- the warning is the fix, not a rename
    assert list(a.uns["spillover"].index) == ["Comp-FITC-A", "PE-A :: CD19"]


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-count,FITC-A,1",
        "3,FITC-A,PE-A,1,0,0,1",  # count does not match the field count
        "2,FITC-A,PE-A,1,x,0,1",  # non-numeric
    ],
)
def test_a_malformed_spillover_is_dropped_with_a_warning(demo_path, tmp_path, payload):
    """The events are still worth having; the matrix is not."""
    path = _refcs(demo_path, tmp_path, payload)
    with pytest.warns(UserWarning):
        a = cykit.read_fcs(path)
    assert "spillover" not in a.uns
    assert a.n_obs == 60_000
