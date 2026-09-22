import numpy as np
import pytest

import cytopy


def test_read_fcs(demo_path):
    a = cytopy.read_fcs(demo_path)
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
    cytopy.asinh_transform(demo, 150.0, layer="X", key_added=None, inplace=True)  # overwrites .X
    assert np.array_equal(demo.layers["raw"], before)
    assert not np.array_equal(demo.X[:, 2], before[:, 2])


def test_store_raw_can_be_turned_off(demo_path):
    a = cytopy.read_fcs(demo_path, store_raw=False)
    assert "raw" not in [k for k in a.layers if k is not None]


def test_gain_is_opt_in(demo_path):
    """$PnG is not applied unless asked for, so the two agree on a gain-1 file."""
    default = cytopy.read_fcs(demo_path)
    gained = cytopy.read_fcs(demo_path, apply_gain=True)
    assert (default.var["png"] == 1.0).all()
    assert np.array_equal(default.X, gained.X)


def test_fluor_channels_excludes_scatter_and_time(demo):
    assert cytopy.fluor_channels(demo) == ["CD3 (FITC-A)", "CD19 (PE-A)", "CD8 (APC-A)"]


def test_channel_index_accepts_marker_and_channel(demo):
    assert cytopy.channel_index(demo, "CD3") == 2
    assert cytopy.channel_index(demo, "FITC-A") == 2
    assert cytopy.channel_index(demo, "CD3 (FITC-A)") == 2
    with pytest.raises(KeyError):
        cytopy.channel_index(demo, "CD4")


def test_compensate_recovers_the_uncontaminated_signal(demo):
    fluor = [2, 3, 4]
    truth = demo.X[:, fluor].astype(np.float64).copy()
    spill = np.array([[1.0, 0.2, 0.0], [0.05, 1.0, 0.1], [0.0, 0.03, 1.0]])
    demo.X[:, fluor] = (truth @ spill).astype(demo.X.dtype)

    cytopy.compensate(
        demo,
        demo.uns["spillover"].__class__(
            spill, index=cytopy.fluor_channels(demo), columns=cytopy.fluor_channels(demo)
        ),
        layer="X",  # the contamination above was written to .X, not to layers["raw"]
        inplace=True,
    )
    assert np.allclose(demo.layers["comp"][:, fluor], truth, rtol=1e-3, atol=1e-2)
    # Channels outside the matrix pass straight through.
    assert np.allclose(demo.layers["comp"][:, 0], demo.X[:, 0])


def test_compensate_uses_the_spillover_from_the_file(demo):
    cytopy.compensate(demo, inplace=True)
    assert "comp" in demo.layers
    assert demo.uns["cytopy"]["compensated_layer"] == "comp"


def test_asinh_transform_scalar_cofactor(demo):
    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    out = demo.layers["asinh"]
    assert np.allclose(out[:, 2], np.arcsinh(demo.X[:, 2] / 150.0), atol=1e-5)
    # scatter and time are copied through untransformed
    assert np.allclose(out[:, 0], demo.X[:, 0])
    assert np.isnan(demo.var["cofactor"].iloc[0])
    assert demo.var["cofactor"].iloc[2] == 150.0


def test_asinh_transform_per_channel_cofactor(demo):
    cytopy.asinh_transform(demo, {"CD3": 10.0, "default": 500.0}, layer="X", inplace=True)
    assert demo.var["cofactor"].iloc[2] == 10.0
    assert demo.var["cofactor"].iloc[3] == 500.0


def test_asinh_transform_chains_off_a_layer(demo):
    cytopy.compensate(demo, inplace=True)
    cytopy.asinh_transform(demo, 150.0, layer="comp", inplace=True)
    assert np.allclose(
        demo.layers["asinh"][:, 2], np.arcsinh(demo.layers["comp"][:, 2] / 150.0), atol=1e-5
    )


def test_asinh_in_place_overwrites_X(demo):
    before = demo.X.copy()
    cytopy.asinh_transform(demo, 150.0, layer="X", key_added=None, inplace=True)
    assert not np.allclose(demo.X[:, 2], before[:, 2])
    assert [k for k in demo.layers if k is not None] == ["raw"]


def test_estimate_cofactors(demo, layer="X"):
    cof = cytopy.estimate_cofactors(demo, layer="X")
    assert set(cof) == set(cytopy.fluor_channels(demo))
    assert all(v > 1 for v in cof.values())


def test_logicle_transform_maps_into_unit_interval(demo):
    cytopy.logicle_transform(demo, layer="X", inplace=True)
    out = demo.layers["logicle"][:, 2]
    assert out.min() >= -0.21 and out.max() <= 1.21
    assert demo.uns["cytopy"]["logicle_params"]["CD3 (FITC-A)"]["M"] == 4.5


def test_subsample(demo):
    small = cytopy.subsample(demo, 1000)
    assert small.n_obs == 1000
    assert cytopy.subsample(demo, 10**9).n_obs == demo.n_obs


def test_h5ad_roundtrip(demo, tmp_path):
    cytopy.asinh_transform(demo, 150.0, layer="X", inplace=True)
    demo.uns.pop("spillover")
    p = tmp_path / "demo.h5ad"
    demo.write_h5ad(p)
    import anndata

    back = anndata.read_h5ad(p)
    assert np.allclose(back.layers["asinh"], demo.layers["asinh"])


def test_transforming_twice_keeps_both_layers_labelled(demo):
    """Raw and compensated, say. One slot meant the first lost its raw-unit ticks."""
    cytopy.compensate(demo, inplace=True)
    cytopy.logicle_transform(demo, layer="raw", key_added="logicle", inplace=True)
    cytopy.logicle_transform(demo, layer="comp", key_added="comp_logicle", inplace=True)

    info = demo.uns["cytopy"]
    assert set(info["logicle_layers"]) == {"logicle", "comp_logicle"}
    for layer in ("logicle", "comp_logicle"):
        assert "CD3 (FITC-A)" in info["logicle_layers"][layer]

    cytopy.asinh_transform(demo, 150.0, layer="X", key_added="a1", inplace=True)
    cytopy.asinh_transform(demo, 5.0, layer="X", key_added="a2", inplace=True)
    assert info["asinh_layers"]["a1"]["CD3 (FITC-A)"] == 150.0
    assert info["asinh_layers"]["a2"]["CD3 (FITC-A)"] == 5.0


def test_both_layers_get_raw_unit_ticks(demo):
    from cytopy.plotting import axis_scale
    from cytopy.scales import PretransformedScale

    cytopy.compensate(demo, inplace=True)
    cytopy.logicle_transform(demo, layer="raw", key_added="logicle", inplace=True)
    cytopy.logicle_transform(demo, layer="comp", key_added="comp_logicle", inplace=True)
    for layer in ("logicle", "comp_logicle"):
        assert isinstance(axis_scale(demo, "CD3", layer), PretransformedScale)


def test_split_samples_is_the_other_end_of_concat(demo, demo_path):
    other = cytopy.read_fcs(demo_path, sample_id="run2")
    pooled = cytopy.concat_samples([demo, other])
    parts = cytopy.split_samples(pooled)

    assert list(parts) == ["demo", "run2"]
    assert all(p.n_obs == 60_000 for p in parts.values())
    assert np.array_equal(parts["demo"].X, demo.X)
    with pytest.raises(KeyError, match="nothing to split on"):
        cytopy.split_samples(demo, key="absent")


def test_slash_in_channel_name_survives_h5ad_roundtrip(tmp_path):
    """Imaging panels name parameters things like ``Delta CoM (SSC/FSC)``.

    HDF5 reads ``/`` as a path separator, and cytopy keys ``uns`` entries by
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

    a = cytopy.read_fcs(path)
    # The index is safe to write, the original is still on hand to read.
    assert "Delta CoM (SSC (Imaging)_FSC)" in a.var_names
    assert not any("/" in n for n in a.var_names)
    assert a.var["channel"].tolist() == channels

    # Looking the channel up by the name the file used keeps working.
    from cytopy.transforms import channel_index

    assert channel_index(a, "Delta CoM (SSC (Imaging)/FSC)") == 1

    cytopy.logicle_transform(a, layer="raw", key_added="logicle", inplace=True)
    assert not any("/" in k for k in a.uns["cytopy"]["logicle_params"])

    out = tmp_path / "imaging.h5ad"
    a.write_h5ad(out)

    import anndata as ad

    back = ad.read_h5ad(out)
    assert list(back.var_names) == list(a.var_names)
    assert set(back.uns["cytopy"]["logicle_params"]) == set(a.uns["cytopy"]["logicle_params"])
