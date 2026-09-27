# cytopy

Cytometry analysis on [AnnData](https://anndata.readthedocs.io), gated
interactively in [napari](https://napari.org).

Read FCS files into an `AnnData` (events × channels), compensate and transform
them, then open a napari window that draws a biexponential density plot of any
two channels and turns polygons you draw into boolean gates in `adata.obs`.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[gui,dev]"
```

## Use

```python
import cytopy

adata = cytopy.read_fcs("sample.fcs")        # events x channels, + layers["raw"]
cytopy.compensate(adata, inplace=True)       # -> adata.layers["comp"]
cytopy.asinh_transform(adata, cofactor=150,  # -> adata.layers["asinh"]
                       layer="comp", inplace=True)

cytopy.open_napari(adata, "asinh", x="CD3", y="CD19")   # or pick them in the window
```

or from a terminal:

```bash
cytopy sample.fcs --compensate --asinh --cofactor 150 -x CD3 -y CD19
```

### Compensation

`compensate` applies a spillover matrix `S` as `X @ inv(S)`, writing
`layers["comp"]`. There are three ways to get `S`, and all of them end up as the
same square DataFrame indexed by detector, with `1` on the diagonal:

```python
import pandas as pd

cytopy.compensate(adata, inplace=True)                  # 1. the file's own $SPILLOVER

# 2. exported by other software: read the CSV yourself, pass the frame
cytopy.compensate(adata, pd.read_csv("matrix.csv", index_col=0), inplace=True)

# 3. single-stain controls: you say which file stains which detector
controls = {"CD3 (FITC-A)": cytopy.read_fcs("controls/FITC-A.fcs"),
            "CD19 (PE-A)": cytopy.read_fcs("controls/PE-A.fcs")}
unstained = cytopy.read_fcs("controls/Unstained.fcs")
spill = cytopy.compute_spillover_matrix(controls, unstained=unstained)
cytopy.compensate(adata, spill, inplace=True)
```

**From single-stain controls.** `compute_spillover_matrix` implements the
Bagwell and Adams matrix-inversion method. For the control stained with dye
*i*, it takes the difference between the positive and negative populations in
every detector *j*, then divides that row through by the difference in the
dye's own detector — so the coefficients do not depend on how bright the
control happened to be, and the diagonal is `1` by construction. Inverting the
assembled matrix is the N-parameter generalisation that lets every detector be
corrected at once.

The positive population is found by splitting the control's stained detector on
an arcsinh scale (Otsu), or you can gate it yourself:

```python
everything = {**controls, "unstained": unstained}

pooled = cytopy.open_napari(everything, "asinh")   # ONE window, every control in it

gated = cytopy.subset_controls(cytopy.split_samples(pooled), "singlets")
unstained = gated.pop("unstained")
spill = cytopy.compute_spillover_matrix(gated, unstained=unstained, positive_gate="positive")
```

There is no special function for controls — [`cytopy.open_napari`](#gating) is
the same one you use on a sample, and it takes the whole dict at once. The
controls are pooled into one object with a `sample` column, and the **samples**
list picks
which you are looking at. That matters because the gates are not all the same
kind:

| gate | samples selected |
| --- | --- |
| `cells` — `FSC-A` × `SSC-A` | **all**: scatter is scatter |
| `singlets` — `FSC-A` × `FSC-H`, parent `cells` | **all** |
| `positive` — the stained channel, parent `singlets` | **one at a time**: each tube stains a different channel |

**A gate only changes the samples you are looking at**, so drawing `positive` on
one tube and then the next under the same name keeps both. `split_samples`
takes them apart again afterwards.

**Gates already on the data are loaded** — outlined where you drew them, listed
under *edit gate*, available as parents. Gating is something you come back to.

The scatter pass matters more than it looks: every number in the matrix is a
median, and a median over cells *and* debris is a median of nothing in
particular. On a real six-colour panel, gating the controls down to cells took
the worst coefficient from 10.1% off to 4.7% off — before a single positive
gate was drawn.

Each control is its own AnnData, so the gate lives in *that* control's
`obs["positive"]` and the same name works for all of them. A control with no
gate falls back to the automatic split, so you can hand-gate only the awkward
ones. `thresholds=` takes a raw cutoff per control instead.

The gating is done on an arcsinh display — on a linear axis the whole negative
population lands in one bin — but **the matrix is computed on the raw values**,
because spillover is linear and `asinh(a) − asinh(b)` is not proportional to
`a − b`. Transforming the controls first inflates the coefficients
several-fold. The negative reference is the control's own negative events by
default — same beads, same autofluorescence — or pass
`unstained=` to use a universal negative instead. A control that does not split
into two populations is an error rather than a quietly wrong row.

**Adjusting one by hand.** The matrix is a DataFrame, so a coefficient is an
assignment:

```python
spill.loc["CD3 (FITC-A)", "CD19 (PE-A)"] = 0.13
```

To see whether it helped, compensate a control with the new matrix and look at
it: a single-stain control compensated correctly has its positive population
level with its negative one in every detector but its own. A population tipping
up is under-compensated, one sliding down is over.

```python
control = cytopy.compensate(controls["CD3 (FITC-A)"], spill)
cytopy.plot_biaxial(control, "CD3 (FITC-A)", "CD19 (PE-A)", layer="comp", cofactor=150)
```

**From a CSV.** There is no reader for this: `compensate` takes a DataFrame, and
`pd.read_csv(path, index_col=0)` is how you get one. Exporters disagree about
delimiters, whether there is a row-name column, whether values are fractions or
percentages, and whether names are decorated as `Comp-FITC-A` or
`FITC-A :: CD3` — a reader that guesses at all of that gets it wrong silently,
and a matrix read wrongly is not something anything downstream can detect. You
can see the file; pandas already has the arguments. Two things to check by
hand: the diagonal should be `1` (`100` means percentages, divide by 100), and
the row and column names have to match channels in your data. If the file holds
a **compensation** matrix — the inverse — invert it yourself with
`np.linalg.inv` before passing it. `write_spillover` writes one back out.

From the terminal:

```bash
cytopy sample.fcs --compensate --asinh          # the file's own $SPILLOVER
```

Any other matrix is a Python job, not a flag — reading a foreign CSV and saying
which control stains which detector are both yours to state.

### Biaxial plots

`plot_biaxial` is the static counterpart to the viewer — the same smoothed 2-D
histogram, so a saved figure matches what was on screen when the gate was
drawn. **Colour is by event density by default**: at a few million events a
per-point scatter is neither fast nor readable, and the structure is where
events pile up.

```python
cytopy.plot_biaxial(adata, "CD3", "CD19", layer="asinh")      # density
cytopy.plot_biaxial(adata, "CD3", "CD19", layer="asinh", color_by="lymphs")  # a gate over a density
cytopy.plot_biaxial(adata, "CD3", "CD19", layer="raw", cofactor=150)        # display-only arcsinh
cytopy.plot_gate(adata, "lymphocytes")                        # redraw a gate where it was drawn
```

Axes are labelled in raw units whenever the layer records the transform that
produced it, exactly as in the viewer. `cofactor=` compresses the axis for
display without writing a layer, which matters when the matrix is large.

### Large runs

Everything is sized for runs of millions of events. What that took, in case you
hit the same walls elsewhere:

* **Group by categorical codes, not strings.** `obs["sample"].astype(str)`
  builds a Python string per event; that one line was 0.8 s of the 1.4 s a
  per-sample plot used to take.
* **Bin by arithmetic.** `density_image` computes bin indices directly and
  counts with `bincount` rather than calling `np.histogram2d`. Identical
  output, 6x faster, and it runs on every widget change.
* **Never widen the matrix.** Intermediates are computed at the storage dtype
  and written a column at a time; a `float64` copy of everything was over 4 GB
  at 46 channels.
* **arcsinh is monotone, so percentiles do not need recomputing.** The cofactor
  window takes each channel's quantiles once, on the raw values, and transforms
  *those* rather than re-percentiling a million transformed values on every
  slider tick. It saves ~9 ms a tick, but the real point is that the axis stays
  pinned to fixed raw values instead of shifting under the population as you
  drag.
* **Transform in the storage dtype.** `np.arcsinh` on a million `float32` values
  is 3.3 ms; on `float64` it is 11.2 ms. On a slider that fires forty times a
  second, that is the difference between live and not.

### Nothing is modified unless you say so

`compensate`, `asinh_transform` and `logicle_transform` return a modified copy
by default and leave the object you passed alone. Pass `inplace=True` to build
the layers up on one object, which is what a pipeline usually wants:

```python
cytopy.compensate(adata, spill, inplace=True)              # adata gains layers["comp"]
comped = cytopy.compensate(adata, spill)                   # adata untouched
```

The default is the safe one on purpose: a call whose result you forget to
assign cannot quietly change the data underneath you.

### Transforms are always explicit

The viewer never transforms anything. It plots the layer you point it at,
exactly as stored, so the plot shows the result of the transform *you* ran.
What it does infer is how to **label** the axes: `asinh_transform` and
`logicle_transform` record their parameters on the AnnData, and the viewer uses
them to put the ticks at decades of the original units — which is what turns an
arcsinh layer into a plot that reads as biexponential.

So `layers["asinh"]` gets an axis marked `-10² 0 10² 10³ 10⁴`, while plotting
raw `X` gets plain linear ticks.

**axis ticks** chooses which units you read, in both windows:

| | what the ticks say |
|---|---|
| `untransformed` (default) | the channel's original units — `-10² 0 10² 10³ 10⁴` — falling back to the stored values when the layer records no transform |
| `transformed` | the numbers actually stored in the layer, e.g. single-digit arcsinh units |

It is labelling only: neither setting moves a point or touches the data. Both
`open_napari` and `open_napari_transform` take `ticks=` to set it up front, and
the box changes it once the window is open.

### Choosing cofactors

Nothing will choose a cofactor for you, and no default is going to be right for
a 40-channel spectral panel. `open_napari_transform` opens a window that shows
what a cofactor *does* — the two channels against each other, and a distribution
of each with its own slider — and hands back a dict. It writes nothing to the
AnnData; applying the answer is still your own `asinh_transform`.

```python
cofactors = cytopy.open_napari_transform(
    adata, "comp", cofactor_range=(100, 20_000),
)
cofactors
# {
#     'CD4 (BV421-A)': 1200,
#     'CD8 (BV510-A)': 3000,
# }

cytopy.asinh_transform(adata, cofactors, layer="comp", inplace=True)
```

What comes back is a `Cofactors`, which is a dict of channel to cofactor and
goes straight into `asinh_transform`. It remembers a little more than a dict
does: `.layer` is the matrix the numbers were chosen against, `.untouched()`
lists the channels still sitting on the value the window seeded them with,
`.n_adjusted` counts the ones you moved, and `.to_source("COFACTORS")` is the
assignment that the **copy as a Python dict** button puts on the clipboard.

Note that `layer` here is the **untransformed** matrix, unlike `open_napari`,
which is pointed at a layer you have already transformed. This window does the
arcsinh itself.

`cofactor_range` is required, because it is the judgement the tool cannot make
for you: it is your prior on where the answer lives, and it is what the slider's
resolution is spent on. The slider is log-spaced across it — a cofactor is a
scale parameter, so 100→200 is the same step as 3000→6000 — which means
**narrowing the range is how you get finer control**.

The box beside each slider takes an exact number: type it, press Enter, and the
slider follows. The slider quantises to a thousandth of its travel, so a value
typed into the box is kept as you typed it while the slider only moves to the
nearest step it has — dragging is for finding the answer, the box is for
pinning it.

Two things make forty channels tractable:

* **Every channel is seeded before you touch anything**, at the geometric
  midpoint of your range. Nothing is ever unset, so moving between channels
  cannot lose a decision. Channels you have moved are marked `●` in the list,
  the rest `○`. Type a number into **set every channel to** and press Enter and
  the whole panel goes to it, marks cleared — they all hold the same value
  again, so none of them is individually tuned. The real workflow is to settle
  the bulk value on two or three channels, put it on the panel, then go hunting
  for the handful that are wrong.
* **`n` and `p` step through the panel** in file order, so confirming a channel
  is one keystroke.

The axes are labelled in raw units, so the knee line and the cofactor read
against the same scale; **axis ticks** switches them to the arcsinh values
being plotted. **axis margin %** is how far past the data the frame runs — a
tenth of the span at each end by default. Note that the span is in *decades*
here, so a tenth of it is a tenth of a decade and the raw value at the end of
the axis moves by considerably more; wind it up only when a peak is sitting on
the frame.

The y axis defaults to a scatter channel and its slider is greyed out, because
scatter is never transformed. One axis moves at a time, so a change in the
picture is attributable to the slider you moved. Pick a fluorescence channel
there and its slider comes alive.

Two things on the canvas read a cofactor at a glance. The vertical **knee line**
sits at the raw value equal to the cofactor, where the arcsinh stops being
linear; it never moves, and the data slides under it, so the judgement is
"negatives comfortably left of the line, positives spread out to its right". The
shaded **band** is the spread of the negative population, which visibly widens as
you lower the cofactor. Below it the status line reports that spread, the
separation between the negatives and the bright tail, and what fraction of
events are negative at all — a channel with fewer than 50 negative events says
so outright, because there is nothing there for a cofactor to sit on.

Press **copy as a Python dict** and paste the literal into your notebook. The
return value works too, but it lives in the memory of a kernel you will restart;
the literal is a record of what you decided, and it makes the notebook
re-runnable without the window.

### Several files at once

`open_napari` takes one object or many. Several are concatenated on the
channels they share and become entries in the **sample** selector, so one
window browses the lot:

```python
cytopy.open_napari([run1, run2, run3], "asinh")            # AnnData you already have
cytopy.open_napari({"healthy": run1, "treated": run2}, "asinh")  # name them yourself
cytopy.open_napari("data/", "raw")                         # every FCS in a directory
cytopy.open_napari(["a.fcs", "b.h5ad"], "raw")             # paths work too
```

Two files called the same thing stay two entries (`demo`, `demo.1`) rather than
merging. **same axes across samples** is on whenever there is more than one:
without it the axes rescale each time you switch, which is what makes two
samples look alike when they are not. Turn it off to let each sample fill the
plot.

Outside a window the same thing is three functions: `read_fcs_dir` reads a
directory, `concat_samples` stacks objects you already have, and
`split_samples` takes a concatenation apart again. That last one is how the
controls above get gated in one window and then handed to
`compute_spillover_matrix` one tube at a time.

Panels need not match — the intersection of the channels is kept. `uns` comes
from the first object, so per-file provenance (spillover) beyond
the first is not carried over; normalise and gate before combining if you need
it kept.

### Gating

```python
cytopy.open_napari(adata, "asinh")   # opens napari; returns the data when you close it
```

Gate as many times as you like in the one window: draw, apply, change the
channels, set **parent gate** to what you just drew, gate again. Each gate is a
boolean column in `adata.obs` and an outline in
`adata.uns["cytopy"]["gates"]`. Call it again on the same object and those gates
come back with it.

### In the viewer

* Pick the two channels and the matrix to plot (`X` or any layer).
* The plot is a smoothed 2-D histogram; `log counts` is on by default so rare
  populations stay visible.
* Select the **gates** layer, draw a polygon (or rectangle/ellipse), name it and
  hit *apply gate from shapes*. The gate lands in `adata.obs[name]` as a
  boolean column and **stays on screen**, outlined on an *applied gates* layer
  and labelled with its name and its share of its own parent. The editable
  layer clears, so the next gate is only what you draw for it — draw several
  shapes before applying to make one gate out of their union.
* To change a gate later, pick it under **edit gate** and hit *load gate onto
  canvas*: the plot switches to the plane it was drawn in, the outline comes
  back editable, and applying it again replaces it. **Its children are
  recomputed**, so a hierarchy stays consistent when you move a parent. *delete
  gate* removes it and everything nested inside it.
* Outside the viewer, `cytopy.gate_mask(adata, name)` recomputes a gate from
  the outline it was drawn with, and `cytopy.recompute_gates(adata, name)`
  brings its descendants back in line.
* The **gates** layer is always kept on top — a gate you cannot see is a gate
  you cannot adjust.
* Shapes are kept in data coordinates, so changing channel, sample, parent gate
  or bin count moves the axes without moving what a shape covers.
* Clipping the axes means a few events are drawn nowhere and cannot be gated.
  Applying a gate says how many, and `cv.off_axis_count()` reports it at any
  time.
* Set **parent gate** to an existing gate to plot only those events; gates
  applied then intersect with the parent, so hierarchies compose.
* One plot. **plot** switches it between the two-channel **density** and a 1-D
  **histogram** — one smoothed distribution of the x channel per sample, each
  in its own colour with a translucent fill under it, and a legend. The y axis
  is **% of mode** — each curve scaled so its tallest point is 100, as flow
  software does — so a rare population reads against a common one. On a
  histogram a drawn shape gates the interval it spans, which is how a threshold
  is set.
* **samples** picks which samples the plot shows, in either mode. Choosing none
  draws them all; a density pools whatever is picked, a histogram draws a curve
  for each.
* The heading above the plot is `<sample> — <parent gate>`, so it always says
  what you are looking at and what it came from.
* **axis range** is four sliders, one per end. Two handles on one track cannot
  be told apart once they meet, and the end you wanted is then the one you
  cannot grab. They travel in display coordinates — a logicle axis is 0..1
  across, so the travel is spread evenly over the plot instead of being spent
  almost entirely inside the top decade — and each label carries its raw-unit
  equivalent. They read back where the axes are, so they are a readout as much
  as an input, and they can be dragged well past the data to frame an outlier
  with room around it. **fit axes to data** puts them back.
* The axes hold every event by default, padded generously (`AXIS_MARGIN`, a
  tenth of the data span at each end) so nothing sits against the frame. **clip outliers**
  trades that for the 0.1–99.9th percentile, which stops a single extreme event
  — and compensation makes those — stretching the axis until everything else is
  a dot in the corner. It is off unless asked for, because clipped events are
  not drawn and fall outside every gate; on a large panel 0.1% a side is
  thousands of them. `CytoViewer(adata, robust=True)` starts with it on;
  `open_napari` does not take it, so from there it is the checkbox.
* Duplicating the plot's layer in napari does not give a second plot: that
  layer is rewritten on every redraw.

### Exporting a gating hierarchy

```python
cytopy.gating_pdf(adata, "gating.pdf")
```

A contents page with the tree — every gate, its count, its share of its parent
and of the file — then one plot per gate: the biaxial it was actually drawn on,
its parent's events underneath, its own outline over them and the events it
kept picked out. Gates come out parents first, so the pages read the way the
gating was done.

Gate provenance (channels, layer, parent, counts) is kept in
`adata.uns["cytopy"]["gates"]`.

## Data model

| where | what |
| --- | --- |
| `adata.X` | the working matrix, one row per event; starts as the file's values |
| `adata.layers["raw"]` | untouched copy of what was read from the file |
| `adata.var` | `$PnN` channel, `$PnS` marker, `label`, range, gain, kind (scatter/fluor/time) |
| `adata.var["cofactor"]` | the cofactor `asinh_transform` last used, per channel |
| `adata.layers["comp"]` | compensated (`compensate`, default `key_added`) |
| `adata.layers["asinh"]` | arcsinh transformed (`asinh_transform`) |
| `adata.layers["logicle"]` | logicle transformed (`logicle_transform`) |
| `adata.obs["sample"]` | source file, for concatenated runs |
| `adata.obs["file"]` | the path it was read from |
| `adata.obs[<gate>]` | boolean gate membership |
| `adata.uns["fcs"]` | the raw FCS TEXT keywords |
| `adata.uns["spillover"]` | `$SPILLOVER` as a DataFrame, as the file wrote it |
| `adata.uns["timestep"]` | `$TIMESTEP`, when the file gives one |
| `adata.uns["cytopy"]["gates"]` | one record per gate: channels, layer, parent, outline |
| `adata.uns["cytopy"]["asinh_layers"]` | the cofactors behind *each* arcsinh layer, for the axis ticks |
| `adata.uns["cytopy"]["logicle_layers"]` | the `T`/`W`/`M`/`A` behind each logicle layer |
| `adata.uns["cytopy"]["asinh_layer"]`, `["logicle_layer"]` | the layer each transform wrote most recently |
| `adata.uns["cytopy"]["logicle_params"]` | the logicle parameters of the most recent call |
| `adata.uns["cytopy"]["compensated_layer"]` | which layer `compensate` last wrote |
| `adata.uns["cytopy"]["spillover_source"]` | where the applied matrix came from |

All of it round-trips through `adata.write_h5ad(...)` — gates, transform
parameters and the spillover DataFrame all come back as they went in.

Reading a file undoes `$PnE` log amplification (analog log amps on older
instruments store already-logged values; modern files write `$PnE = 0,0` and
this does nothing). `$PnG` gain is **not** applied unless you pass
`apply_gain=True` — instruments disagree about whether the gain is already
baked into the stored values, matching flowCore's `linearize` default. Both
follow flowCore exactly: log-undoing only ever runs on `$DATATYPE = I`
channels, and a channel is never both log-linearised and gain-divided.

`$SPILLOVER` is read as written. Detector names are kept exactly as the file
spells them and checked against the channels it declares — a name that matches
nothing is warned about, not renamed. The matrix is stored whatever it holds:
if the diagonal is not `1` you get a warning saying so, because that means it
is a compensation matrix, or percentages, or neither, and only you can tell
which. Nothing is inverted on the way in; that is `compensate`'s job.
`convert_spillover=True` divides by 100 for a file that stores percentages
— off by default and never inferred from the values.

## Scales

`cytopy.scales` implements the logicle / biexponential transform of
Moore & Parks (2012) with the standard `T`, `W`, `M`, `A` parameters, plus
arcsinh, log and linear. Each scale maps data to display coordinates, back
again, and knows where its decade ticks go. `PretransformedScale` is the one
the viewer uses: identity on the values, but borrowing an inner transform to
place raw-unit ticks.

`LogicleScale.from_data` picks `T` and `W` the way FlowJo and flowCore do —
`W` widens until the linear region covers the spread of the negative
population.


## API reference

Everything below is on the top-level `cytopy` namespace. The names under
*windows* are the only ones that need napari, and they are imported on first
use, so `import cytopy` in a script that never opens a window never loads Qt.

### Reading files

| | |
| --- | --- |
| `read_fcs(path)` | one FCS file into an AnnData, events x channels |
| `read_fcs_dir(directory)` | every FCS in a directory, concatenated |
| `concat_samples(adatas)` | stack objects on the channels they share |
| `split_samples(adata)` | take a concatenation apart, one AnnData per sample |

### Compensation

| | |
| --- | --- |
| `compensate(adata, spillover)` | apply a matrix, writing `layers["comp"]` |
| `compute_spillover_matrix(controls)` | derive one from single-stain controls |
| `subset_controls(controls, gate)` | keep only the gated events, in every control |
| `write_spillover(spill, path)` | a matrix out as CSV; read one back with `pd.read_csv(path, index_col=0)` |

### Transforms

| | |
| --- | --- |
| `asinh_transform(adata, cofactor, layer=)` | `asinh(x / cofactor)`, one cofactor or a dict of them |
| `logicle_transform(adata, layer=)` | logicle, per channel, onto a 0..1 scale |
| `subsample(adata, n)` | a random subset of events, optionally per sample |
| `find_channel_name(adata, name)` | resolve a var_name, marker or detector name to its column index |
| `get_fluor_channels(adata)` | everything but scatter and time |

### Windows

| | |
| --- | --- |
| `open_napari(adata, layer)` | the plotting and gating window; hands the data back |
| `open_napari_transform(adata, layer, cofactor_range=)` | the cofactor window; hands back a `Cofactors` |
| `Cofactors` | a dict of channel to cofactor, plus `.layer`, `.untouched()`, `.n_adjusted` and `.to_source()` |
| `current_viewer()`, `current_transform_window()` | the object behind whichever window was opened last |
| `CytoViewer`, `CofactorWindow` | the classes, which take a `viewer=` if you want the plot inside a napari window you already have |
| `Panel` | one plot inside a `CytoViewer`: the napari layers it draws into, and what it is showing |
| `as_one_anndata(data)` | what both windows do to their first argument |
| `faded_colormap(name)` | a colormap whose bottom end is transparent rather than dark |

### Gates

| | |
| --- | --- |
| `gate_mask(adata, name)` | recompute a gate from the outline it was drawn with |
| `recompute_gates(adata, name)` | bring its descendants back in line after it moved |
| `add_gate(adata, name, mask)` | record a gate from a mask you worked out yourself |
| `gate_record(adata, name)` | its outline, channels, layer and parent, decoded into a `GateRecord` |
| `GateRecord` | that record: `x`, `y`, `layer`, `parent`, `vertices`, `has_outline()` |
| `gate_stats(adata, name)` | count, percent of file, percent of parent |
| `gate_order(adata)` | every gate, parents before their children |
| `gate_children(adata, name)` | the gates nested directly inside one |
| `polygon_mask`, `ellipse_mask`, `shapes_mask`, `rectangle_to_polygon` | the geometry the gates are built on |

### Plots

| | |
| --- | --- |
| `plot_biaxial(adata, x, y, layer=)` | the static counterpart to the viewer |
| `plot_gate(adata, name)` | a gate redrawn in the plane it was drawn in |
| `gating_pdf(adata, path)` | the hierarchy as a PDF, one plot per gate |

### Density

The two functions the viewer and the static plots share, so a figure matches
what was on screen.

| | |
| --- | --- |
| `density_image(x, y, axes)` | 2-D histogram, oriented for a napari image layer |
| `density_curve(values, lo, hi, bins)` | 1-D smoothed distribution, for the histogram plot |
| `Axes2D` | display coordinates to histogram pixels and back |

### Scales

| | |
| --- | --- |
| `get_scale(kind, x)` | build one by name, fitted to the data where that helps |
| `LinearScale`, `LogScale`, `AsinhScale`, `LogicleScale` | the transforms themselves |
| `PretransformedScale(inner)` | identity on the values, inner scale for the ticks |
| `Scale` | the base class: `forward`, `inverse`, and where the decade ticks go |

## Examples and tests


```bash
python examples/make_demo_fcs.py demo.fcs           # synthetic 6-channel data
python examples/make_demo_fcs.py demo.fcs controls  # + single-stain controls
python examples/quickstart.py demo.fcs
pytest                                              # the viewer tests need a display
```
