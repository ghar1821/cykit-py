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
cytopy.compensate(adata, inplace=True)                  # 1. the file's own $SPILLOVER
cytopy.compensate(adata, "matrix.csv", inplace=True)    # 2. exported by other software

controls, unstained = cytopy.read_controls("controls/")   # 3. single-stain controls
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
controls, unstained = cytopy.read_controls("controls/")
everything = {**controls, "unstained": unstained}

pooled = cytopy.open_napari(everything, "asinh")   # ONE window, every control in it

gated = cytopy.subset_controls(cytopy.split_samples(pooled), "singlets")
unstained = gated.pop("unstained")
spill = cytopy.compute_spillover_matrix(gated, unstained=unstained, positive_gate="positive")
```

There is no special function for controls — [`cytopy.gate`](#gating) is the same
one you use on a sample, and it takes the whole dict at once. The controls are
pooled into one object with a `sample` column, and the **samples** list picks
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
particular. On a real six-colour panel it took the worst residual from 10.1%
down to 4.7% before a single positive gate was drawn.

Each control is its own AnnData, so the gate lives in *that* control's
`obs["positive"]` and the same name works for all of them. A control with no
gate falls back to the automatic split, so you can hand-gate only the awkward
ones. `thresholds=` takes a raw cutoff per control instead.

The gating is done on an arcsinh display — on a linear axis the whole negative
population lands in one bin — but **the matrix is computed on the raw values**,
because spillover is linear and `asinh(a) − asinh(b)` is not proportional to
`a − b`. Transforming the controls first inflates the coefficients several-fold. The negative reference is the control's own
negative events by default — same beads, same autofluorescence — or pass
`unstained=` to use a universal negative instead. A control that does not split
into two populations is an error rather than a quietly wrong row.

`read_controls` matches each file in a directory to the detector it stains by
name (`FITC-A.fcs`, `Compensation Controls_PE-A.fcs`, `CD3 FITC.fcs` all work,
against both `$PnN` and `$PnS`), falling back to whichever channel separates
best. It picks the unstained tube out by name too.

**Adjusting one by hand.** The matrix is a DataFrame, so a coefficient is an
assignment — and there are two ways to see whether the change helped:

```python
spill.loc["CD3 (FITC-A)", "CD19 (PE-A)"] = 0.13

cytopy.compensation_residuals(controls, spill, unstained=unstained)
#               CD3 (FITC-A)  CD19 (PE-A)  CD8 (APC-A)
# CD3 (FITC-A)           0.0        0.070       -0.006   <- under-compensated
# CD19 (PE-A)           -0.0        0.000       -0.000

cytopy.plot_compensation(controls, spill, unstained=unstained)
```

A single-stain control compensated correctly has its positive population level
with its negative one in every detector but its own. `compensation_residuals`
measures exactly that — it derives a matrix *from the compensated controls*,
which should come out as the identity, and reports what it is instead. Read a
cell as the fraction of the dye's signal still leaking: **positive is
under-compensated, negative is over-compensated**, zero is right.

`plot_compensation` is the same thing to look at: a row per control, a column
per detector, with a dashed line at the negative population's level. A
population tipping up is under-compensated, one sliding down is over.

**From a CSV.** `read_spillover` copes with what the common exporters write:
with or without a row-name column, comma / tab / semicolon separated, fractions
or percentages, and names decorated as `Comp-FITC-A` or `FITC-A :: CD3`. Pass
`inverted=True` if the file holds a compensation matrix rather than a spillover
matrix. `write_spillover` writes one back out.

From the terminal:

```bash
cytopy sample.fcs --controls controls/ --asinh
cytopy sample.fcs --spillover matrix.csv --asinh
```

### Reports

Every step that drops events records what it dropped, so at the end you can
account for all of them without having kept the intermediates:

```python
cytopy.filter_log(adata)
#      step                                     reason  n_before  n_removed  n_after
#    debris        outside the scatter gate (4,109 ...     60000       4109    55891
#      dead                 7AAD positive (3,098 ...     55891       3098    52793
```

`cytopy.report` turns that, and the plots behind it, into one self-contained
HTML file — no assets alongside it, opens anywhere:

```python
cytopy.report(adata, "qc.html", title="Run 1")
```

It picks up whatever the data carries: the event tally at each step and every
gate in the plane it was drawn in. Sections with nothing behind them are left
out rather than left empty.

### Biaxial plots

`plot_biaxial` is the static counterpart to the viewer — the same smoothed 2-D
histogram, so a figure in a report matches what was on screen when the gate was
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

`view` takes one object or many. Several are concatenated on the channels they
share and become entries in the **sample** selector, so one window browses the
lot:

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
* **axis range** is one slider per axis, holding both ends. It travels in
  display coordinates — a logicle axis is 0..1 across, so the slider is spread
  evenly over the plot instead of spending nearly all its length inside the top
  decade — and its label carries the raw-unit equivalent. It reads back where
  the axes are, so it is a readout as much as an input, and it can be dragged
  well past the data so an outlier can be framed with room around it.
* The axes hold every event by default, padded generously (`AXIS_MARGIN`, half
  the data span at each end) so nothing sits against the frame. **clip outliers**
  trades that for the 0.1–99.9th percentile, which stops a single extreme event
  — and compensation makes those — stretching the axis until everything else is
  a dot in the corner. It is off unless asked for, because clipped events are
  not drawn and fall outside every gate; on a large panel 0.1% a side is
  thousands of them. `robust=True` to `view` starts with it on.
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
| `adata.var` | `$PnN` channel, `$PnS` marker, range, gain, kind (scatter/fluor/time) |
| `adata.layers["comp"]` | compensated |
| `adata.layers["asinh"]` | arcsinh transformed |
| `adata.obs["sample"]` | source file, for concatenated runs |
| `adata.obs[<gate>]` | boolean gate membership |
| `adata.uns["cytopy"]["filters"]` | what each step removed, and why |
| `adata.uns["fcs"]` | the raw FCS TEXT keywords |
| `adata.uns["spillover"]` | `$SPILLOVER` as a DataFrame |
| `adata.uns["cytopy"]["spillover_source"]` | where the applied matrix came from |

Everything round-trips through `adata.write_h5ad(...)` except `uns["spillover"]`
being restored as a plain array.

Reading a file undoes `$PnE` log amplification (analog log amps on older
instruments store already-logged values; modern files write `$PnE = 0,0` and
this does nothing). `$PnG` gain is **not** applied unless you pass
`apply_gain=True` — instruments disagree about whether the gain is already
baked into the stored values, matching flowCore's `linearize` default. Both
follow flowCore exactly: log-undoing only ever runs on `$DATATYPE = I`
channels, and a channel is never both log-linearised and gain-divided.

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


```bash
python examples/make_demo_fcs.py demo.fcs           # synthetic 6-channel data
python examples/make_demo_fcs.py demo.fcs controls  # + single-stain controls
python examples/quickstart.py demo.fcs
pytest                                              # the viewer tests need a display
```
