# cykit-py

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
import cykit

adata = cykit.read_fcs("sample.fcs")        # events x channels, + layers["raw"]
cykit.compensate(adata, inplace=True)       # -> adata.layers["comp"]
cykit.asinh_transform(adata, cofactor=150,  # -> adata.layers["asinh"]
                       layer="comp", inplace=True)

cykit.open_napari(adata, "asinh", x="CD3", y="CD19")   # or pick them in the window
```

or from a terminal:

```bash
cykit sample.fcs --compensate --asinh --cofactor 150 -x CD3 -y CD19
```

### Compensation

`compensate` applies a spillover matrix `S` as `X @ inv(S)`, writing
`layers["comp"]`. There are three ways to get `S`, and all of them end up as the
same square DataFrame indexed by detector, with `1` on the diagonal:

```python
import pandas as pd

cykit.compensate(adata, inplace=True)                  # 1. the file's own $SPILLOVER

# 2. exported by other software: read the CSV yourself, pass the frame
cykit.compensate(adata, pd.read_csv("matrix.csv", index_col=0), inplace=True)

# 3. single-stain controls: you say which file stains which detector
controls = {"CD3 (FITC-A)": cykit.read_fcs("controls/FITC-A.fcs"),
            "CD19 (PE-A)": cykit.read_fcs("controls/PE-A.fcs")}
unstained = cykit.read_fcs("controls/Unstained.fcs")
spill = cykit.compute_spillover_matrix(controls, unstained=unstained)
cykit.compensate(adata, spill, inplace=True)
```

**From single-stain controls.** `compute_spillover_matrix` implements the
Bagwell and Adams matrix-inversion method. For the control stained with dye
*i*, it takes the difference between the positive and negative populations in
every detector *j*, then divides that row through by the difference in the
dye's own detector — so the coefficients do not depend on how bright the
control happened to be, and the diagonal is `1` by construction. Inverting the
assembled matrix is the N-parameter generalisation that lets every detector be
corrected at once.

Gating the controls is two windows, one after the other.

**1. Clean up the controls** in [`cykit.open_napari`](#gating), the same window
you use on a sample. It takes the whole dict at once, pooled into one object
with a `sample` column:

```python
everything = {**controls, "unstained": unstained}
pooled = cykit.open_napari(everything, "asinh")   # draw cells, then singlets, on all samples

gated = cykit.subset_controls(cykit.split_samples(pooled), "singlets")
unstained = gated.pop("unstained")
```

The scatter pass matters more than it looks: every number in the matrix is a
median, and a median over cells *and* debris is a median of nothing in
particular. On a real six-colour panel, gating the controls down to cells took
the worst coefficient from 10.1% off to 4.7% off — before a single positive
gate was drawn.

**2. Gate the positives and negatives, and compute the matrix**, in the
compensation window:

```python
spill = cykit.open_napari_compensation(gated, unstained, "asinh")
cykit.compensate(adata, spill, inplace=True)
```

It has two steps, switched with the **step** box.

*gate controls* shows one control at a time as a distribution of its own
detector, with the **unstained overlaid in another colour** so you can see where
unstained events sit; untick **show unstained** to see the control on its own
(the axis stays put). Pick **positive** or **negative** under **draw**, drag a
box over that population and hit *apply gate from shapes*; only the extent along
the x axis counts. Positive gates are shaded green and negative ones red. A
line marks each gate's median. To
change one, pick it under **draw** and hit *adjust gate*: it comes back on the
canvas as a box you can drag or resize, and applying again replaces it. *delete
gate* removes it. Once the matrix has been computed, a gate changed afterwards
is flagged in the header, because the table no longer follows from the gates
until you compute again. Both gates are gates **on the control**. The unstained is
never gated: a box that also covers its curve gates only the control.

By default the negative population is the control's own negative gate — same
cells, same autofluorescence. Tick **use unstained as negative** to take it from
the whole unstained tube instead, which is what a control without usable
negatives of its own (beads that are all positive) needs. There is nothing to draw
then: the whole tube is the negative, and a red line marks its median. It is per control and
recorded on it (`uns["cykit"]["use_unstained"]`), so it is still set when you
reopen the window. A negative gate the control already has is kept but not
used, and unticking puts it back. The **control** list marks
each one `●` once it has both gates, and `n`/`p` step through them. Gates
already on the controls are loaded and shown as bands.

*compute spillover matrix* refuses until every control has both gates — nothing
is split automatically here — then fills in the table and moves to the second
step.

*check compensation* has two views, picked with the **view** box.

**grid** is the **N × N grid** of biaxial plots: one row and one column per
single-stain control. Tile (*i*, *j*) plots control *i*'s own detector on x
against detector *j* on y, so it shows the coefficient `S[i, j]`. Beside it is a
second N × N grid with the same rows and columns, but with detector *j* on x and
a scatter channel on y. Choose the scatter channel with **scatter y**: `SSC-A` by
default, or `<none>` to hide the second grid. The same coefficient moves the
positive population up in the first grid and sideways in the second. Scatter axes are fitted to the 0.1–99.9th percentile of the
events rather than the detector's `$PnR`, which would squash every population
into a strip.

**two plots** is two large biaxial plots with full axes. The controls are split
into two columns, the left one for the left plot and the right one for the
right. Each plot picks its own control, x and y, scatter included, and has:

* **x range** and **y range** — min and max boxes, in the units the axis is
  ticked in (raw units on an arcsinh or logicle axis). They show the current
  range, so they are a readout as well; type a value and press Enter to fix that
  end, or clear the box to put it back on automatic. Changing the plot's control
  or channel clears its ranges, and *fit axes* clears them all.
* **bins** — how finely that plot's density is binned, from 16 to 1024. Both
  plots are drawn at the size of the finer one, so the coarser one's bins show
  as bigger squares and the two stay side by side at the same size.

Both views show the controls **compensated with the matrix in the table**,
through the transform `layer` recorded, so they read on the same axes as the
gating step, and both redraw whenever a cell is edited (about 0.3 s for the grid
of a 20-colour panel).

A row compensated correctly has its positive population level with its
negative one in every tile but the diagonal. A population tipping up is
under-compensated, one sliding down is over. The green (positive) and red (negative) lines across
each tile are the positive and negative medians. Click a tile in either grid,
or a cell of the table, to pick it out: it is outlined in both grids, the table selects that cell, and the
status line gives the coefficient and the positive-minus-negative gap in raw
units, which should be close to 0. The axes stay fixed while you tune, so it is the
population that moves; *fit axes* refits them. Each tile is scaled to its own
peak, so a dim control is as readable as a bright one. `tile_bins=` sets the
resolution of each tile (64 by default). Each cell has its own − and + to move it by one step, 0.001
by default (the **− / + step** box under the table changes it), or type a value
into it and press Enter.
Edited cells are highlighted; *reset*
puts back the matrix you started from (or last computed); *copy as Python*
puts a `pd.DataFrame(...)` literal on the clipboard for your notebook.

**A matrix the files already carry is loaded.** When the controls have a
`$SPILLOVER`, the table starts from it and the window **opens on the check
step**, so you can tune the instrument's matrix without gating anything.
The matrix always covers **exactly the detectors you have controls for**, square
over those and nothing else. A file's `$SPILLOVER` usually covers the whole
panel, but a detector nobody stained has no dye in it, so it is dropped rather
than inverted: kept, its row would be treated as a dye that is not there and
push error into the real channels. `compensate` then leaves that channel as it
is. This is the `spillover=` argument, `"file"` by default. Controls carrying different
matrices, or a diagonal that is not `1`, are an error rather than a guess. Pass
a DataFrame to start from one you have, or `None` for the identity.

The window writes nothing but the gates: `obs["positive"]` and
`obs["negative"]` on each control, recorded like any other gate. So the matrix
can be recomputed without the window:

```python
spill = cykit.compute_spillover_matrix(
    gated, positive_gate="positive", negative_gate="negative",
    unstained=unstained, use_unstained=["CD8 (APC-A)"],   # only if you ticked any
)
```

Without the window, `compute_spillover_matrix` falls back for any control
missing a gate: no `positive_gate` means an automatic split of the stained
detector on an arcsinh scale (Otsu), and no `negative_gate` means the events
outside the positive gate, or `unstained=` as a universal negative.
`thresholds=` takes a raw cutoff per control instead.

The gating is done on an arcsinh display — on a linear axis the whole negative
population lands in one bin — but **the matrix is computed on the raw values**,
because spillover is linear and `asinh(a) − asinh(b)` is not proportional to
`a − b`. Transforming the controls first inflates the coefficients
several-fold. A control that is not brighter than its negatives in its own
detector is an error rather than a quietly wrong row.

**Adjusting one by hand** outside the window: the matrix is a DataFrame, so a
coefficient is an assignment.

```python
spill.loc["CD3 (FITC-A)", "CD19 (PE-A)"] = 0.13
control = cykit.compensate(controls["CD3 (FITC-A)"], spill)
cykit.plot_biaxial(control, "CD3 (FITC-A)", "CD19 (PE-A)", layer="comp", cofactor=150)
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
cykit sample.fcs --compensate --asinh          # the file's own $SPILLOVER
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
cykit.plot_biaxial(adata, "CD3", "CD19", layer="asinh")      # density
cykit.plot_biaxial(adata, "CD3", "CD19", layer="asinh", color_by="lymphs")  # a gate over a density
cykit.plot_biaxial(adata, "CD3", "CD19", layer="raw", cofactor=150)        # display-only arcsinh
cykit.plot_gate(adata, "lymphocytes")                        # redraw a gate where it was drawn
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
cykit.compensate(adata, spill, inplace=True)              # adata gains layers["comp"]
comped = cykit.compensate(adata, spill)                   # adata untouched
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
cofactors = cykit.open_napari_transform(
    adata, "comp", cofactor_range=(100, 20_000),
)
cofactors
# {
#     'CD4 (BV421-A)': 1200,
#     'CD8 (BV510-A)': 3000,
# }

cykit.asinh_transform(adata, cofactors, layer="comp", inplace=True)
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
cykit.open_napari([run1, run2, run3], "asinh")            # AnnData you already have
cykit.open_napari({"healthy": run1, "treated": run2}, "asinh")  # name them yourself
cykit.open_napari("data/", "raw")                         # every FCS in a directory
cykit.open_napari(["a.fcs", "b.h5ad"], "raw")             # paths work too
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
cykit.open_napari(adata, "asinh")   # opens napari; returns the data when you close it
```

Gate as many times as you like in the one window: draw, apply, change the
channels, set **parent gate** to what you just drew, gate again. Each gate is a
boolean column in `adata.obs` and an outline in
`adata.uns["cykit"]["gates"]`. Call it again on the same object and those gates
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
* Outside the viewer, `cykit.gate_mask(adata, name)` recomputes a gate from
  the outline it was drawn with, and `cykit.recompute_gates(adata, name)`
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
cykit.gating_pdf(adata, "gating.pdf")
```

A contents page with the tree — every gate, its count, its share of its parent
and of the file — then one plot per gate: the biaxial it was actually drawn on,
its parent's events underneath, its own outline over them and the events it
kept picked out. Gates come out parents first, so the pages read the way the
gating was done.

Gate provenance (channels, layer, parent, counts) is kept in
`adata.uns["cykit"]["gates"]`.

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
| `adata.uns["cykit"]["gates"]` | one record per gate: channels, layer, parent, outline |
| `adata.uns["cykit"]["asinh_layers"]` | the cofactors behind *each* arcsinh layer, for the axis ticks |
| `adata.uns["cykit"]["logicle_layers"]` | the `T`/`W`/`M`/`A` behind each logicle layer |
| `adata.uns["cykit"]["asinh_layer"]`, `["logicle_layer"]` | the layer each transform wrote most recently |
| `adata.uns["cykit"]["logicle_params"]` | the logicle parameters of the most recent call |
| `adata.uns["cykit"]["compensated_layer"]` | which layer `compensate` last wrote |
| `adata.uns["cykit"]["spillover_source"]` | where the applied matrix came from |

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

`cykit.scales` implements the logicle / biexponential transform of
Moore & Parks (2012) with the standard `T`, `W`, `M`, `A` parameters, plus
arcsinh, log and linear. Each scale maps data to display coordinates, back
again, and knows where its decade ticks go. `PretransformedScale` is the one
the viewer uses: identity on the values, but borrowing an inner transform to
place raw-unit ticks.

`LogicleScale.from_data` picks `T` and `W` the way FlowJo and flowCore do —
`W` widens until the linear region covers the spread of the negative
population.


## API reference

Everything below is on the top-level `cykit` namespace. The names under
*windows* are the only ones that need napari, and they are imported on first
use, so `import cykit` in a script that never opens a window never loads Qt.

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
| `compute_spillover_matrix(controls)` | derive one from single-stain controls, by `positive_gate=`/`negative_gate=` or automatically |
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
| `open_napari_compensation(controls, unstained, layer)` | the compensation window; hands back the spillover matrix, edited live |
| `Cofactors` | a dict of channel to cofactor, plus `.layer`, `.untouched()`, `.n_adjusted` and `.to_source()` |
| `current_viewer()`, `current_transform_window()`, `current_compensation_window()` | the object behind whichever window was opened last |
| `CytoViewer`, `CofactorWindow`, `CompensationWindow` | the classes, which take a `viewer=` if you want the plot inside a napari window you already have |
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
