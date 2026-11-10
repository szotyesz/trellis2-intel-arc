# XPU measurements

Run 2026-11-02 with `scripts/xpu_probe.py` in the `trellis` container: torch 2.15.0+xpu, compute-runtime 26.35, Arc Pro B60.

## Runtime

- `torch.xpu` initialises, with one device. `libsycl` loads from `/opt/venv`.
- `mem_get_info`: 23.91 GiB total (`clinfo` reports 22.71 GiB).
- `_fused_sdp_choice` returns 4 (not MATH) for bf16, fp16 and fp32.
- All smoke-tested ops ran with the CPU fallback disabled. A 6 GiB allocation works without relaxed limits.

## SDPA memory, `[1, 12, L, 128]`, torch peak in GiB

| L | bf16 | fp16 | fp32 |
|---|---|---|---|
| 4096 | 0.01 | 0.01 | 0.77 |
| 8192 | 0.02 | 0.02 | 3.05 |
| 16384 | 0.05 | 0.05 | 12.09 |
| 32768 | 0.09 | 0.09 | OOM, tried 48 GiB |
| 49152 | 0.14 | 0.14 | not run |

bf16 and fp16 scale with L, so no chunked attention is needed. **fp32 builds the L×L tensor, so do not fall back to fp32 flow models.**

Driver-level peaks are not useful here: the sampler stayed near 6.47 GiB after the 6 GiB allocation test.

Timings are not recorded. They imply an implausible throughput (0.26 s at L=49152), so `torch.xpu.synchronize()` may not have waited for the work. Measure real speed from pipeline runs.

## Accuracy against CPU

- SDPA at L=1024, maximum absolute error: bf16 0.029, fp16 0.0034, fp32 0.00001. No NaNs.
- `torch.mm`, 2^18 to 2^22 rows: fp32 error 0.0. fp16 error stays about 0.027 at every size, so no corruption appears at large row counts.

## Pipeline runs

Measured 2026-11-03 and 04 on one machine with 12 inputs, seed 42. Total is wall time for one image in a fresh container, with a 200,000-triangle target and a 1024 px texture. Figures cover successful runs from the matrix below, so higher levels look cheaper than they are.

| Type | Completed | Total, median (range), min | Generate, median, s | Peak VRAM, median, GiB | Raw triangles, median, M |
|---|---|---|---|---|---|
| `512` | 12/12 | 1.9 (1.3–2.9) | 29 | 2.8 | 3.3 |
| `1024` | 10/12 | 5.2 (1.7–14.1) | 92 | 8.7 | 12.1 |
| `1024_cascade` | 7/12 | 5.0 (2.1–7.9) | 81 | 7.9 | 10.1 |
| `1536_cascade` | 4/12 | 6.4 (3.3–10.6) | 176 | 12.0 | 16.1 |

This matrix ran **before** the GPU memory fixes below, so its completion counts are out of date. It needs repeating. Peak system RAM was about 17 GiB at every level.

Same-seed runs are not identical: one input gave 3.01M and 3.08M raw triangles in two runs.

## System RAM

RAM, not VRAM, first limited `1024_cascade`. All models stayed in RAM, reaching 23.8 GiB before baking added about 6 GiB, over the container's 26 GiB cap. Skipping unused models, dropping each after its last stage and calling `malloc_trim` cut the peak to about 16 GiB.

## Baking

The extracted mesh has many edges shared by three or more faces (221K of 4.3M on one input), and about 9% of manifold edges have inconsistent winding. These stalled the simplifier near 248K triangles and made xatlas take about 3 hours. Fixing winding and unwelding those faces before simplifying brings a 3M-triangle mesh to under a minute of unwrapping. A 12.7M-triangle mesh took 37 s to simplify and 247 s to unwrap.

xatlas still makes many small charts (about 32K on a 100K-face test), so UV seams are plentiful.

## GPU memory failures

All 15 GPU failures in the matrix happened while decoding. Of the 10 that still failed after the first fix, all were in the `conv_none` matmul. Successive fixes, each retested on inputs that still failed:

| Fix in `backends/conv_none.py` | Result |
|---|---|
| Sync the queue after each kernel slot | Recovered 5 of 15 |
| Chunk the matmul at 2M rows | Recovered 1 of the remaining 10; failures moved to ordinary tensor allocations |
| `empty_cache` before convolutions over 4M rows | Recovered 2 of 5 tested |

Not every run was retested with the final code, so only the repeat matrix will give the true completion counts. Known failures on the final code: `farmer` at `1536_cascade`, `fishbowl` at `1024_cascade`, `interior` at `1024`.

The remaining failures still show about 10–12 GiB reserved by the allocator but unused, with 8–10 GiB in use, so fragmentation inside a decoder pass looks like the main cause. The control run (`machine`, `1024_cascade`) kept its speed: 377 s against 386 s.

After a device error, Python can hang on exit, so `generate.py` now exits immediately on a GPU error.

## Colour

Output is often darker or less saturated than the input, and differs between levels. Findings so far:

- **Baking is not the cause by inspection.** The channel layout, value scaling and inpainting match upstream's `to_glb`, with no gamma step in either.
- **XPU sampling is not the cause.** From identical inputs and noise at 512, XPU and CPU texture latents differ by a mean of 0.0064 against a latent standard deviation of 2.66 (0.24%).
- **Not yet tested:** decoder output on XPU against CPU (the CPU decode ran out of RAM in the test container), `conv_none` against upstream's FlexGEMM, and DINOv3 features on XPU against CPU.
- **Possibly expected:** the model predicts PBR base colour, which removes lighting, so some darkening may be normal. A reference run of upstream on CUDA would settle this.

## Update, 2026-11-06

### GPU memory, final state

With the convolution chunked by bytes and stored at the input precision, every benchmark failure that was retested completed its GPU phase, including every `1536_cascade` case, at 13–19 GiB peak VRAM. Three of those runs then failed on the CPU: two timed out in UV unwrapping after more than 35 minutes and one exceeded the container's RAM cap while baking a 46M-triangle mesh. Those are the baking problems below.

### Baking large meshes

Measured on saved raw meshes: the machine at `1024_cascade` (12.2M triangles) and the bear at `1536_cascade` (26.1M).

| Step | Before | After |
|---|---|---|
| Repair of non-manifold edges | Faces on them got private vertices, which the simplifier never removes: 62% of final faces were isolated triangles on the 12M mesh | Vertices split into fans connected through manifold edges: 1–3% isolated |
| Meshes over 8M triangles | Repair and simplification needed over 26 GiB on 46M triangles | Vertex clustering at 2 voxels first: 12M → 2.7M in 7 s, then everything is small |
| Simplify 12.2M → 200K | 37 s, 6.1 GB peak RAM | 9 s, 1.9 GB |
| Simplify 26.1M → 200K | Not attempted before | 17 s |
| Unwrap, 12.2M input | 247 s | 16–25 s |
| Unwrap, 26.1M input | Over 35 min, killed | 196–214 s |
| Texels with voxel data at 1536 | 70–89% trilinear | 97–99.7% after a nearest-voxel fallback within 3 voxels |

Full runs after the change: farmer at `1536_cascade` (57M triangles, previously killed at 26 GiB) baked in 32 s with 6 GB; bear at `1536_cascade` in 227 s; interior at `1024` in 42 s (previously over 50 minutes).

Clustering at 3 voxels, which the 57M mesh got, left it fragmented into 91K pieces. The factor is now capped at 2. The farmer at that level has not been rerun since; the repeated benchmark will show it.

### Pairing sheets at non-manifold edges

An alternative repair pairs the two faces that continue straight through a non-manifold edge instead of cutting all of them apart. On the 512 machine mesh it gives 121K vertices for 200K faces instead of 203K, 51K boundary edges instead of 137K, and 24K UV charts instead of 38K, but xatlas takes 686 s instead of 24 s; 359 s against 25 s on the 1024 mesh. It was added as `--repair pair`, off by default, with its visual benefit unchecked; the bake comparison in the 2026-11-07 update below made it the default.

### Triton sparse convolution

One Triton kernel gathers the neighbour rows of each output row and multiplies them in one pass, fp32 accumulation. Triton's JIT needs `g++` and the Level Zero headers in the image; the SYCL headers come from the `intel-sycl-rt` wheel.

- Kernel alone, 400K rows, 64 channels, fp16: 3.1 ms against 265.7 ms for the loop.
- Per-call profile of a `1024_cascade` run (`TRELLIS_CONV_PROFILE=1`): the loop spends 13 s in convolution. The first Triton version spent 44 s, because `@triton.autotune` and compile-time channel counts cost about 4.5 s per distinct layer shape, and there are eighteen. With one fixed configuration and runtime channel counts, the kernel is compiled once and served from the cache: convolution time 5 s, generation 115 s against 121 s, both decodes 3 s against 7 s, peak VRAM 7.7 GiB against 6.8 GiB for the chunked loop.
- Inputs under 16K rows use the loop: 2,048 rows at 1024 channels took 0.5 s per call in Triton and 0.04 s looped.

So the gain is in decoding, not in generation time, which the flow transformers dominate. The earlier 9–12 GiB peaks at `1024_cascade` came from the loop's temporaries, which the chunking had already removed.

## Update, 2026-11-07

### GPU memory: allocator fragmentation

The two remaining benchmark failures, fishbowl at `1024` and town at `1536_cascade`, died in a decoder layer norm asking for 4.6–5.1 GiB while the allocator held 5–7 GiB reserved but unusable. `torch==2.15.0+xpu` supports expandable segments on XPU, so `generate.py` now sets `PYTORCH_ALLOC_CONF=expandable_segments:True`. Both inputs then completed: fishbowl at 19.6 GiB peak with a 45.9M-triangle mesh, town at 20.2 GiB with 47.3M triangles. No other run changed.

### Cracks in baked meshes

The white patches on the farmer's knees at `1536_cascade`, the speckles on cloth and paper at the cascade levels and the holes in the interior's walls are all gaps in the mesh, not texture: the farmer's texture has 306 white texels in total. Measured on a saved 512 machine mesh (2.9M triangles), counting edges that have one face after merging vertices by position:

| Step | Open edges |
|---|---|
| Raw extractor output | 3.3%, plus 221K edges shared by three or more faces |
| After splitting those edges into fans | 3.3%: both sides of every cut still coincide |
| After one quadric decimation pass to 200K | 30.3% |

The decimator moves the two sides of each cut independently, so every cut opens into a crack. The same happens to the fishbowl at `512`: its glass is a double-walled shell with 4% such edges, so nearly the whole surface is cut, and after decimation it is 68,737 pieces with 134 triangles covering half the area. The first benchmark's repair gave faces on those edges private vertices, which produced isolated triangles instead of cracks and left that shell smooth, so the "regression" is the same defect in a different form.

A render-based gap measure, pixels filled by a 5-pixel morphological closing of the silhouette at 768 px over four views, puts numbers on what the eye sees: the speckled typewriter at `1024_cascade` 7.6% of its silhouette, the farmer 0.8–0.9%, the clean-looking bear 0.4%.

Fixes tried on the 512 mesh, none adopted:

| Variant | Open edges after | Faces | Unwrap |
|---|---|---|---|
| Current: cut, decimate, cut | 30.6% | 199K | 24 s |
| Decimate first, cut only for xatlas | 8.9%, gap pixels 0.36% against 0.91% | 179K | xatlas ran over 30 min on this and on the fishbowl before being killed |
| Cut, decimate with borders locked | 12.0% | 852K, stalls far from the target | – |
| Cut, decimate, weld open vertices within 1 or 2 voxels | 22.3% / 10.5% | 154K / 109K: the welds also collapse the triangles along the cuts | – |
| Cut, decimate tracking collapses, weld copies of each original vertex | 9.8% | 80K: each survivor absorbed many originals whose other copies survived elsewhere, so the welds chain | 19 s |
| Pair sheets first, decimate, weld within 2 voxels | 8.8% | 167K | – |

On a 12.2M-triangle `1024` mesh, which goes through vertex clustering first, the picture is the same: 2.4% open edges raw, 27.8% after cut and decimation, 7.2% decimating first. Pairing then welding within 2 voxels gave 10.0% while keeping 186K of 200K faces there, against 109K on the 512 mesh, because a voxel is a smaller fraction of a final triangle at higher levels. Its unwrap time was not measured and it is not wired in.

Filling the extractor's small holes first (3,147 loops of at most 16 edges and 0.03 perimeter) changed nothing downstream. The practical lever today is `--repair pair`, which halves the cuts; its bake comparison follows.

### Pairing sheets: bake comparison

Three `1024_cascade` meshes captured with `--save-raw` and baked both ways with `scripts/bake_raw.py`. Open edges are counted after merging vertices by position; gap pixels as above.

| Asset, raw triangles | Open edges, cut → pair | Gap pixels, cut → pair | Unwrap, cut → pair |
|---|---|---|---|
| Typewriter, 22.3M | 24.4% → 13.9% | 5.39% → 0.53% | 17 s → 122 s |
| Stroller, 12.0M | 43.8% → 27.9% | 2.45% → 0.49% | 2 s → 334 s |
| Farmer, 17.5M | 55.5% → 30.4% | 0.64% → 0.28% | 3 s → 1,170 s |

Pairing halves the open edges and removes most of the visible speckle: the typewriter's paper and the stroller's cloth read as surfaces again, and the farmer's shirt loses its noise. Texel hit rates and triangle counts are unchanged. The price is the unwrap, 2 to 20 minutes against seconds. `pair` is now the default, since the output is what matters here, and `--repair cut` remains for quick previews.

### Background removal

Scored on the 12 example images, each composited over white, grey, gradient and noise backgrounds, as IoU of the mask against the image's own alpha. SAM 2.1 large and SAM 3 through `transformers` 5.22.2 on the XPU.

| Method | Mean IoU | Median | Notes |
|---|---|---|---|
| SAM 2, grid of 16 points, union of masks that touch at most one edge of the frame | 0.84 | 0.97 | Mid-grey backgrounds: farmer 0.21, bear 0.52, stroller 0.48. Scenes 0.5–0.7 |
| SAM 2, true bounding box 5% loose | 0.78 | 0.95 | A box covering over 85% of the image makes the model pick the background; such boxes fall back to the grid |
| SAM 2, centre point | 0.62 | 0.78 | Fails where the centre is a gap: wings 0.00 |
| SAM 3, the object's name | 0.85 | 0.97 | The same number on every background. Keeps only what the name covers: farmer 0.72 without the basket, bear 0.77 without the pedestal. "room" found nothing. Several names are unioned: the farmer on a gradient scored 0.72 with "farmer", 0.84 with "farmer basket" and 0.98 with "farmer basket vegetables"; the phrase "farmer with a basket" stayed at 0.72 |
| SAM 3, generic words "object", "thing", "product" | 0.03–0.15 | 0.00 | Unusable: the model needs a concrete noun |
| SAM 3, bounding box | 0.02 | 0.00 | Not working through this API; not investigated |

Picking the mask by SAM 2's own score chooses a part of the object (pineapple 0.00, machine 0.03 with a centre point); the largest of its three masks is the whole object. SAM 3's mask decoder crashed the XPU `group_norm` kernel with a segmentation fault; a group norm written in plain tensor operations avoids it. SAM 3 takes about 2 s per image on the B60 and 42 s on the CPU; SAM 2 under a second.

`transformers` moved from 4.57.6 to 5.22.2 for SAM 3. A `512` generation on the new image produced 1.53M raw triangles for the pineapple against 1.55M before, within the run-to-run variation, and the unit tests pass.

### Spikes from an int32 overflow in the mesh repair

The first benchmark with `--repair pair` as the default (7 November, all 48 main runs successful) produced straw-like triangles spanning the whole model on the stroller and the farmer at `1536_cascade`, and both overshot the triangle target (228,887 and 213,375 faces). Measured as the share of edges longer than 20 times the median edge, the stroller went from 0.12% in the 6 November run to 1.90%, the farmer from 0.04% to 0.47%.

The cause is an integer overflow, present since the repair was written. `orient_faces` and `split_non_manifold_vertices` key each edge as `min(a, b) * (n_vertices + 1) + max(a, b)`. SciPy's component labels and the simplifier's faces are int32, so on any mesh with more than 46,341 vertices the product wraps around and unrelated edges share a key. Those edges are then treated as one non-manifold edge: cut mode splits vertices that needed no split, and pair mode can join corners of different vertices into one, which stretches a triangle across the model. In cut mode only the final repair after decimation ran on int32 faces; in pair mode the second split always received int32 labels from the first, which is why the damage grew with the pairing default and with mesh size. All index arrays are now int64, and a test runs the repair on a 163,842-vertex sphere with int32 faces.

Same raw mesh, before and after the fix, long edges after the final repair:

| Mesh | cut, before → after | pair, before → after |
|---|---|---|
| Stroller `1024_cascade`, 12.0M raw | 0.370% → 0.325% | 0.015% → 0.000% |
| Stroller `1536_cascade`, 32.7M raw | – → 0.240% | – → 0.016% |

The long edges that remain in cut mode come from the decimator working on a mesh that is mostly boundary, not from the repair; pairing leaves far fewer. The fixed stroller at `1536_cascade` reaches the 200,000 target with both repairs. The 7 November results are kept under `benchmark/results-v3-spikes` and the benchmark was started again on the fixed code.

### Pairing stalls on some meshes: an off-surface guard

The benchmark on the fixed repair (8 November, 52 of 52 real runs successful) brought back one of the two 1536 problems in a different form. On the farmer at `1536_cascade` the quadric simplifier stopped at 243,837 faces on the paired mesh, with the hat and hands flattened into large facets, and the baker's trilinear hit rate fell from 94.5% to 85.5%; on the town it stalled far above the target, so the clustering fallback ran and overshot to 79,319 faces with a 58% hit rate. The same meshes with the cut repair reach the target in two passes. The simplifier's aggressiveness setting is not a lever: level 5 changes little and level 3 stalls everywhere.

The practical measure is the share of face area whose centroid has no occupied voxel within one voxel, a proxy for the texels the baker will miss. Pair against cut on the same raw mesh, after the final repair:

| Raw mesh | Pair | Cut | Chosen |
|---|---|---|---|
| Fishbowl `512`, 7.3M | 0.3% | 0.2% | pair |
| Stroller `1024_cascade`, 12.0M | 0.3% | 3.3% | pair |
| Typewriter `1024_cascade`, 22.3M | 1.3% | 1.7% | pair |
| Plant `1024_cascade`, 5.5M | 1.1% | 0.8% | pair |
| Farmer `1024_cascade`, 17.5M | 2.0% | 3.2% | pair |
| Stroller `1536_cascade`, 32.7M | 6.7% | 6.0% | pair |
| Town `1024`, 25.7M | 11.3% | 11.1% | pair |
| Farmer `1536_cascade`, 54.4M | 14.8% | 2.7% | cut |
| Town `1536_cascade`, 45.6M | 40.8% | 9.6% | cut |

`simplify_mesh` now applies this: with `--repair pair` it decimates the paired mesh and, if more than 3% of the area is off the surface or the decimator stopped more than 10% above the target, decimates the cut mesh as well and keeps the cut result when that lies at least 3 percentage points closer to the surface. On the seven meshes where pairing is sound this costs nothing or one extra decimation of about a minute; on the two where it fails it restores the cut result, with its cracks but on the surface. The generator prints which repair was used and why. The town at `1024` shows that the proxy also measures what pairing cannot fix: a scene with many thin walls loses a tenth of its area to bridging triangles under either repair.

## Update, 2026-11-10

### Bridging triangles: a decimation stage that stays on the shell

The town and the interior lost about a tenth of their area to triangles off the voxel shell under either repair. Measured on the saved town `1024` mesh (25.7M raw triangles) after each step of `simplify_mesh`, as the share of face area whose centroid has no occupied voxel within one voxel (the proxy from the previous section) and, from a KD-tree over the voxel centres, the share more than one and more than three voxels away:

| Step | Faces | Proxy | > 1 voxel | > 3 voxels |
|---|---|---|---|---|
| Raw | 25.7M | 0.0% | 0.0% | 0.0% |
| Clustered at two voxels | 5.65M | 0.0% | 0.0% | 0.0% |
| Paired repair | 5.65M | 0.0% | 0.0% | 0.0% |
| First quadric decimation pass | 265K | 10.5% | 17.9% | 4.2% |
| Second pass and final repair | 194K | 11.3% | 19.5% | 4.1% |

All of it enters in the decimation. The far faces are large, with a median edge of 18 voxels against 7 for the mesh, and of the 4.1% beyond three voxels, 1.6% have all three vertices on the surface and span a window, an arcade or an alley, while 1.9% hang from a few vertices placed off the surface: the eight largest shared one vertex 14.8 voxels out. The rest, 15% of the area between one and three voxels off, is facades approximated by triangles too large for their relief. The loss is a budget effect: decimating the same mesh to 400K faces leaves 0.9% beyond three voxels and 9.4% beyond one, 800K leaves 0.0% and 0.8%, and 1.6M leaves 0.1% beyond one voxel.

Quadric error measures how far a vertex moves from the planes around it and nothing about the triangle between the vertices, so a vertex on a facade slides along it at no cost and the faces fan over the openings, while every collapse in the foliage and the balustrades has a high error and keeps its triangles. The renders show this: at 200K faces the cypresses are still dense and the facades are a few large facets. Rescaling the coordinates to voxel units, which shifts the simplifier's threshold schedule by six orders of magnitude, changed nothing on the final mesh (17.4% beyond one voxel against 17.6%) and made the paired mesh stall and flatten, so the order of collapses is not the lever.

`backends/surface_decimate.py` is the fix: a quadric decimator over a priority queue that refuses any collapse whose new triangles would have a centroid off the shell by the proxy's test, and takes such collapses, in error order, only once no clean collapse is left. The C++ decimator brings the mesh to four times the target first, where its faces still follow the shell, and the Python stage finishes at about 200,000 collapses a minute. Two details matter. The minimiser of each quadric is taken by a pseudo-inverse anchored at the edge midpoint and dropped if it lies more than two edge lengths away, which removes the displaced hub vertices. And every open edge gets a constraint quadric, the plane through the edge perpendicular to its face, weighted by ten times the squared edge length: without it the collapses along a flat sheet's border are free, the borders erode and the cracks between cut sheets open. On the stroller's cut mesh that constraint took the silhouette gap pixels from 2.21% to 0.55%, under the plain decimator's 0.81%; a weight of 100 gained nothing and pushed 5% of the area over a voxel off.

Same repaired meshes, plain decimation against the shell stage, both at the 200,000 target and after the final repair. Gap pixels are the render-based crack measure from the 7 November section:

| Raw mesh | Proxy, plain → shell | > 1 voxel, plain → shell | Gap pixels, plain → shell | Area, plain → shell | Shell stage |
|---|---|---|---|---|---|
| Town `1024`, 25.7M, pair | 9.7% → 0.0% | 17.7% → 3.5% | 0.19% → 0.22% | 7.95 → 8.01 | 187 s |
| Stroller `1024_cascade`, 12.0M, pair | 0.3% → 0.0% | 2.7% → 0.2% | 0.31% → 0.29% | 3.39 → 3.42 | 168 s |
| Stroller `1024_cascade`, 12.0M, cut | 3.0% → 0.0% | 9.1% → 1.2% | 0.81% → 0.55% | 2.85 → 2.93 | 188 s |
| Plant `1024_cascade`, 5.5M, pair | 1.1% → 0.0% | 3.9% → 0.0% | 7.78% → 7.95% | 1.58 → 1.61 | 175 s |
| Fishbowl `512`, 7.3M, pair | 0.3% → 0.0% | 1.4% → 0.0% | 0.00% → 0.00% | 9.64 → 9.68 | 159 s |
| Typewriter `1024_cascade`, 22.3M, pair | 1.3% → 0.0% | – → 0.5% | 0.10% → 0.10% | 9.19 → 9.24 | 156 s |
| Town `1536_cascade`, 45.6M, pair | 40.8%, stalled → 0.1% | – → 9.4% | – → 0.17% | – → 8.19 | 202 s |
| Farmer `1536_cascade`, 54.4M, pair | 14.8%, stalled → 0.0% | – → 12.6% | – → 0.25% | – → 5.81 | 230 s |

The surface area of the 4× intermediate is 8.33 for the town, so the plain result had lost 5% of the area to collapsed detail and gained some back as bridging faces, and the shell result keeps 96% of it. On the two meshes where pairing stalled and the guard fell back to cutting, the C++ decimator reaches the 800K intermediate and the shell stage, which lets interior vertices merge into border vertices, reaches the target, so pairing stands and the fallback is not needed; the guard stays as a safety net. What remains on scenes is relief of one to two voxels that 200,000 triangles cannot follow: 3.5% of the town's area at `1024`, 9 to 13% at `1536_cascade`, where a voxel is smaller. The fewer than 0.1% of collapses that would leave the shell were never needed on any of these meshes. The stage costs about three minutes of CPU per asset and is on whenever `simplify_mesh` gets the voxel coordinates; `shell_decimation=False` restores the plain path.

Baked through `scripts/bake_raw.py`, the town `1024` mesh confirms it where it counts: 98.2% of the texels find a voxel by trilinear sampling and 99.99% within three voxels, against 68.4% and 84.2% in the 8 November benchmark run, and the facades carry their windows, arcades and balconies instead of flat facets with smeared colour. The unwrap took 17 minutes against 27 in that run (a different raw mesh, so indicative only), the simplification 226 s instead of 43 s.
