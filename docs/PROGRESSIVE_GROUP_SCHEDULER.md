# Progressive Original-Group Scheduler

## Scope

The opt-in scheduler uses cheap pilot admission, then races actual independently
encoded AC groups from the original image when the cheap front is ambiguous.
Real-group measurements preserve the full-image backend; cropped probes are
used only for the existing cheap admission decision. This implementation changes
pixel-lossless effort-9 scheduling only, not the decoder, JPEG byte transcode,
lossy coding, or the wire format.

The existing pilot remains available as the control/fallback. This is not a
guarantee of exhaustive minimum size or a proven 0.25% final-size bound.

## Execution

1. Resolve automatic, neighborhood, literal, and cross-channel candidates on
   the original image. Existing literal/extended-RCT policy is preserved.
2. By default, require the existing scan's continuous-tone evidence (more than
   half of sampled tuples distinct, fewer than half flat). Sparse/low-color
   content keeps the cheap pilot before any real-group encoding begins.
   This is content routing, not a filename/dataset allowlist. Also require
   independently encoded original groups for all racing candidates.
   Shared/global layouts, actual patch dictionaries, shifted extra channels,
   unsupported requests, effort 10, small images and samples wider than eight
   bits retain the old pilot. Wide-group encoding is correct in unit tests,
   but its performance/selection policy is not ready for default admission.
   No patches, palettes, MA tools, or sample depth are silently disabled.
3. Build a macrogrid using the largest candidate group size. Rank cells using
   the full-image activity scan, with deterministic low/mid/high strata that
   include blank regions. Compare candidates over the same spatial rectangles.
   With default routing, decline the real-group race if its initial matched
   regions across all candidates alone exceed twice the original pixel count.
   This avoids spending several full-image equivalents just to choose a route.
4. Run the existing cheap coarse/fine front with its same projection, model
   cost adjustment, worker limits and eligibility rules. If only one finalist
   remains, encode it and any required representation witnesses once directly.
   Do not rerun the front or probe the losing candidates' original groups.
   Otherwise encode actual original groups in the matched rectangles, retaining encoded
   sections. Original geometry, stream IDs, local transforms and MA settings
   are preserved. Group IDs are deduplicated and never encoded twice within a
   candidate's session.
5. Estimate full-image size using measured bytes per sampled pixel in each
   stratum and that stratum's total original pixel area. Tree/model overhead
   remains in measured bytes; it is not subtracted as a crop-only cost.
6. Resume contenders rather than rerun their completed groups. Close losers
   receive additional matched regions, using the completed reference's saved
   group lengths without re-encoding it. Finish the
   original TOC and standard headers only when every required group exists.
7. Keep complete actual outputs in the existing search ledger. Partial retained
   wire bytes may stop a loser only once they reach the incumbent's actual size.

Candidates whose weighted estimate loses by at least 1% can be heuristically
screened after at least three regions if the sum of all positive regional
gains is no more than 0.25% of estimated reference size. Tiny blank-region
headers or tiny wins in a content region cannot veto clear overall losses.
Close challengers can receive three more regions at a time, up to nine by
default. The weighted rejection margin divides by the number of sampling
rounds, with a 25-basis-point floor. This is explicitly distinct from a rigorous
wire-byte bound. Material local winners and learned-literal witnesses are
protected. Fully sampled candidates
are finalized and compared by actual size, not screened by estimates.

Global representation witnesses use full native encoding. They do not reuse
groupwise sessions because their transforms/models have different dependencies.
The two kinds of work must not be spliced together.

## API And CLI

```rust
use rs_jxl::scheduler::{pilot, progressive::GroupRaceOptions};

let policy = pilot::PilotOptions {
    group_race: Some(GroupRaceOptions::default()),
    ..Default::default()
};
let result = pilot::encode_with_options(&image, &params, &options, policy)?;
// None means no group race: fallback OR the cheap front resolved one candidate.
// report.events distinguishes "progressive-cheap-decisive" from old routes.
```

```powershell
.\target\release\rs-jxl.exe encode input.png output.jxl `
  --source-effort 9 --source-search --source-pilot `
  --source-progressive-groups --threads 16 --source-sample-limit 33554432
```

The development tool accepts `--pilot-progressive-groups` together with
`--search --content-pilot`. Its JSON includes matched regions, actual groups
encoded/reused, candidate time, scores, final sizes, and states:

- `Complete`: a finished standard JXL was compared.
- `Screened`: empirical rejection, not a certified size bound.
- `ByteBounded`: retained encoded bytes already reached the incumbent.
- `Sampled`: partial work only; never a returned image.

The lab JSON's `search_report.routes` distinguishes a cheap decisive route from
actual group racing and the old pilot. A single finalist with witnesses reserves
only those full encodes, not the whole racing plan. Independent probes have
separate cumulative work limits.

The lab-only `--pilot-progressive-all-content` disables content/work routing,
the wide-integer admission gate and cheap-front admission, not layout/resource
checks. Its API equivalent is `content_route: false`. This
forced mode is intentionally not the default: earlier forced runs regressed
sparse images and screenshots. `encoder_lab input.png --scan-only` prints the
source scan's color/flat counts and classification without encoding anything.

`probe_encodes` and `probe_pixels` include the cheap admission front. Zero means
no independent cropped probe was encoded, not zero analysis work. Probes are
in memory, not temporary files. Inspect `progressive.group_pixels`, candidate
timings and the entire encoder-call time. `full_encodes` counts full-completion
attempts, including genuine byte-bounded attempts, as in the existing ledger.

`GroupRaceOptions` defaults: three initial matched regions, nine maximum
regions, 100 basis points initial weighted discard margin, and 256 MiB retained
encoded payload/output allowance. Independent groups are dispatched together
in bounded batches covering at most four full 1024-square groups (and no more
than the caller's worker count). The
allowance is checked at work/finalization checkpoints; it is not an RSS limit.
Worker scratch, input buffers and temporary framing/container copies are not
included. Existing output, pixel, MA and search-work limits still apply. There
are no timeouts and no silent effort reductions.

## Validation And Measurements

Implementation tests cover original IDs/geometry, reordered continuation,
duplicate groups, header/container size limits, poisoned/incomplete sessions,
RGB/RGBA 8/16-bit samples including invisible RGB, full-resolution extra
channels, and unsupported-layout fallback. Pixel tests compare decoded output
with original samples, not just with another encoder's output.

The benchmark matrix is kept under
`out/progressive-groups-20261026/matrix.json`. Both policies use the same release
binary, effort 9 and 16 threads. Each output is independently decoded with
official `djxl` and compared with the original pixels. Encoder-call timing
includes the entire search, excluding CLI startup and input/output I/O.

The first implementation (`slow-first/report.json`) was rejected for rollout:
all three original-pixel checks passed and sizes were retained, but floral
rose from 23.16 s to 80.88 s. Tiny blank-header ties kept expensive losers alive
and one-large-group continuation batches underused independent-group
parallelism. This failure motivated material-score tests, adaptive follow-up
regions and bounded multi-group dispatch. The first run is retained as evidence,
not presented as the final result.

The ungated revision (`slow-revised` and `mixed-revised`) independently verified
11 images without size changes. It accelerated wallhaven from 25.18 s to
10.87 s and pizza from 34.44 s to 24.88 s, but floral still took 29.45 s versus
20.55 s. Six additional sparse/screenshot cases also regressed. This is the
reason for retaining the cheap policy for non-continuous content, not evidence
for enabling real-group racing indiscriminately.

Intermediate hybrid measurements (`slow-final/report.json`, two measured rounds with
alternating variant order):

| Image | Old pilot s | Hybrid s | Hybrid MPix/s | JXL bytes | bpp | Route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| wallhaven-mlzoy1 | 26.497 | 11.330 | 0.732 | 14,178,604 | 13.675351 | Real groups |
| floral_frame_PNG270 | 24.145 | 22.742 | 0.794 | 5,100,273 | 2.259042 | Old pilot |
| pizza_PNG44037 | 39.233 | 28.280 | 0.453 | 17,818,198 | 11.116226 | Real groups |

All sizes matched the control and every output passed original-pixel checking
with official djxl. Wallhaven took 57.2% less encoder-call time; pizza took
27.9% less. Floral follows the same old encoding policy; its timing difference
is not claimed as an algorithmic speedup. These are targeted measurements, not
a corpus-wide performance guarantee or a 1 MPix/s guarantee.

The larger mixed holdout (`mixed-final/report.json`) rejected that admission
policy despite its two favorable examples. Cabbage rose from 2.812 s to
15.663 s for 0.88% fewer bytes, ice cream from 3.153 s to 26.583 s for 1.03%
fewer bytes, and the 16-bit `1230.hdr` rose from 27.646 s to 83.850 s while
becoming 0.143% larger. All pixels were correct. The new race had bypassed the
cheap front's existing one-candidate fast decisions. This is why the current
implementation first resolves decisive cheap fronts and retains the old policy
for wide samples. It is not acceptable to count those size gains as a successful
performance optimization.

After adding cheap admission (`cheap-front/report.json`, one paired round):

| Image | Old pilot s | New route s | New MPix/s | JXL bytes | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| cabbage_PNG99102 | 2.909 | 3.019 | 3.622 | 7,849,058 | 5.742448 |
| ice_cream_PNG98208 | 3.876 | 3.568 | 2.946 | 7,672,410 | 5.838830 |
| wallhaven-mlzoy1 | 27.382 | 13.736 | 0.604 | 14,178,604 | 13.675351 |
| pizza_PNG44037 | 41.119 | 36.872 | 0.348 | 17,818,198 | 11.116226 |
| 1230.hdr | 36.346 | 36.070 | 0.333 | 20,172,530 | 13.448353 |
| 1491.hdr | 7.425 | 6.620 | 1.813 | 16,153,050 | 10.768700 |

Every output matched original pixels and the old output size. Cabbage/ice cream
use one full encode; wallhaven/pizza fall from four completion attempts to one
while retaining their partial candidate group work. Wide images use the old
policy; their timing variation is not an algorithmic gain. These timings include
cheap admission overhead. Across rounds, group backend time also varies; the
entire difference from earlier timings cannot be attributed to admission.
The release measured here precedes diagnostic-collector and final-bound edge
fixes; subsequent final-build smoke tests are recorded separately.

Final build, both variants with `--representations` enabled
(`final-diagnostics/report.json`, one paired round):

| Image | Old pilot s | New route s | New MPix/s | JXL bytes | bpp | Route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cabbage_PNG99102 | 3.068 | 3.227 | 3.389 | 7,849,058 | 5.742448 | Cheap decisive |
| wallhaven-mlzoy1 | 29.954 | 13.617 | 0.609 | 14,178,604 | 13.675351 | Real groups |
| pizza_PNG44037 | 42.163 | 27.573 | 0.465 | 17,818,198 | 11.116226 | Real groups |
| stripe.com | 19.150 | 20.417 | 0.425 | 480,260 | 0.443160 | Old pilot |

All eight outputs passed official original-pixel checking. Final sizes matched
the controls. Wallhaven's winner reused 36 completed AC groups; pizza's reused
40. Their total candidate group work was 17,141,760 and 31,015,584 pixels,
respectively, plus the separately reported cheap probes. Thus one completion
attempt does NOT mean one full image's total analysis work. Retained encoded
payload checkpoints peaked at 29.21 and 42.63 MiB, not total process RSS.

Wallhaven and pizza took 54.5% and 34.6% less call time in this round. Cabbage
and stripe instead took 5.2% and 6.6% more; admission overhead and run-to-run
variation remain. Do not claim universal non-regression from these measurements.
Do not combine timing from different binaries/rounds into a best-case result.

Final verification: 320 scheduler tests, 12 group/session tests, 88 CLI tests
and 36 development-tool tests passed. The two CLI tests requiring explicit
external fixture environment variables were not run by the unit-test command;
the benchmark harness independently used official djxl. The final regression
test covers both fully sampled candidates and a last batch reaching an actual
payload cutoff before framing would exceed the output cap. Diagnostic
collectors are fresh for direct full trials, wide extra channels obey the
depth gate, and the explicit representation-witness switch is preserved.

The new scheduler remains opt-in (`PilotOptions::default().group_race == None`).
Enabling it uses content/work/depth routing and cheap admission by default.
No decoder, JPEG byte-transcode, lossy, or CI change was made for this work.

## Ownership

- `modular/group_session.rs`: independently resumable native group work.
- `encode/session.rs`: single-image metadata and complete JXL framing.
- `scheduler/group_regions.rs`: matched, stratified spatial sampling.
- `scheduler/progressive.rs`: candidate racing, continuation and diagnostics.
- `scheduler/native_front.rs`: existing cheap pilot with native worker dispatch.
- `scheduler/pilot.rs`: opt-in integration and ordinary-policy fallback.

Experimental preparation caches, MA histogram binning and square cropped
probes are independent previous experiments, not enabled by this scheduler.
