# Cooperative JPEG MA Handoff

Source: frozen read-only libjxl `b866b3defec13494bfbaf705db93581cda118e37`,
`enc_modular.cc:1518-1588`, `enc_lossless_probe_pool.h`,
`enc_lossless_workers.h`, and `modular/encoding/enc_ma_histograms.h`.

## API and Execution

Both `jpeg::TranscodeOptions` and the coefficient backend's `Options` add
`cooperative_ma: Option<scheduler::WorkerLimits>`, default `None`. Some enables
the source cooperative JPEG split route; it does not select an effort, quality,
sampling fraction, different tree domains, or a new learning algorithm.
Limits validate before reading JPEG input or creating a pool. CLI defaults
remain unchanged; the parent added `cooperative_ma: None` to its full literal.

`transcode::with_threads` lends `Option<&rayon::ThreadPool>` to its closure.
Direct encoding and every whole-frame JPEG search candidate forward this same
borrow to `Prepared::encode`, then `Shared::build_with_execution`. The old
auxiliary `Shared::build` entry remains available. No new pool, global Rayon
dispatch, asynchronous lifetime, or source environment-variable lookup is added.

The cooperative branch requires a caller pool, a nonserial execution scope,
the explicit option, and more than one useful nonempty domain. Predefined-tree
domains count, matching source `useful_splits`; empty domains do not. Current
JPEG policies are freshly constructed for the normal source model route, with
no MA-search instrumentation, direct audit, final-plan replay, or channel-tree
tool enabled. Those future routes must preserve the source exclusions.

The source requires `JXL_MA_SPLIT_PARALLEL=1`, while
`JXL_MA_CHUNK_PARALLEL` and `JXL_LOSSLESS_JPEG_MA_SPEEDUPS` allow unset or `1`.
The typed option expresses that explicit combined opt-in; None preserves the
source's default domain-parallel, serial-split execution. A single useful domain
does not enable this JPEG-only cooperative route.

`WorkerPlan::cooperative` uses the source 32 MiB per-learner histogram estimate.
Maximum live roots are `min(domains, tasks, max(1, scratch_bytes / 32MiB))`.
Default WorkerLimits are eight tasks and 1 GiB; one oversized root may still run.
This is a concurrency estimate, not a total allocation/RSS bound. Rayon keeps
the existing documented executor differences: child tasks may use the supplied
pool's full width; C++ queue priority, quantum and quiet wakes are not reproduced.

Only split execution is enabled inside cooperative roots. Gathering remains
serial, with unchanged sampling/RNG/quantiles and actual root stream IDs. The
same atomic selected-sample ceiling covers all domains, alongside each learner's
local limit. Deduplication cannot evade the ceiling; exhaustion returns an error,
not a truncated tree or reduced fraction. Domain merge order, effective contexts,
shared entropy, JPEG search candidates and winner selection remain unchanged.

One-thread requests and enclosing serial scopes pass no pool. A null auxiliary
runner enforces serial work throughout model construction. Cooperative errors
stop new root claims, drain active tasks, and propagate the error; state does not
outlive the borrowed pool.

## Focused Verification

Added unit coverage for default-off and single-domain dispatch, explicit-pool
identity, null/enclosing serial scopes, root caps under nested work, ordered
results, invalid limits, cancellation and recovery after unwinding. Source
learning comparisons cover zero/fractional/full sampling, empty ID domains,
predefined domains, effective tree/token/model equality, and both local and
shared selected-sample errors. JPEG API tests cover early option validation,
unchanged sinks on failure, pool lending, and unchanged whole-frame search
trials/winners/output under the opt-in.

New ignored official oracle:
`jpeg::transcode_reference_tests::rust_cooperative_jpeg_ma_with_official_decoders`.
It needs only `RS_JXL_DJXL`, uses existing 444/420 restart fixtures, and checks
exact reconstructed JPEG bytes for serial opt-in plus one/two cooperative roots.
Two official decoder paths yield 12 comparisons. Cached entropy is erased before
encoding, as in the existing independent acceptance harness.

No Cargo/build/test/benchmark/decoder or commit command is run by this agent.
Parent central validation subsequently passed all workspace tests in debug and
optimized profiles, plus all 12 independent official-decoder comparisons above.
Strict all-target Clippy and Rust 1.89 all-target checks passed. These fixture
results do not establish general speedups. No ARM changes are included.
