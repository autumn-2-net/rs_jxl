# libsais-rs

`libsais-rs` is a Rust translation of [`IlyaGrebnov/libsais`](https://github.com/IlyaGrebnov/libsais) for suffix array construction and related transforms.

This crate currently tracks upstream `libsais` version `2.10.4`. 

* 2026-08-01: Added CI
* 2026-05-17: parallelization audit, work on 64bit. Appears to be functional
* 2026-05-15: Feature complete and seemingly tested
* 2026-04-24: Appears to be a functional translation on par with speed. More testing needed, compare with original version before you consider swapping it out


## This is an LLM-mediated faithful (hopefully) translation, not the original code! 

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation



## Usage

```toml
[dependencies]
libsais-rs = "0.2"
```

```rust
use libsais_rs::{libsais, SaSint};

fn main() {
    let text = b"banana";
    let mut sa = vec![0 as SaSint; text.len()];

    let rc = libsais(text, &mut sa, 0, None);
    assert_eq!(rc, 0, "libsais failed with status {rc}");

    println!("{sa:?}");
}
```

Notes:

- `sa.len()` must be at least `text.len() + fs`
- `fs` is extra scratch space made available at the tail of `sa`
- `freq`, when used, must have length at least `256`
- The 64-bit API is available as `libsais_rs::libsais64::libsais64` and uses `libsais_rs::SaSint64`.
- 16-bit input APIs are available under `libsais_rs::libsais16` and `libsais_rs::libsais16x64`.

## Development

The test suite is *differential*: every Rust function is checked against the original C implementation. See [Using the `upstream-c` feature](#using-the-upstream-c-feature) below for the prerequisites, then:

```bash
cargo test --features upstream-c
```

Run the local Rust-vs-C benchmark example with:

```bash
cargo run --release --features upstream-c --example bench_vs_c
```

## Using the `upstream-c` developer feature

The `upstream-c` Cargo feature builds the original C `libsais` alongside the Rust translation and exposes `libsais*_upstream_c*` wrappers around it. It is also what gates the differential test suite. **The upstream C source is not bundled in the crate**, so anyone — contributor or downstream user — who wants this feature must fetch it themselves.

Requirements:

1. The upstream C source at `./libsais/` next to `Cargo.toml`:

   ```bash
   git clone --depth 1 --branch v2.10.4 https://github.com/IlyaGrebnov/libsais.git libsais
   ```

2. GCC with OpenMP (`libgomp`). Other compilers/runtimes (clang/`libomp`, MSVC/`vcomp`) are not supported (let us know if this is a problem).

Practical consequence: the feature only works from a source checkout of this crate that has `./libsais/` populated. It is **not** usable as a regular opt-in feature from a `crates.io` dependency, because Cargo unpacks dependency crates into a location where you cannot inject the C tree. If you need the original C from a downstream project, depend on the upstream C library directly instead.

Default (no `upstream-c`) end users do not need GCC, OpenMP, or any C source — the default build is pure Rust + rayon.

## Performance

Original benchmark baseline: the upstream C source used through the
`upstream-c` feature is commit `b6e52ef33fe1` (`git describe`:
`v2.10.4-1-gb6e52ef`).

The repository includes [`examples/bench_vs_c.rs`](examples/bench_vs_c.rs), which compares the current Rust translation against the upstream C implementation in a single-threaded suffix-array-construction configuration. Requires the `upstream-c` feature (see prerequisites above).

Latest local snapshot, rerun 2026-07-14 at Rust repo commit
`7ef8966377ab23341ac7a967f614034ca18b1017`:

```text
README.md                            len=    8911 iter=200  rust=   0.351 ms  c=   0.286 ms  ratio= 1.23x  rust_rss=    2560 KiB  c_rss=    2240 KiB  rss_ratio= 1.14x
libsais/src/libsais.c                len=  388397 iter= 40  rust=  10.486 ms  c=   8.316 ms  ratio= 1.26x  rust_rss=    4800 KiB  c_rss=    4160 KiB  rss_ratio= 1.15x
generated/mixed-1MiB                 len= 1048576 iter= 10  rust=  33.932 ms  c=  28.836 ms  ratio= 1.18x  rust_rss=    8320 KiB  c_rss=    7360 KiB  rss_ratio= 1.13x
```

For the rustification roll-up, the raw rows are tracked in
`pres_rustification/benchmarks/libsais.tsv`; the aggregate uses C/Rust time
(speedup 0.82x) and Rust/C RSS ratio (1.14x), while this README table prints
Rust/C time as `ratio`.

Command used:

```bash
cargo run --release --features upstream-c --example bench_vs_c
```

These numbers are a local snapshot, not a stability guarantee. Repeated runs can move by a few percent depending on machine, compiler, and system load. RSS is measured as Linux `VmHWM` from separate child processes for the Rust and C implementations.

## Upstream Sources

This crate is a translation of [`IlyaGrebnov/libsais`](https://github.com/IlyaGrebnov/libsais) 2.10.4. The upstream C source is intentionally **not** bundled in this repository or the published crate; fetch it yourself if you need it (see [Using the `upstream-c` feature](#using-the-upstream-c-feature)).

Upstream project:

- <https://github.com/IlyaGrebnov/libsais>

## License

Apache License 2.0.
