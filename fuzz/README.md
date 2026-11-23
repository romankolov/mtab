# Fuzzing

These opt-in libFuzzer targets exercise the in-memory safetensors parser and
loader, and Symphonia audio decoding through a temporary file. They do not use
or download model weights. `cargo-fuzz` requires a separate nightly toolchain;
the workspace's Rust 1.75 minimum applies to the product, not fuzz tooling.

```sh
cargo install cargo-fuzz --locked
rustup toolchain install nightly --profile minimal
python3 fuzz/seed.py
cargo +nightly fuzz run safetensors -- -max_total_time=60 -max_len=1048576
cargo +nightly fuzz run audio_decode -- -max_total_time=60 -max_len=1048576
```

Corpora, crashes and build outputs stay ignored locally. Seed inputs should be
synthetic or permissively licensed and contain no gated weights. The audio
target writes per-process files in the OS temporary directory; run it where
that is writable and not against untrusted concurrent processes. OS resource
limits, sanitizer behavior, the supplied corpus and short run duration bound
what this can establish; a smoke run is not proof of memory or parser safety.

Smoke runs (30,000 executions each) found one upstream defect: Symphonia
0.5.5 panics on a WAVE/AIFF `fmt ` chunk whose sample rate is zero, because
`TimeBase::new(1, 0)` aborts. `[patch.crates-io]` points
`symphonia-format-riff` at `vendor/symphonia-format-riff`, which is unmodified
0.5.5 source plus a zero-sample-rate rejection returning an unsupported-format
error. The replayable seed is `fuzz/seeds/audio_decode/zero_sample_rate.wav`,
and `mtab-audio` has a matching integration test.

## Findings

Bounded smoke runs (30,000 executions for safetensors, then time-boxed audio
sessions) found the following. Each was reproduced, minimized by libFuzzer,
fixed in the vendored Symphonia crates, and kept as a tracked seed plus a
`mtab-audio` regression test.

| finding | trigger | fix |
|---|---|---|
| `TimeBase` assert abort | WAVE/AIFF `fmt ` sample rate 0 | reject zero rate (`vendor/symphonia-format-riff`) |
| ADTS frame-count underflow | MPEG-4/ADTS probe starting past EOF | checked subtraction and zero step (`vendor/symphonia-codec-aac`) |
| up-front 4 GiB allocation | FLAC picture MIME/description/data length, Vorbis comment length | bounded chunked reads (`vendor/symphonia-metadata`) |
| application block underflow/allocation | FLAC application block length < 4 or inflated | checked length, bounded reads (`vendor/symphonia-utils-xiph`) |
| forged scalefactor-band count | AAC `max_sfb` past the sample-rate band table | bounds check (`vendor/symphonia-codec-aac`) |
| decoded-size multiplication | header-declared frame counts and per-packet growth | `MAX_DECODED_SAMPLES`/`MAX_DECODED_SECONDS` in `mtab-audio` |

The safetensors target ran 2,024,150 executions in 61 s, then 150 s more with
the grown corpus and no findings. The
audio target ran 713,851 executions in 241 s after the AAC fix above with no
further findings. Sessions were limited by `-rss_limit_mb=512`/`-timeout=10`
because the workspace has 1 GiB of RAM and two cores. Absence of further findings is not
evidence of safety; a longer, multi-core, larger-input campaign is outstanding.
