# GuitarSet real-guitar fixture

`guitarset_solo_5s.wav` is the first five seconds of the mono microphone
recording `00_BN1-129-Eb_solo_mic.wav`. `guitarset_solo_5s.jams` is a modified
five-second excerpt of its annotation: other namespaces were removed, the six
`note_midi` tracks were retained, events outside the clip were removed and
note durations crossing the boundary were clipped. The clip has 20 annotated
note events, including simultaneous notes. `guitarset_solo_5s_pitch.json` is
an additional derived excerpt of the original six `pitch_contour` namespaces:
only voiced positive-frequency observations in the first five seconds are
retained, grouped by original string index. `guitarset_audio_manifest.json`
references and hashes the local five-second clip and JAMS for an end-to-end
harness smoke run; it is not a held-out or corpus-scale selection.

**Attribution and license:** GuitarSet, version 1.1.0, by Qingyang Xi, Rachel
M. Bittner, Xuzhou Ye, Johan Pauwels and Juan P. Bello,
[Zenodo record 3371780](https://doi.org/10.5281/zenodo.3371780), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The fixture is a
modified excerpt as described above; the license of these data is separate
from the MIT-licensed mtab source code and from MuScriptor's gated CC BY-NC
weights. The dataset's published known annotation errors may affect scores.

SHA-256:

- WAV: `2308bb2dff9e2e332a9e9eb53e6a2e9915519a0388ef7e42a9b1102259f3f60a`
- JAMS: `d9b29010bb0ef6a49b5a2553e3b14819a9090cd4e170f2004853a91f37a7a1fd`
- Pitch contour JSON: `edb7162ae786068c10d69f78dd3958cdb035bda56e5ceb3c3b321c3efc11abdd`
- Original full WAV: `2a6057d0485d8ee2f4a04175ee085d610c0f7a13194e083b13ce803bec091d43`
- Original full JAMS: `2258ef6f5dc58437f2807dfc0c05416ad8498cd345f95d6563b13d19283e6d54`

Recreate from those original files with:

```sh
python tools/extract_guitarset_fixture.py \
  --audio 00_BN1-129-Eb_solo_mic.wav \
  --annotation 00_BN1-129-Eb_solo.jams \
  --output-dir crates/mtab-cli/tests/fixtures/guitarset
```

To regenerate the pitch-contour JSON from the original JAMS:

```sh
python tools/extract_guitarset_pitch_fixture.py 00_BN1-129-Eb_solo.jams \
  crates/mtab-cli/tests/fixtures/guitarset/guitarset_solo_5s_pitch.json
```

The scripts verify the original file hashes. Python is only a fixture
regeneration dependency, not a runtime dependency. This single excerpt is not
a held-out corpus, a representative quality benchmark, or a tab-labeler-v3
comparison.
