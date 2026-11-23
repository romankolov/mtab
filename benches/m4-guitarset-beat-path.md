# GuitarSet: preliminary 50-recording beat-path evaluation

This is a **rhythm-development holdout diagnostic**, not a full-corpus or
project-wide held-out benchmark. The rhythm algorithm and tightness of 3.0
were fixed before these 50 recordings were selected. They exclude player 00,
whose five-second excerpt informed rhythm development. Other mtab work has
used GuitarSet annotations across the corpus, so the project as a whole has
not been isolated from these examples. This deterministic selection covers
players 01–05, five styles, and both solo/comp, but only the lexicographically
first take per group: **four tempos (100, 114, 129, 130 BPM), one progression
per style, all 4/4**. It is not representative of all music, tempo ranges,
performance rubato, time signatures, instruments, or mixes.

This evaluation exposes a substantial limitation. With reference tempo
provided, the 50-file **micro beat F1 was 52.82%**; selecting the strongest
positive autocorrelation candidate without a prior or calibrated abstention
reduced it to **45.48%**. A correct-looking tempo does not resolve beat phase,
and even the oracle-tempo path has substantial errors. The autocorrelation
selection is an evaluation baseline, not a CLI feature or a production
tracker. No parameters or code were adjusted to improve these 50 results.

Reference beats are the GuitarSet recording **click-track** positions, not
measurements of local performed-beat deviations. Both reference and estimated
beats before 5 seconds are discarded, and events are matched one-to-one within
70 ms using `mir_eval` 0.8.2. Micro F1 aggregates TP/FP/FN across recordings;
macro F1 is the arithmetic mean of per-recording F1. Notes and fretting are
not evaluated here. No model weights or MuScriptor inference are involved.
See the [GuitarSet paper](https://archives.ismir.net/ismir2018/paper/000188.pdf)
and [dataset 1.1.0 record](https://zenodo.org/records/3371780) (CC BY 4.0;
Q. Xi, R. Bittner, X. Ye, J. Pauwels and J. P. Bello). Recordings are not
committed to this repository.

## Provenance and reproduction

Run used mtab commit `fddde9cfefd56a2b6283c7863039daab8bce82c7`, Python
3.13, mir_eval 0.8.2, NumPy 2.5.5 and SciPy 1.18.1. The report below is the
unmodified stdout of `tools/evaluate_guitarset_beat_paths.py`. Its input hashes
identify the bytes analyzed. The annotation archive matched its published MD5
`b39b78e63d3446f2e54ddb7a54df9b10`. The 50 WAVs were extracted with CRC
checks via HTTP-range reads from Zenodo's published mono-mic ZIP; because the
complete audio ZIP was **not** downloaded, its published MD5 was not checked.
CRC and the recorded SHA-256 hashes are not independent authenticity proofs.
For stronger full-archive verification, use the steps below.

```sh
# External data, approximately 696 MB downloaded; do not commit it.
mkdir -p /tmp/mtab-guitarset/audio
curl -L -o /tmp/mtab-guitarset/annotation.zip \
  https://zenodo.org/api/records/3371780/files/annotation.zip/content
curl -L -o /tmp/mtab-guitarset/audio_mono-mic.zip \
  https://zenodo.org/api/records/3371780/files/audio_mono-mic.zip/content
printf '%s  %s\n' \
  b39b78e63d3446f2e54ddb7a54df9b10 /tmp/mtab-guitarset/annotation.zip \
  275966d6610ac34999b58426beb119c3 /tmp/mtab-guitarset/audio_mono-mic.zip | md5sum -c
unzip -q /tmp/mtab-guitarset/audio_mono-mic.zip -d /tmp/mtab-guitarset/audio
# Python 3.13 was used for the measured run; Python is a test-only dependency.
python -m pip install mir_eval==0.8.2 numpy==2.5.5 scipy==1.18.1
cargo build --locked --release -p mtab-audio \
  --example onset_dump --example periodicity_dump --example beat_path_dump
python tools/evaluate_guitarset_beat_paths.py \
  --annotation-zip /tmp/mtab-guitarset/annotation.zip \
  --audio-dir /tmp/mtab-guitarset/audio \
  --examples-dir target/release/examples > guitarset-beat-report.md
```

The three release-example SHA-256 values for this run, respectively, were
`6cda6107d3a90d245ed97fe736b98c6e220a03c71733cc59ba25c5dfc0ff3665`,
`c110e7fb702a5f4cc1fd085f59758ba0d085f3d038d7eb97bbb53e0b743ad6e1`, and
`41c18a6c79d03fef8d57d7d89a404da2d49218a18fa5019406f8c59a3389f9e6`.
Executable hashes vary with build environment; these are provenance, not
required hashes for reproducing metrics. The output includes all individual
recording and input hashes below.

---

# GuitarSet beat-path diagnostic: frozen 50-recording selection

Fixed parameters: quarter-note references, trim both sequences below 5.0 s; mir_eval 0.8.2, ±70 ms maximum-cardinality matching; tightness 3.0. Oracle uses annotated constant BPM; auto uses strongest positive autocorrelation local maximum (no prior, calibrated confidence or abstention except no peak).

Selection: players 01–05, styles BN/Funk/Jazz/Rock/SS, comp and solo; lexicographically first take for each group. Player 00 was excluded because its excerpt informed rhythm development. This is a small fixed selection, not the full corpus or a project-wide held-out set (GuitarSet annotations informed other work).

Annotation archive MD5: `b39b78e63d3446f2e54ddb7a54df9b10`. The per-recording SHA-256 values below identify analyzed bytes, not independent proof of authenticity.

| Mode | TP | FP | FN | Micro F1 | Mean per-recording F1 |
|---|---:|---:|---:|---:|---:|
| Oracle tempo | 1006 | 923 | 874 | 0.5282 | 0.5284 |
| Top autocorrelation candidate | 880 | 1110 | 1000 | 0.4548 | 0.4463 |

Top tempo within 4% of reference: 20/50; within 4% of half, equal or double reference: 32/50. These are simple diagnostic ratios, not calibrated confidence or a standard corpus metric.

| Recording | Reference BPM | Candidate BPM | Oracle TP/FP/FN | Oracle F1 | Auto TP/FP/FN | Auto F1 |
|---|---:|---:|---|---:|---|---:|
| 01_BN1-129-Eb_comp | 129.00 | 65.22 | 37/0/0 | 1.0000 | 19/0/18 | 0.6786 |
| 01_BN1-129-Eb_solo | 129.00 | 86.96 | 33/5/4 | 0.8800 | 12/15/25 | 0.3750 |
| 01_Funk1-114-Ab_comp | 114.00 | 153.85 | 26/16/12 | 0.6500 | 13/37/25 | 0.2955 |
| 01_Funk1-114-Ab_solo | 114.00 | 230.77 | 25/13/13 | 0.6579 | 38/39/0 | 0.6609 |
| 01_Jazz1-130-D_comp | 130.00 | 130.43 | 23/14/14 | 0.6216 | 23/14/14 | 0.6216 |
| 01_Jazz1-130-D_solo | 130.00 | 130.43 | 34/4/3 | 0.9067 | 34/4/3 | 0.9067 |
| 01_Rock1-130-A_comp | 130.00 | 51.72 | 29/11/8 | 0.7532 | 9/5/28 | 0.3529 |
| 01_Rock1-130-A_solo | 130.00 | 130.43 | 4/34/33 | 0.1067 | 4/34/33 | 0.1067 |
| 01_SS1-100-C#_comp | 100.00 | 50.00 | 34/6/5 | 0.8608 | 19/1/20 | 0.6441 |
| 01_SS1-100-C#_solo | 100.00 | 200.00 | 31/9/8 | 0.7848 | 26/55/13 | 0.4333 |
| 02_BN1-129-Eb_comp | 129.00 | 85.71 | 25/13/12 | 0.6667 | 14/11/23 | 0.4516 |
| 02_BN1-129-Eb_solo | 129.00 | 51.72 | 18/19/19 | 0.4865 | 10/6/27 | 0.3774 |
| 02_Funk1-114-Ab_comp | 114.00 | 115.38 | 37/1/1 | 0.9737 | 37/1/1 | 0.9737 |
| 02_Funk1-114-Ab_solo | 114.00 | 113.21 | 35/4/3 | 0.9091 | 35/4/3 | 0.9091 |
| 02_Jazz1-130-D_comp | 130.00 | 65.22 | 26/12/11 | 0.6933 | 18/1/19 | 0.6429 |
| 02_Jazz1-130-D_solo | 130.00 | 127.66 | 35/3/2 | 0.9333 | 35/2/2 | 0.9459 |
| 02_Rock1-130-A_comp | 130.00 | 52.17 | 16/23/21 | 0.4211 | 5/12/32 | 0.1852 |
| 02_Rock1-130-A_solo | 130.00 | 127.66 | 13/24/24 | 0.3514 | 13/23/24 | 0.3562 |
| 02_SS1-100-C#_comp | 100.00 | 100.00 | 3/37/36 | 0.0759 | 3/37/36 | 0.0759 |
| 02_SS1-100-C#_solo | 100.00 | 98.36 | 29/11/10 | 0.7342 | 29/11/10 | 0.7342 |
| 03_BN1-129-Eb_comp | 129.00 | 127.66 | 11/28/26 | 0.2895 | 11/28/26 | 0.2895 |
| 03_BN1-129-Eb_solo | 129.00 | 130.43 | 14/22/23 | 0.3836 | 14/23/23 | 0.3784 |
| 03_Funk1-114-Ab_comp | 114.00 | 45.80 | 20/20/18 | 0.5128 | 10/6/28 | 0.3704 |
| 03_Funk1-114-Ab_solo | 114.00 | 89.55 | 28/10/10 | 0.7368 | 17/15/21 | 0.4857 |
| 03_Jazz1-130-D_comp | 130.00 | 43.80 | 5/33/32 | 0.1333 | 1/11/36 | 0.0408 |
| 03_Jazz1-130-D_solo | 130.00 | 130.43 | 3/35/34 | 0.0800 | 3/35/34 | 0.0800 |
| 03_Rock1-130-A_comp | 130.00 | 86.96 | 31/7/6 | 0.8267 | 11/14/26 | 0.3548 |
| 03_Rock1-130-A_solo | 130.00 | 130.43 | 18/19/19 | 0.4865 | 18/19/19 | 0.4865 |
| 03_SS1-100-C#_comp | 100.00 | 100.00 | 0/40/39 | 0.0000 | 0/40/39 | 0.0000 |
| 03_SS1-100-C#_solo | 100.00 | 193.55 | 15/27/24 | 0.3704 | 36/44/3 | 0.6050 |
| 04_BN1-129-Eb_comp | 129.00 | 130.43 | 18/18/19 | 0.4932 | 15/22/22 | 0.4054 |
| 04_BN1-129-Eb_solo | 129.00 | 240.00 | 13/25/24 | 0.3467 | 24/44/13 | 0.4571 |
| 04_Funk1-114-Ab_comp | 114.00 | 150.00 | 17/23/21 | 0.4359 | 11/39/27 | 0.2500 |
| 04_Funk1-114-Ab_solo | 114.00 | 230.77 | 25/13/13 | 0.6579 | 31/46/7 | 0.5391 |
| 04_Jazz1-130-D_comp | 130.00 | 96.77 | 33/4/4 | 0.8919 | 12/16/25 | 0.3692 |
| 04_Jazz1-130-D_solo | 130.00 | 136.36 | 5/33/32 | 0.1333 | 7/32/30 | 0.1842 |
| 04_Rock1-130-A_comp | 130.00 | 130.43 | 37/0/0 | 1.0000 | 37/0/0 | 1.0000 |
| 04_Rock1-130-A_solo | 130.00 | 51.72 | 22/17/15 | 0.5789 | 9/6/28 | 0.3462 |
| 04_SS1-100-C#_comp | 100.00 | 50.00 | 25/16/14 | 0.6250 | 13/7/26 | 0.4407 |
| 04_SS1-100-C#_solo | 100.00 | 101.69 | 19/22/20 | 0.4750 | 19/22/20 | 0.4750 |
| 05_BN1-129-Eb_comp | 129.00 | 130.43 | 2/34/35 | 0.0548 | 2/36/35 | 0.0533 |
| 05_BN1-129-Eb_solo | 129.00 | 200.00 | 6/30/31 | 0.1644 | 18/41/19 | 0.3750 |
| 05_Funk1-114-Ab_comp | 114.00 | 230.77 | 23/16/15 | 0.5974 | 38/38/0 | 0.6667 |
| 05_Funk1-114-Ab_solo | 114.00 | 222.22 | 11/28/27 | 0.2857 | 29/47/9 | 0.5088 |
| 05_Jazz1-130-D_comp | 130.00 | 130.43 | 12/25/25 | 0.3243 | 12/25/25 | 0.3243 |
| 05_Jazz1-130-D_solo | 130.00 | 146.34 | 14/25/23 | 0.3684 | 15/28/22 | 0.3750 |
| 05_Rock1-130-A_comp | 130.00 | 65.22 | 23/15/14 | 0.6133 | 17/2/20 | 0.6071 |
| 05_Rock1-130-A_solo | 130.00 | 133.33 | 12/27/25 | 0.3158 | 12/27/25 | 0.3158 |
| 05_SS1-100-C#_comp | 100.00 | 130.43 | 16/27/23 | 0.3902 | 15/35/24 | 0.3371 |
| 05_SS1-100-C#_solo | 100.00 | 193.55 | 15/25/24 | 0.3797 | 27/50/12 | 0.4655 |

## Input hashes

| Recording | WAV SHA-256 | JAMS SHA-256 |
|---|---|---|
| 01_BN1-129-Eb_comp | `4f7f2359cc26d32307c8cc6793579d9d5ce602f483194ca079efd719a870b671` | `5591debe5bf770446f431a9148ade1577d8def54936b5ce82289ff859787795f` |
| 01_BN1-129-Eb_solo | `76330537c6a88df56c7e338da9dd81d84cbf7f3a5ae7cd1b32a663bded6b76b4` | `ecdd877eaf6208cfefa47b74c1a6a5211146e12dfa0f5132345a4c456cd67fc8` |
| 01_Funk1-114-Ab_comp | `9ee4888317faba89ba1a9929d6e3b044a3363eb925e6c432b2b30e4f3c83ab22` | `429e06078c84f38ae3d551bdf594d0cac75a58c43783b23e1d436363a9fbe97a` |
| 01_Funk1-114-Ab_solo | `8e301305985cf6d26bfe0abfb76b60adb29df962d91071985b5dc65eba0cdd4e` | `42fe7a65d53eec16a3dfa5cea2b9af01fbd1e09b6025f80c45501f25c28d6fd8` |
| 01_Jazz1-130-D_comp | `d45184fa42482eefc1d21d5cef3777bf85e772bf9d8ba7508ff441b77a8ee199` | `23c5bad0ce5eb208786ac1823957600bee8a0bb588aab80d7b335a2e2ca90352` |
| 01_Jazz1-130-D_solo | `726991869569d7b76fe69b36fbefb3571f030556f020a7e3d599f0728d60f3ef` | `39226e2e4bc3178272d22f940b66986fec22474cf8c2b0ec4838942b469b942d` |
| 01_Rock1-130-A_comp | `1ead2a124768727b6ee3ddcfec2df7ad5f7087fa74ac0a49822d0028685f8731` | `546bce99efd52256b4c6a03f0281292d5ba30761d7f799e3ada703d1617873d1` |
| 01_Rock1-130-A_solo | `650a35f45d80be91e46859568fbb1a85392bcd7a8a5116dbf02b8ac9cba47fb1` | `a49b8dfbf4e98c5e05151ed591b8d0546d9c10cdba3b47d45d5a3a2d8c2fd220` |
| 01_SS1-100-C#_comp | `c8b3cb71fa42be463da53bc2114f42741366a9b1e5b80a158a38402961a3c5a2` | `556990d214788813d5088d1e8c9a933d6b34e8f64b3bf2747e396c476e3c2aec` |
| 01_SS1-100-C#_solo | `2b9c9eda652c2ebf9dab32d906f757973427c00c3df9f945e357a54156e47069` | `86377a1d21a57900b5df1f4e05230faedc83e6a17a5c7277887f2e5fc1c2e8ed` |
| 02_BN1-129-Eb_comp | `54d0b5d30a727ddbf3f9dbe18663063d6c55c979627ab6993fdede63f0a54cc0` | `38208980f05c69d1560306553651f687a3c6260f47d98b64e33eb74c33e14931` |
| 02_BN1-129-Eb_solo | `9a99f25ef470da34b6a85189d2895f64b59c20b653114b59b24cf7efa3239f13` | `3ff0c7c98cc3abb4c37a0023030991c0c5b112caad2c53962636c62f260524aa` |
| 02_Funk1-114-Ab_comp | `cbe0eac7ebf0f5f0c38642d56b5317187a9fbc406bc8ab1425452d88b56420cf` | `978f22805ee73be2299e215d2ccb1107cf2639c25829db9a5cdb285d56b2454c` |
| 02_Funk1-114-Ab_solo | `7c10e95b9d378a4bae8b3d5a6ce11841a3fdfa4e65059df0e6fe3526eb31b7a2` | `e7252a03938166bae5d7b394798f16f681fe40a3316f10c2c2d48fb7ad0168c5` |
| 02_Jazz1-130-D_comp | `7b769c278bb3d44440420a79cba1e486119f823a33f78465aa9acdcdef277df5` | `e06bd1f4e088b0a0892366193a2d951420288cd9709bfaaacdafd320bf55489e` |
| 02_Jazz1-130-D_solo | `178728e013d354d2dd8c0818df8dc03cf553d81e6bb43d45a996fc6df5652433` | `5f93ac1ad29010d98348ec1ef30bf075ed31f77fa9c7261788a02fc02dba66b2` |
| 02_Rock1-130-A_comp | `4fc7ab1c7da41459abe8c30d9ae70c944affdc5398dba171d08197add4b0e608` | `fca5f9e779762fd4084d0eb015a4281ad858a8aae21c05e9fad76ce9ca6aa22e` |
| 02_Rock1-130-A_solo | `567cb38a651c2ea950e155a9d31a9d93305983b20573e3fe88723d8d8f66afd2` | `8bd8efd12c095c40be6791bdd74b56e02bb015a556c02d9677a840d22eecbfcd` |
| 02_SS1-100-C#_comp | `2010d893d0ee8c5aa8ed64eb9890f52c2bacb46d74878936002ae6d93a5c082a` | `b5f238ad900ad97a92d35d7eb4a210f10ba98c1b6f9847d8d38795ef417a6505` |
| 02_SS1-100-C#_solo | `1866e928ccc36ab697754b30bbfc293e603deb445362372472e5b01f320f9623` | `d323a965eb92ed360aaa3038f55f8694c690b6331aad9d7bf1b17a9ed7583fb2` |
| 03_BN1-129-Eb_comp | `8fef5aa0807d145e57c5a4befdc729cf3f185cfcc41c64fee7ead7cb1876b629` | `f41001c65b9b3b62187ed3aace924d01773ec41afe8aceab878af72e0009ece0` |
| 03_BN1-129-Eb_solo | `51ec6f1d43b4866a02ca5ff43bb5cfcb7f2b8069fe12b078385e091a8f8d78f4` | `44dd4061b27bb3320d1e42180f5602e71932be8c388945f8b06a46d2df6d7aef` |
| 03_Funk1-114-Ab_comp | `796ccebfb5265044b6f8b33b84d2b2004f7dfa9b2b44b28dd0de65e45c257f0d` | `194ddead88ec8d96bf7ae28bbaa17678287757d8b9eb1aa1e8f65e5584feebde` |
| 03_Funk1-114-Ab_solo | `f69c5fb997091ffb63e265c1a99bf7f1b5bc39dcb1d53b9879f747d0b5d17ebb` | `5b833751007206b4c2418f7e1ec250a09041bffcbafff44a8c4dc6a762ab7432` |
| 03_Jazz1-130-D_comp | `c02e73454f0cfcc2fb7c1a3eba799627ee3787964385041fb1ccd847c735903c` | `0700f585fc0461d7a4d64379c3ad8acaa0961fc93f6960627552bed5ed0a79bc` |
| 03_Jazz1-130-D_solo | `088c1afa042e7b860228b1a74015cdd7f6bb59c44cc25e8cc70c17047c32a802` | `16f13c05c317abc57386e2614b3919a98512d9951af47985311640fc7c82512f` |
| 03_Rock1-130-A_comp | `5e71387daf08213651f8b8f0d9b1db7d589cbc6b9fefd8e052e13a1f8f7286ca` | `502ab4f7788bad71bafe9572ba185e698b3cf82576be105acd6d4112b711f1fa` |
| 03_Rock1-130-A_solo | `eb0175a92ae22478d64060e5c72043ad8fa58d7c75ee519d1558746db3c1d5b1` | `0b297ab7bfa569c6614f5bcb6710662c0c48b0cd42e3fe57c067be61f087c4db` |
| 03_SS1-100-C#_comp | `8cfa48ee4515f40d1ac1329b532e4421f56b471d592b5c7d0acc1b173e633876` | `ca54e877739c8eeacc865ec95bf2c09d1bcaad12e2c2d578a346c791e1e2654d` |
| 03_SS1-100-C#_solo | `26913110e0daa319160ed395eda4fe77ee3cd589c335b0da59449c97114c410d` | `74a2594c91af47e47407a9bb8d7b997e6e98083cf1e1169358353870edd16571` |
| 04_BN1-129-Eb_comp | `ca4b487760400edabb87ed478a523ad0587c020b8463f97429af4009789f4ebd` | `81c4818b93f86a45a582d1b958837b5579c599a2b0a214e9025c63981c87538a` |
| 04_BN1-129-Eb_solo | `3003186fc9122d51d761c1b2ef2591d528c51aaaf991fa2a885f500b4c97bcaf` | `117e877427a2d363f8060e5731b3be4f73b94de5af0e1094bd38a73a84539c4b` |
| 04_Funk1-114-Ab_comp | `6e2a6fc4fdce9b250ab11a090a8f5c4efcc12f83a60b17c8ca206ffeda07a8f0` | `6999adca5d88869c5cd72539767598c4cfe671f768ff2f8ce8f8a68fc4902dff` |
| 04_Funk1-114-Ab_solo | `baf346f568302a185484de83a2ec0a273b560eed501c16fed3f633e02c12dc4c` | `388c93e6cb86e48c50a1745fff0db3b77f204c9596ae9d8927d8d5c4d4a12c47` |
| 04_Jazz1-130-D_comp | `69a9b13b9b4206c7b05ababa7dc4f63027212596aa9d255cac9301487099c899` | `a895d5a9d077e060a832ed9a9e4b33f599b840180d894f91edb6591021ae1185` |
| 04_Jazz1-130-D_solo | `71243666f41a2f1f0c68da42733348891f20677e0624a77bf01d9554bd2d2afb` | `0a6e1e1866ed1b35bf226b379a7e31d3c0a953d3c4812c960ee1f41d47f3acee` |
| 04_Rock1-130-A_comp | `139d6293709a89d256105fc65850129754de98b8d076e4eb1da8f706af80a26a` | `9654afe844b4e8e43d0c992ab7598abe66a42b6ec46771f49d6e33e1675a1a6b` |
| 04_Rock1-130-A_solo | `ae20b3c54b9c5b2b001eb6acf07efeab26faee270cdbec238d62bc4055d51a2e` | `3ac77ff9ffdbd32b2b6f322cfd8de4a39b87fe029d2576168d9787f19a480d35` |
| 04_SS1-100-C#_comp | `39b592e5ab79e457a7901e865b4320e04014903465acdebd64235b4f44416650` | `9cee71992ad50e4aca49266702a4c92098a9299bc4051c3daf7aed160ec5914c` |
| 04_SS1-100-C#_solo | `753a0b4d27b68ed362a1c509bb14300dd23f2c5bf2a8513c76c28006aa408543` | `fe955570ebba65ee8d0b1dba965bada2832bdb889f386844e694b87c2cf54088` |
| 05_BN1-129-Eb_comp | `a55c99690ece8135baa9b05ffc42ee13ee408bcd3c2b0ef01fe373f69e844c3a` | `e79cad99fdcc086ae3faf75e0d91c736d2baab98429b66973095f819b7f7b24d` |
| 05_BN1-129-Eb_solo | `ccf801f3a6e7b1ab38dd9bf1ae917d13c51ef93c977af41cdabcb0548f55a261` | `8b9c47dae61790596142ffc898995a82d066229fa5b2947facc4a83750a921e1` |
| 05_Funk1-114-Ab_comp | `63300b35df98320564fda1505df27b55bc519707728c1185e5b34da0e478b097` | `10b1126a6ab6d8638b366baff9a9a70c2ef4a4334488c1b2bd8dc63c4d9cbf77` |
| 05_Funk1-114-Ab_solo | `1bc53f0d2132e1d40efe3db88ae7c74ef53d256ed60a98dbed77270537307a1f` | `96dfafa94b6e96aecd6fcbd880d4d6b136f58ba7a76f6e903dbc2497b3c91f5b` |
| 05_Jazz1-130-D_comp | `dc0d6d6351c1bf82ec4d186d93306b4f0bef6189ed691ec0ba1b57a74e6365a7` | `c31cd7420ffbcdd18cb401e08634e0e6965a972c4d804717c6b8de0fac01b57e` |
| 05_Jazz1-130-D_solo | `957fd5a4a98c8b9f804b44a664e521dd6268c4e892cce1a0eeb5364dec52afa9` | `a29056c9fb0e6b4bee228cc6f17480e93750a54f375cbcc87edd4bef7b8e5c30` |
| 05_Rock1-130-A_comp | `711b2cc0a44a3890bf028c282e8edbc3f8ebfc2ec35a3806524abe7537c3db19` | `49a8b51c72e3adc1affbcb41dd36978bd088d4f18c98a761f005d403d8bcc174` |
| 05_Rock1-130-A_solo | `9d66f4912edf54047b3753d4b14c217accf8868944c426d9aaf33e465bab0c8c` | `06f4129018f0be6367ea6262a1e4ff6b1eaeb85f6879e5a5bc52d56dcb1c43b2` |
| 05_SS1-100-C#_comp | `a8299b8c499cc67eb2e81e9e3afcc8382680d8a51150ce928407bb492e39c5fd` | `199a6a8ac159c11eeaaae75cedb51d0e1ab4a383eaf03d3763d9b81167dfba65` |
| 05_SS1-100-C#_solo | `7e1f99d1d851130cc9c8be6890afcf2ab5d09849240a242de2d90d99b1f7ed9b` | `d5bb7ae6b49f5dde00101ce39f221125365486386b6f804b2564c7701412f77c` |
