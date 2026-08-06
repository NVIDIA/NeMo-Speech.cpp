# Pascal fork WAV fixture

This directory reserves `teste-en.wav` for a small English ASR functional test and a repeatable
file-based benchmark. Its expected transcript is stored in `teste-en.txt`:

```text
Ask not what your country can do for you. Ask what you can do for your country.
```

## Publication status: pending license review

The proposed source fixture is a maintainer-local WAV. Its original source and redistribution
license could not be confirmed from the available repository history or file metadata. Therefore
`teste-en.wav` is deliberately **not included** in this fork and must not be committed, released,
or represented as redistributable until a manual license review has approved it.

The scripts already expect this path:

```text
test_files/fork/asr/teste-en.wav
```

After approval, copy the original file byte-for-byte and compare SHA-256 values before adding it.
The maintainer-local candidate measured as follows; these values are provided for manual review,
not as a redistribution grant:

| Property | Candidate value |
| --- | --- |
| WAV encoding | PCM signed 16-bit little-endian |
| Sample rate | 24,000 Hz |
| Channels | 1 (mono) |
| Duration | 3.845083 s |
| SHA-256 | `148B936B43CE7C546A866E64DA059F0458AEE2D65E617F16E9D94F06E8D99ED6` |
| Origin | Maintainer-local candidate; source/license pending review |

It is intended only for functional testing and reproducible benchmarking once its legal status is
verified. Do not invent a license for this recording. A reviewer should document the actual source,
license, and any attribution requirements before it is added to the repository.

## Short English benchmark fixture

`short-en.wav` is reserved for the two-to-three-second HTTP latency benchmark. No suitable short
English WAV with a verified redistribution license was found in the local project material, so it is
not included and `short-en.txt` is intentionally not created. Before adding one, document its exact
phrase, duration, PCM16/16 kHz/mono format, SHA-256, source, license, and attribution here.
