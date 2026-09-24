# Diarization fixtures

`ami_en2002d_2132.wav` is 60 s of meeting EN2002d (Mix-Headset, from 2132.5 s)
from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The excerpt is
unmodified 16 kHz mono PCM. Three speakers take turns, with brief overlap; a
fourth says only a few words.

`ami_en2002d_2132.rttm` is the matching reference, cut from the word-aligned
`only_words` RTTM of [AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup)
(Apache-2.0), with times relative to the excerpt.

`ami_en2002d_2132.json` is the reference transcript by speaker, taken from the
AMI manual word annotations (CC BY 4.0) for the same excerpt.
