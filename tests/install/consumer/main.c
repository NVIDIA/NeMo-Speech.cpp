// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <stdio.h>

#if defined(HAVE_NEMO_SPEECH_ASR)
#include "nemo_speech/asr.h"
#endif
#if defined(HAVE_NEMO_SPEECH_DIAR)
#include "nemo_speech/diar.h"
#endif
#if defined(HAVE_NEMO_SPEECH_NMT)
#include "nemo_speech/nmt.h"
#endif
#if defined(HAVE_NEMO_SPEECH_TTS)
#include "nemo_speech/tts.h"
#endif

int
main(void) {
#if defined(HAVE_NEMO_SPEECH_ASR)
    const char* version = nemo_speech_asr_version();
    if (version == NULL || version[0] == '\0') {
        fprintf(stderr, "installed ASR library returned an empty version\n");
        return 1;
    }
    printf("ASR: %s\n", version);
#endif
#if defined(HAVE_NEMO_SPEECH_DIAR)
    printf("Diarization: %s\n", nemo_speech_asr_version());
#endif
#if defined(HAVE_NEMO_SPEECH_NMT)
    const char* nmt_version = nemo_speech_nmt_version();
    if (nmt_version == NULL || nmt_version[0] == '\0') {
        fprintf(stderr, "installed NMT library returned an empty version\n");
        return 1;
    }
    printf("NMT: %s\n", nmt_version);
#endif
#if defined(HAVE_NEMO_SPEECH_TTS)
    const char* tts_version = nemo_speech_tts_version();
    if (tts_version == NULL || tts_version[0] == '\0') {
        fprintf(stderr, "installed TTS library returned an empty version\n");
        return 1;
    }
    printf("TTS: %s\n", tts_version);
#endif
    return 0;
}
