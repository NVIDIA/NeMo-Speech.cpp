# NMT models and conversion

The NMT pipeline runs a **Riva-Translate** decoder (a Mistral/Llama-architecture
text model) through llama.cpp. The model is a standard Hugging Face checkpoint,
not a NeMo checkpoint; the root converter delegates this architecture to the
pinned llama.cpp converter.

Hugging Face: [nvidia/Riva-Translate-4B-Instruct-v2](https://huggingface.co/nvidia/Riva-Translate-4B-Instruct-v2)

Unlike the NeMo model converters, NMT conversion requires the pinned llama.cpp
submodule and its additional Python dependencies. Conversion is run from a
source checkout; the Python tools are not included in native release archives.
See [Model conversion](../model-conversion.md) for the base environment setup.

```bash
git submodule update --init llama.cpp
python3 -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python3 convert_model.py nvidia/Riva-Translate-4B-Instruct-v2 \
    --outfile riva-translate-4b-instruct-v2.q8_0.gguf --outtype q8_0
```

## Notes

- The language pair is selected per request, not baked into the model: see
  [configuration](configuration.md).
- The underlying llama.cpp converter may print a `fix_mistral_regex` warning
  while reading the Hugging Face tokenizer. It is benign for this model.

## Precision

The converter accepts `f32`, `f16`, `bf16`, and `q8_0`; f16 and q8_0 are the
tested runtime precisions on CPU and GPU backends, and q8_0 is smaller and the
example-config default.

Once converted, point the server at it: see [NMT configuration](configuration.md).
