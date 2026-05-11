# SPINNV2

SPINNV2 is a clean-room prototype for a satellite-oriented portable inference framework and model translation toolchain.

The project follows an ahead-of-time deployment route:

```text
ONNX -> SIR -> SPK -> lightweight Runtime / generated C
```

Current status: M0 engineering skeleton.

## M0 Smoke Checks

```bash
python -m spinnv2.compiler --help
python -m spinnv2.compiler --print-target cpu_ref
python -m unittest discover -s tests/compiler -p 'test*.py' -v
cmake -S runtime -B build/runtime
cmake --build build/runtime
ctest --test-dir build/runtime
```

If `pytest` is installed, the compiler tests are also compatible with:

```bash
pytest tests/compiler
```
