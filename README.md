# SPINNV2

SPINNV2 is a clean-room prototype for a satellite-oriented portable inference framework and model translation toolchain.

The project follows an ahead-of-time deployment route:

```text
ONNX -> SIR -> SPK -> lightweight Runtime / generated C
```

Current status: M2 static memory planning.

## M0 Smoke Checks

```bash
python -m spinnv2.compiler --help
python -m spinnv2.compiler --print-target cpu_ref
pytest tests/compiler
pytest tests/e2e
cmake -S runtime -B build/runtime
cmake --build build/runtime
ctest --test-dir build/runtime
```

The compiler can also emit a memory-plan CSV:

```bash
python -m spinnv2.compiler compile model.onnx -o build/model.spk --memory-plan-csv build/memory_plan.csv
```

If `pytest` is unavailable, the compiler unit tests can still run with:

```bash
python -m unittest discover -s tests/compiler -p 'test*.py' -v
```
