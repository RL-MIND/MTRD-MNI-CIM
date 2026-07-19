# Simulator Interfaces

This capsule provides separate interfaces for:

- NeuroSim circuit-level power, performance, and area (PPA) estimation.
- A NeuroSim-source-gated PyTorch RRAM functional extension for supported
  task metrics.
- IBM AIHWKit algorithm-level PCM inference for supported operators.

They are not interchangeable. A circuit PPA measurement is not Accuracy, Dice,
or PSNR. A functional task metric is not a C++ circuit result.

## 1. Common Status

```bash
bash code/run simulate status
```

The JSON response records the detected NeuroSim revision/PPA binary, the
functional-adapter matrix, the native source distribution, explicit Eq. (1)
support, installed AIHWKit version, and runtime-probe status. Retain it with
every formal experiment.

## 2. NeuroSim

### 2.1 Pinned source

The bundled DNN+NeuroSim source is pinned to:

```text
cddb7d346a9f1fc5a39b6c3abcb378c4b2dfc555
```

The Docker build verifies the bundled archive SHA-256, extracts it without a
Git clone, records `NEUROSIM_COMMIT`, and compiles `NeuroSIM/main`. The source
is third-party non-commercial software under CC BY-NC 4.0.

### 2.2 RRAM functional task evaluation

The RRAM functional extension implements the explicit weight model:

```text
W_g = W_nominal * exp(theta), theta ~ N(0, sigma^2)
```

It programs every mapped weight realization once per `fixed-trial` evaluation
and keeps that realization fixed for the complete configured test set. The
manifest records each trial seed, layer ordering, clean/mapped/programmed
hashes, empirical theta statistics, operator coverage, and quantization
profile.

This interface is a source-gated PyTorch extension. It does not call the
upstream native NeuroSim CIM-array kernel. Its manifests set
`paper_backend_match=false`, `platform_match=false`, and
`upstream_native_cim_array_kernel_used=false`. The native upstream sampler uses
a different Gaussian-conductance model and does not provide the VGG16, UNet,
or DnCNN paths implemented by this capsule.

Verify the functional gate before the relevant task evaluation:

```bash
bash code/run simulate neurosim-functional --model vgg16
bash code/run simulate neurosim-functional --model unet
bash code/run simulate neurosim-functional --model dncnn
```

Run task metrics through the task entry points documented in
[REPRODUCING.md](REPRODUCING.md):

```bash
bash code/run classification cim evaluate ...
bash code/run denosing --config /data/manifests/denosing.json evaluate ...
bash code/run segmentation --config /data/manifests/segmentation.json evaluate ...
```

None of these commands invokes the C++ PPA executable.

### 2.3 Circuit PPA capability

The C++ PPA wrapper reports chip/CIM-array area, clock period, per-image
latency, dynamic/leakage energy, throughput, energy efficiency, compute
efficiency, and simulator runtime when the upstream executable emits them.
It does not report task accuracy, Dice, or PSNR.

The slim runtime image contains the already-built executable and intentionally
omits `make`, `g++`, and development headers. Rebuild only on a development or
builder environment with a C++ toolchain:

```bash
bash code/run simulate neurosim-build \
  --neurosim-root "$PWD/NeuroSim" \
  --jobs 4
```

The wrapper requires the pinned commit marker and canonical source-tree
SHA-256 `8d2a0a4db81838bf2e9738c3140871cc09966431d3ac8693acdad70c6e281d61`.
It excludes only compiler artifacts, Python bytecode, and Git metadata from
that identity. A changed `Param.cpp` is a new physical configuration and is
rejected by the pinned wrapper unless it receives a separately versioned
source identity.

### 2.4 PPA smoke test and inputs

Exercise input validation, C++ execution, output parsing, and provenance with:

```bash
bash code/run simulate neurosim-smoke \
  --neurosim-root /opt/neurosim \
  --timeout-seconds 60 \
  --output-dir /results/neurosim/smoke
```

This creates a deterministic 64x128 fixture and requires positive area,
latency, and dynamic energy. It is a circuit-PPA smoke test only.

`--network-csv` has exactly eight numeric columns per non-empty row:

```text
input_height,input_width,input_channels,kernel_height,kernel_width,output_channels,pooling_flag,stride
```

The first six dimensions and stride are positive integers; the pooling flag is
integer `0` or `1`. The wrapper rejects invalid, non-finite, non-integral,
zero-sized, and non-eight-column rows. A network row maps to one ordered pair
of weight and unfolded-input matrices. For `H, W, Cin, Kh, Kw, Cout, flag,
stride`, it requires:

```text
weight: (Cin * Kh * Kw) rows by Cout columns
input:  (Cin * Kh * Kw) rows by
        (((H - Kh + 1) / stride) * ((W - Kw + 1) / stride)) columns
```

Both spatial quotients must divide exactly. Weight values must be finite in
`[-1, 1]`; inputs must be integers in `[0, 2^INPUT_BITS - 1]`.

Run a PPA request with one `--layer WEIGHT INPUT` pair for every network row:

```bash
bash code/run simulate neurosim-ppa \
  --neurosim-root /opt/neurosim \
  --network-csv /data/neurosim/network.csv \
  --layer /data/neurosim/layer0_weight.csv /data/neurosim/layer0_input.csv \
  --synapse-bits 8 \
  --input-bits 8 \
  --subarray-rows 128 \
  --parallel-rows 128 \
  --timeout-seconds 300 \
  --output-dir /results/neurosim/ppa
```

The wrapper records exact command arguments, source/executable/`Param.cpp`
hashes, every input hash/shape, raw simulator output, parsed values, and units
in `neurosim_ppa.json`. `parallel-rows` must be in `[1, subarray-rows]`.

### 2.5 Python API

Put `code/` on `PYTHONPATH` before programmatic use:

```python
from pathlib import Path

from simulators.neurosim import NeuroSimLayerFiles, NeuroSimPPARequest, run_neurosim_ppa

request = NeuroSimPPARequest(
    network_csv=Path("/data/neurosim/network.csv"),
    layers=(
        NeuroSimLayerFiles(
            Path("/data/neurosim/layer0_weight.csv"),
            Path("/data/neurosim/layer0_input.csv"),
        ),
    ),
    neurosim_root=Path("/opt/neurosim"),
    synapse_bits=8,
    input_bits=8,
    subarray_rows=128,
    parallel_rows=128,
)
result = run_neurosim_ppa(request, Path("/results/neurosim/ppa"))
```

## 3. AIHWKit

### 3.1 Runtime probe

The supported version is IBM AIHWKit 1.1.0:

```bash
bash code/run simulate aihwkit-probe
```

The probe verifies the installed distribution, converts a small
Conv2d/Linear model, programs a fixed realization, runs fixed-trial and
per-MAC diagnostic paths, checks finite output, and confirms logical layers
are not split into independently scaled tiles. Only `fixed-trial` is eligible
for formal metrics.

### 3.2 Implemented PCM equation

The supported PCM model is:

```text
W_g = W + Normal(0, (eta * max(W))^2)
```

`max(W)` is the signed maximum over one complete logical layer. AIHWKit uses
`weight_scaling_omega=1.0`, `weight_scaling_columnwise=False`, unsplit logical
layers, and digital bias. Its internal `max(abs(W))` scaling is compensated so
the capsule's effective model-space noise standard deviation remains
`eta * max(W)`. Symmetric 8-bit quantization uses `max(abs(W))` for its own
representable range; that is not PCM `W_max`.

When requested, 8-bit I/O resolution is `1 / (2^8 - 2)` with deterministic
rounding. Input/output analog noise and drift are disabled for this explicit
additive-noise model.

### 3.3 Realization scope and operator coverage

Formal evaluation requires `fixed-trial`: one programmed additive weight
realization frozen across all batches in a trial. The `per-mac` path is a scale
diagnostic only, because AIHWKit 1.1.0 exposes no replayable seed or RNG state
for its forward noise. The workflow rejects it for formal Accuracy, Dice, and
PSNR metrics.

The conversion path supports `Conv2d` and `Linear`, covering VGG16 and DnCNN.
It rejects models containing `ConvTranspose2d`; UNet PCM evaluation therefore
fails closed instead of mixing digital transposed convolutions into a claimed
complete analog mapping.

### 3.4 Python API

```python
from simulators.aihwkit import convert_model, runtime_probe

status = runtime_probe()
if not status["runtime_ready"]:
    raise RuntimeError(status)

analog_model = convert_model(
    digital_model,
    eta=0.06,
    input_bits=8,
    output_bits=8,
    seed=2025,
    realization_scope="fixed-trial",
)
```

Conversion/programming use an isolated PyTorch RNG context. For
`fixed-trial`, `seed` controls the programmed realization without consuming the
caller's ambient CPU RNG stream. Prefer the task commands for complete data,
checkpoint, trial, metric, and manifest handling.

## 4. Reporting Rules

Use these labels precisely:

- **NeuroSim PPA**: output of the pinned C++ executable captured by a PPA
  manifest.
- **NeuroSim-source-gated PyTorch functional extension**: the RRAM Eq. (1)
  task metric path.
- **AIHWKit**: output from a fully converted supported model under the recorded
  AIHWKit configuration.
- **PyTorch equation injection**: training or analysis that does not use either
  simulator backend.

