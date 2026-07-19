# Code Ocean Environment

## Base Image

The Dockerfile starts from the project-specified Code Ocean image:

```dockerfile
# hash:sha256:1358ae631cea9e003084cc9c733c5879b67872978be4e594a41dba1219af31d9
FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8
```

The `# hash:sha256:...` line is Code Ocean environment metadata. Standard
Docker treats it as a comment, and the `io.codeocean.image.base.hash` label in
this Dockerfile records the same metadata rather than enforcing an OCI digest.
Therefore the local
`docker build` command below resolves the mutable `FROM` tag at build time; it
does not prove that the resulting base matches the recorded Code Ocean hash.

Retain the Code Ocean environment hash and the immutable digest of the built
release image. For a local archival build, also record the resolved base image
ID and repository digest with `docker image inspect`. Use a registry-verified,
digest-qualified `FROM` reference when exact local base-image enforcement is
required. Do not claim that a local build matches the Code Ocean hash merely
because the comment or metadata label is present.

## Pinned Components

The multi-stage build:

- compiles checksum-verified CPython 3.10.15;
- installs an exact CPU PyTorch/torchvision pair in the `runtime` target;
- replaces that pair with hash-locked PyTorch 2.10.0/torchvision 0.25.0
  CUDA 12.6 wheels and enumerated CUDA dependencies in the default
  `gpu-runtime` target;
- installs IBM AIHWKit 1.1.0 from its hash-pinned wheel;
- installs every direct and transitive Python dependency from hash-locked
  requirement files;
- verifies the bundled NeuroSim source archive;
- records NeuroSim commit
  `cddb7d346a9f1fc5a39b6c3abcb378c4b2dfc555`;
- exposes a runtime wrapper that also verifies canonical source-tree SHA-256
  `55fe07c04d37536ac18cdbc4f393506da01d42703096245594c56c0d4012f4d1`;
- compiles `/opt/neurosim/NeuroSIM/main` from source; and
- retains the builder package list, compiler version, and stripped NeuroSim
  binary SHA-256 under `/opt/mtrd/build-provenance/`; and
- omits the compiler toolchain, development headers, CPython source tree, and
  NeuroSim object files from the final runtime stage. Generated Python bytecode
  caches are also omitted; the corresponding Python sources remain present.

The relevant inputs are:

```text
environment/Dockerfile
environment/requirements-bootstrap.txt
environment/requirements-lock.txt
environment/requirements-gpu-lock.txt
environment/neurosim-cddb7d346a9f1fc5a39b6c3abcb378c4b2dfc555.tar.gz
```

The NeuroSim archive digest is stored in `NEUROSIM_ARCHIVE_SHA256` in the
Dockerfile and is checked before extraction.

## Build

From the capsule root:

```bash
docker build --platform linux/amd64 \
  -f environment/Dockerfile \
  -t mtrd-mni-cim environment
```

This builds the final `gpu-runtime` target. A smaller CPU-only simulator and
interface-validation image remains available explicitly:

```bash
docker build --platform linux/amd64 \
  --target runtime \
  -f environment/Dockerfile \
  -t mtrd-mni-cim:cpu environment
```

Then validate the runtime through the public interfaces:

```bash
docker run --rm --platform linux/amd64 \
  --gpus all \
  -v "$PWD/code:/code:ro" \
  mtrd-mni-cim \
  bash /code/run status

docker run --rm --platform linux/amd64 \
  --gpus all \
  -v "$PWD/code:/code:ro" \
  mtrd-mni-cim \
  bash /code/run simulate aihwkit-probe

docker run --rm --platform linux/amd64 \
  -v "$PWD/code:/code:ro" \
  mtrd-mni-cim \
  bash /code/run simulate status

docker run --rm --platform linux/amd64 \
  -v "$PWD/code:/code:ro" \
  mtrd-mni-cim \
  bash /code/run simulate neurosim-smoke \
    --output-dir /tmp/neurosim-smoke
```

In Code Ocean, use its `/code`, `/data`, and `/results` mounts rather than the
local bind-mount example.

The smoke output above is ephemeral. For a retained local result, mount a host
directory at `/results` and place `--output-dir` below that mount.

## Validation and Publication

The frozen source tree includes a public regression suite. Its numerical
publication is conditional on rebuilding the image from the published tree,
rerunning the public suite with the attached immutable assets, regenerating
the release checksum manifest, and recording the immutable Code Ocean image
digest. Local image IDs and mutable base-tag resolutions are not publication
identities.

The final build must record the resolved PyTorch, CUDA, cuDNN, AIHWKit, and
NeuroSim identities. It must also confirm that the runtime image excludes the
build-only compiler toolchain. These checks validate the executable environment
only; they do not verify numerical agreement with the paper.

## CPU and GPU Boundary

The requested base image contains CUDA 11.7 and cuDNN 8. The final
`gpu-runtime` target uses PyTorch 2.10.0+cu126 and torchvision 0.25.0+cu126;
their CUDA 12.6 user-space libraries are installed from the fully hash-locked
`requirements-gpu-lock.txt`. NVIDIA driver compatibility is still a host
requirement. Run `bash code/run status` to record the software and simulator
identity. For every formal training command, first run the workflow-specific
training preflight, verify `cuda_available=true` and the GPU identity in its
environment block, and pass the explicit CUDA device option documented in
`REPRODUCING.md`.

AIHWKit 1.1.0 remains in the same image and is validated against the PyTorch
2.10 ABI. This does not imply that every AIHWKit conversion is CUDA-capable:
the documented fixed-trial PCM workflows retain their validated device and
operator restrictions. The `runtime` target keeps the smaller CPU PyTorch
pair for simulator probes and CPU-only validation.

The GPU target places framework caches below `/tmp`. This is required because
Code Ocean and local Docker commonly launch the image with a numeric UID that
has no entry in the image's passwd database. The final image was explicitly
tested with `--user 1000:1000` and a read-only `/code` mount.

Every formal GPU result must record the image ID or immutable registry digest,
complete package versions, CUDA/cuDNN versions, GPU model, and deterministic
settings. Results generated by another local environment must be reported as a
separate runtime even when the source and configuration are identical.

## Runtime Identity

Before each experiment, retain:

```bash
bash code/run status > /results/runtime-status.json
bash code/run simulate status > /results/simulator-status.json
python -m pip freeze --all > /results/pip-freeze.txt
```

The workflow manifests additionally record package versions, command
arguments, source or configuration identities, dataset identities, and
checkpoint SHA-256 values where supported.

## Network and Long-Term Rebuilds

The NeuroSim source archive is bundled, so its C++ build does not clone the
upstream repository. CPython and Python wheels are still resolved from their
checksum- or hash-locked upstream URLs during a fresh Docker build.

For archival reproduction, retain:

- the successfully built Code Ocean image and immutable digest;
- the source capsule;
- all input wheel/source files or a trusted package mirror;
- the immutable Data Asset version; and
- the complete `/results` provenance bundle.

Version pins alone do not guarantee that external package servers will remain
available indefinitely.

## Local Python Installation

For a compatible CPython 3.10 x86-64 environment:

```bash
python -m pip install -r requirements.txt
bash code/run status
bash code/run simulate aihwkit-probe
```

The Dockerfile remains the authoritative environment definition. A local
installation must be reported separately if its resolved packages differ.
