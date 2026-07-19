# MTRD-MNI-CIM Code Ocean Capsule

This branch contains the exported Code Ocean version of MTRD-MNI-CIM, including
the source code, pinned Python dependencies, NeuroSim sources, and reproducibility
metadata. The original reference implementation remains on the [`main`](https://github.com/RL-MIND/MTRD-MNI-CIM/tree/main) branch.

All commands below are run from the repository root.

## 1. Environment setup

The recommended platform is Linux x86-64 with an NVIDIA GPU. CPU execution is
supported for checks and small smoke runs, but complete training is much slower.

```bash
git clone --branch code-ocean --single-branch \
  https://github.com/RL-MIND/MTRD-MNI-CIM.git
cd MTRD-MNI-CIM

conda create -n mtrd python=3.10 -y
conda activate mtrd

python -m pip install --require-hashes \
  -r environment/requirements-bootstrap.txt
python -m pip install --require-hashes \
  -r environment/requirements-lock.txt
```

For NVIDIA GPU training, replace the CPU PyTorch packages with the locked CUDA
packages:

```bash
python -m pip install --require-hashes --no-deps \
  -r environment/requirements-gpu-lock.txt
```

Set the capsule paths and verify the runtime:

```bash
export PYTHONPATH="$PWD/code"
export NEUROSIM_HOME="$PWD/code/NeuroSim"
export MTRD_CIFAR_ROOT="$PWD/data/paper/datasets/cifar"
export MTRD_RESULTS_ROOT="$PWD/results"
export MTRD_CHECKPOINT_ROOT="$MTRD_RESULTS_ROOT/classification/cim/checkpoints"

mkdir -p "$MTRD_CIFAR_ROOT" "$MTRD_RESULTS_ROOT"
python -m capsule.run status
```

Inside Code Ocean, use its `/code`, `/data`, and `/results` mounts for these
three locations and set `PYTHONPATH=/code`. The Code Ocean **Run** button calls
`bash /code/run`, which launches the complete default CIFAR-10 RRAM pipeline;
use `python -m capsule.run` for the individual commands below.

## 2. CIFAR-10

The following one-epoch commands are a quick pipeline check. `--download`
downloads the official torchvision CIFAR-10 files when they are not present.

```bash
python -m capsule.run classification cim train-teachers \
  --dataset cifar10 \
  --device-type rram \
  --data-root "$MTRD_CIFAR_ROOT" \
  --checkpoint-root "$MTRD_CHECKPOINT_ROOT" \
  --device cuda \
  --download \
  --epochs 1 \
  --allow-nonpaper-training

python -m capsule.run classification cim train-student \
  --dataset cifar10 \
  --device-type rram \
  --data-root "$MTRD_CIFAR_ROOT" \
  --checkpoint-root "$MTRD_CHECKPOINT_ROOT" \
  --device cuda \
  --download \
  --epochs 1 \
  --allow-nonpaper-training
```

## 3. CIFAR-100

CIFAR-100 uses the same workflow and automatically selects a 100-class VGG16:

```bash
python -m capsule.run classification cim train-teachers \
  --dataset cifar100 \
  --device-type rram \
  --data-root "$MTRD_CIFAR_ROOT" \
  --checkpoint-root "$MTRD_CHECKPOINT_ROOT" \
  --device cuda \
  --download \
  --epochs 1 \
  --allow-nonpaper-training

python -m capsule.run classification cim train-student \
  --dataset cifar100 \
  --device-type rram \
  --data-root "$MTRD_CIFAR_ROOT" \
  --checkpoint-root "$MTRD_CHECKPOINT_ROOT" \
  --device cuda \
  --download \
  --epochs 1 \
  --allow-nonpaper-training
```

These are smoke runs, not paper results. For the complete training schedule,
remove `--epochs 1` and `--allow-nonpaper-training`; the defaults are 200 epochs
for teachers and 300 epochs for the MTRD student. Use `--device cpu` only when a
CUDA device is unavailable.

To train for PCM instead of RRAM, repeat the selected dataset commands with
`--device-type pcm`. Keep RRAM and PCM checkpoints in their generated separate
directories.

## 4. Device simulation

First record the available simulator versions and capabilities:

```bash
python -m capsule.run simulate status --neurosim-root "$NEUROSIM_HOME"
```

### RRAM accuracy with the NeuroSim functional model

Check the VGG16 functional adapter, then evaluate either dataset. The example
uses one trial for a quick check; use `--trials 20` for the configured full run.

```bash
python -m capsule.run simulate neurosim-functional \
  --neurosim-root "$NEUROSIM_HOME" \
  --model vgg16

DATASET=cifar10  # change to cifar100 when required
ROLE_MANIFEST="$MTRD_CHECKPOINT_ROOT/$DATASET/rram/checkpoint-roles.generated.json"

python -m capsule.run classification cim evaluate \
  --dataset "$DATASET" \
  --device-type rram \
  --backend neurosim \
  --data-root "$MTRD_CIFAR_ROOT" \
  --checkpoint-root "$MTRD_CHECKPOINT_ROOT" \
  --checkpoint-role-manifest "$ROLE_MANIFEST" \
  --checkpoint-role both \
  --method-realization-policy paired \
  --realization-scope fixed-trial \
  --trials 1 \
  --quantization-bits 8 \
  --neurosim-home "$NEUROSIM_HOME" \
  --output-dir "$MTRD_RESULTS_ROOT/evaluation"
```

### PCM accuracy with AIHWKit

This requires checkpoints trained with `--device-type pcm`:

```bash
python -m capsule.run simulate aihwkit-probe

DATASET=cifar10  # change to cifar100 when required
ROLE_MANIFEST="$MTRD_CHECKPOINT_ROOT/$DATASET/pcm/checkpoint-roles.generated.json"

python -m capsule.run classification cim evaluate \
  --dataset "$DATASET" \
  --device-type pcm \
  --backend aihwkit-additive \
  --data-root "$MTRD_CIFAR_ROOT" \
  --checkpoint-root "$MTRD_CHECKPOINT_ROOT" \
  --checkpoint-role-manifest "$ROLE_MANIFEST" \
  --checkpoint-role both \
  --method-realization-policy paired \
  --realization-scope fixed-trial \
  --trials 1 \
  --quantization-bits 8 \
  --output-dir "$MTRD_RESULTS_ROOT/evaluation"
```

### NeuroSim circuit-level PPA

The commands above report task accuracy under device non-idealities. NeuroSim
circuit-level power, performance, and area (PPA) simulation is separate:

```bash
python -m capsule.run simulate neurosim-build \
  --neurosim-root "$NEUROSIM_HOME" \
  --jobs 4

python -m capsule.run simulate neurosim-smoke \
  --neurosim-root "$NEUROSIM_HOME" \
  --timeout-seconds 60 \
  --output-dir "$MTRD_RESULTS_ROOT/neurosim/smoke"
```

For custom network CSV and layer-matrix inputs, see
[`code/SIMULATORS.md`](code/SIMULATORS.md) and use the same
`python -m capsule.run simulate` command prefix shown here. Generated
checkpoints, evaluation CSVs, summaries, and provenance manifests are written
below `results/`.
