# Immutable Data Assets

The repository does not include datasets or trained checkpoint binaries.
Attach them to Code Ocean as immutable, read-only Data Assets mounted at
`/data`. Write generated checkpoints and results to `/results`.

## Recommended Layout

```text
/data/
|-- paper/
|   |-- datasets/
|   |   |-- cifar/
|   |   |   |-- cifar-10-batches-py/
|   |   |   |-- cifar-100-python/
|   |   |-- denoising/
|   |   |   |-- berkeley400/
|   |   |       |-- train/        # exactly 400 images
|   |   |       |-- Set12/        # exactly 12 images
|   |   |   |-- dncnn-source-h5/  # optional immutable source artifacts
|   |   |       |-- train.h5
|   |   |       |-- val.h5
|   |   |-- carvana/
|   |       |-- imgs/
|   |       |   |-- train_hq/
|   |       |-- masks/
|   |           |-- train_masks/
|   |-- checkpoints/
|   |   |-- fig4/
|   |   |-- fig5/
|   |       |-- classification/
|   |       |-- denoising/
|   |       |-- segmentation/
|   |-- manifests/
|   |   |-- paper-assets.json
|   |   |-- checkpoint-roles.json
|   |   |-- carvana-author-confirmed-split.json
|   |   |-- paper-fig5-dncnn-unet.json
|   |-- raw-reference/
|       |-- author-generated CSV files, when released
|-- neurosim/
    |-- network.csv
    |-- layer0_weight.csv
    |-- layer0_input.csv
```

Only the directories needed by a selected workflow are required.

## Environment Variables

```bash
export MTRD_DATA_ROOT=/data
export MTRD_RESULTS_ROOT=/results
export MTRD_CIFAR_ROOT=/data/paper/datasets/cifar
```

Workflow-specific command-line paths take precedence over these variables.

## Upload and Verify

Do not add datasets to the source capsule. Upload each dataset as an immutable
Code Ocean Data Asset, mount it read-only at `/data`, then run the matching
asset preflight below. Each command writes a machine-readable report under
`/results` and exits nonzero when its required raw-data contract fails.

```bash
bash code/run assets cifar \
  --dataset cifar100 \
  --data-root /data/paper/datasets/cifar \
  --output /results/preflight/cifar100.json

bash code/run assets denoising \
  --berkeley-root /data/paper/datasets/denoising/berkeley400 \
  --set12-dir /data/paper/datasets/denoising/berkeley400/Set12 \
  --output /results/preflight/berkeley400-set12.json

bash code/run assets carvana \
  --image-dir /data/paper/datasets/carvana/imgs/train_hq \
  --mask-dir /data/paper/datasets/carvana/masks/train_masks \
  --output /results/preflight/carvana.json
```

The CIFAR command checks torchvision's required extracted files against their
official MD5 values and records a tree SHA-256 by default. The denoising
command requires 400 Berkeley training PNGs and 12 Set12 PNGs, rejects
duplicate train/test image content, and records content identities. The
Carvana command verifies image/mask pairing and requires 5,088 labeled pairs.
It deliberately does not infer a paper split from directory names.

To validate a supplied author split as well as the raw Carvana asset, add:

```bash
bash code/run assets carvana \
  --image-dir /data/paper/datasets/carvana/imgs/train_hq \
  --mask-dir /data/paper/datasets/carvana/masks/train_masks \
  --split-manifest /data/paper/manifests/carvana-author-confirmed-split.json \
  --require-paper-split \
  --output /results/preflight/carvana-paper-split.json
```

`--require-paper-split` accepts only an `author_verified=true` manifest. A
derived split can be checked with `--allow-derived-split`, but it remains an
explicitly non-paper reconstruction run.

## CIFAR-10 and CIFAR-100

The CIFAR root is the directory containing torchvision's extracted folders:

```text
/data/paper/datasets/cifar/cifar-10-batches-py
/data/paper/datasets/cifar/cifar-100-python
```

Figure 4 verifies the required CIFAR-10 batch files and records their
individual and aggregate SHA-256 values. Figure 5 classification records the
selected dataset tree identity.

The current training protocol reuses the official CIFAR test split for
epoch-level Eq. (6) feedback/checkpoint selection and final evaluation. This
is not an independent holdout and must be disclosed.

## Berkeley400 and Set12

The DnCNN workflow requires:

- exactly 400 source images in `berkeley400/train`; and
- exactly 12 test images in `berkeley400/Set12`.

The preflight hashes names and contents, rejects train/test duplicate content,
and records preprocessing provenance. DnCNN uses `cv2.imread` and BGR channel
index 0 for both Set12 feedback and evaluation. Its HDF5 training preprocessor
also preserves the released literal `cv2.resize` dsize expression
`(int(height * scale), int(width * scale))`; do not normalize that expression
to OpenCV's conventional width/height order.

The experiment JSON must explicitly choose one HDF5 provenance mode under
`data.denoising_h5`:

```json
{
  "mode": "regenerated-from-raw",
  "directory": "/results/paper_fig5_dncnn_unet/denoising_h5"
}
```

This mode regenerates `train.h5` and `val.h5` from the uploaded PNGs and writes
a provenance manifest under the configured `/results` directory. It is the
default in both public Figure 5 templates.

```json
{
  "mode": "source-provided",
  "train_h5": "/data/paper/datasets/denoising/dncnn-source-h5/train.h5",
  "val_h5": "/data/paper/datasets/denoising/dncnn-source-h5/val.h5",
  "train_h5_sha256": "5b32b18d1591a4af56b745343e668c5c938b0ea85dec360b128e520788181bdf",
  "val_h5_sha256": "7061b217e6aeac5d5932baa1b61c367b1e2b5c351f6e230b89616518a55be516"
}
```

Use `source-provided` only after uploading an immutable historical HDF5 pair.
The shown digests identify the audited released pair; replace both values only
when a different source artifact has its own recorded provenance. The workflow
requires both digests or neither, validates configured digests exactly when
present, validates the released flat sequential-dataset layout, validates every
sample shape and `float32` dtype, derives the expected training patch count
from the mounted Berkeley images, and checks that each `val.h5` tensor exactly
matches its mounted Set12 BGR-channel tensor. It never writes a
`source-provided` input; `train --overwrite-h5` is rejected in that mode.

Both modes still require the Berkeley400 and Set12 PNG assets because they bind
the HDF5 artifact to the data asset. Evaluation reads Set12 directly and does
not open training HDF5 files. No HDF5 artifact is bundled in this capsule.

## Carvana

The UNet workflow indexes the 5,088 labeled pairs in the official
`train_hq`/`train_masks` asset by canonical sample ID. The unlabeled official
Carvana test set is not part of this protocol. The workflow does not infer the
paper split from directory names.

The split manifest must have this shape:

```json
{
  "schema_version": 1,
  "author_verified": true,
  "train_ids": ["... exactly 4700 unique IDs ..."],
  "test_ids": ["... exactly 318 unique IDs ..."],
  "excluded_ids": ["... exactly 70 unique IDs ..."]
}
```

The three lists must be disjoint and must partition all 5,088 paired samples.
The test list must contain one image for each of 318 cars. The 70 excluded IDs
are explicit because the manuscript's stated 4,700 training and 318 test
samples sum to 5,018 rather than the stated total of 5,088.

No author-confirmed split is bundled. Do not set `author_verified` to true
without author confirmation. Without that asset, exact Carvana numerical
reproduction is blocked.

## Checkpoints

Every external checkpoint must have:

- an unambiguous role;
- a path relative to the data mount;
- an exact byte size;
- a SHA-256 digest; and
- the architecture, dataset, device model, noise level, and training-protocol
  identity that produced it.

Use `metadata/checkpoint-role-manifest.template.json`. A matching filename is
not sufficient evidence that a checkpoint belongs to a published curve.

### Figure 4 generated layout

```text
/results/paper/fig4/checkpoints/cifar10/vgg16/
|-- ckpt_clean.pth
|-- ckpt_rram_0.1.pth ... ckpt_rram_0.5.pth
|-- ckpt_pcm_0.02.pth ... ckpt_pcm_0.1.pth
|-- mtrd_rram_<balancing-policy>_S0.3.pth
|-- mtrd_pcm_<balancing-policy>_S0.06.pth
```

### Figure 5 classification generated layout

```text
<checkpoint-root>/<dataset>/<device-type>/
|-- teacher_clean.pt
|-- teacher_<device-type>_<noise>.pt
|-- student_mtrd_<device-type>.pt
```

### Figure 5 image generated layout

```text
<checkpoint-root>/<task>/clean/teacher.pth
<checkpoint-root>/<task>/<device-model>/teacher_<noise>.pth
<checkpoint-root>/<task>/<device-model>/mtrd.pth
```

`task` is `denoising` or `segmentation`. `device-model` is `rram` or `pcm`.

Legacy checkpoint conversion, if required, must be strict. Never use
`strict=False` to hide missing or unexpected model keys. Retain both source
and converted SHA-256 values.

## Portable CIFAR Checkpoints

Figure 5 CIFAR training writes portable raw PyTorch `state_dict` files for
each selected clean teacher and MTRD student, plus an adjacent JSON manifest.
The raw weight file contains no optimizer state and can be attached later as a
read-only Data Asset. Test one mounted CIFAR-10 or CIFAR-100 weight directly
on the complete official test split with:

```bash
bash code/run fig5-classification test-checkpoint \
  --dataset cifar100 \
  --data-root /data/paper/datasets/cifar \
  --checkpoint /data/paper/checkpoints/fig5/classification/cifar100/rram/student_mtrd_rram.pt \
  --normalization-profile dataset-native \
  --device cuda \
  --output-dir /results/checkpoint-test/cifar100-rram-mtrd \
  --require-checkpoint-manifest
```

The command strict-loads the VGG16 architecture, writes one prediction row per
test image, and records accuracy, data identity, weight SHA-256, normalization,
and runtime provenance. Omit `--require-checkpoint-manifest` only when
inspecting a legacy or externally supplied weight whose training provenance is
not available; strict loading then proves compatibility, not paper-curve role
or training provenance. This clean PyTorch check is separate from formal
NeuroSim/AIHWKit evaluation, which still requires a checkpoint-role manifest.

The historical project checkpoint folders are deliberately not copied into
`/code`. They contain several gigabytes of candidate and intermediate weights,
including multiple files that could plausibly fill the same nominal role.
Publish the selected subset as an immutable `/data` asset only after the author
confirms the checkpoint-to-panel mapping. A matching architecture and a
successful strict load prove compatibility, not provenance.

## Asset Publication Checklist

Before publishing a numerical result:

1. pin the Code Ocean Data Asset ID and version;
2. populate `metadata/paper-assets.template.json` outside the source tree;
3. populate every checkpoint role and hash;
4. retain the exact protocol JSON and its SHA-256;
5. run the workflow preflight;
6. store generated data only in `/results`; and
7. archive raw rows, summaries, manifests, and logs together.

Dataset and checkpoint licenses are controlled by their respective providers.
This repository does not grant additional rights to external assets.
