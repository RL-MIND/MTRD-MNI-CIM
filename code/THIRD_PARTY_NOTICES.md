# Third-Party Notices

The MIT license in the capsule root applies to the first-party MTRD-MNI-CIM
code only. It does not replace the licenses of bundled or installed
third-party components.

## DNN+NeuroSim

- Component: DNN+NeuroSim V1.5 source under `NeuroSim/` and the matching
  archive under `environment/`.
- Upstream project:
  <https://github.com/neurosim/NeuroSim/tree/2DInferenceV1.5-dev>
- Bundled revision:
  `cddb7d346a9f1fc5a39b6c3abcb378c4b2dfc555`.
- Developers: Prof. Shimeng Yu's group, Georgia Institute of Technology, as
  identified by the upstream project.
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0).
- License text:
  <https://creativecommons.org/licenses/by-nc/4.0/legalcode>

The upstream project states that the model is made publicly available on a
non-commercial basis and that copyright is maintained by its developers.
Users are responsible for complying with the attribution and non-commercial
conditions. The root MIT license does not grant commercial rights to
NeuroSim.

The source snapshot retains upstream copyright and license headers. The
`NeuroSim/README.md` file also records the upstream license statement.

## NVIDIA PyTorch Quantization

- Component: the `pytorch-quantization` source bundled inside the NeuroSim
  snapshot at `NeuroSim/pytorch-quantization/`.
- Copyright notice in the bundled license: Copyright 2020 NVIDIA Corporation.
- License: Apache License 2.0.
- Complete bundled license and additional attributions:
  `NeuroSim/pytorch-quantization/LICENSE`.
- License text: <https://www.apache.org/licenses/LICENSE-2.0>

The bundled `LICENSE` file includes additional notices for incorporated
third-party material. Preserve that file when redistributing the NeuroSim
snapshot.

## IBM Analog Hardware Acceleration Kit

- Component: `aihwkit==1.1.0` installed by the Docker environment.
- Upstream project: <https://github.com/IBM/aihwkit>
- License: MIT License, as declared by the upstream project and package.

AIHWKit is installed as a third-party wheel and is not relicensed by this
capsule. Preserve its package metadata and license when distributing a built
environment.

## Python and System Dependencies

PyTorch, torchvision, NumPy, SciPy, Pillow, and the other packages listed in
`environment/requirements-*.txt` retain their respective upstream licenses.
The Code Ocean base image and Ubuntu packages are also governed by their
providers' terms.

A released environment should preserve the installed package metadata and a
complete resolved package inventory. Review all dependency licenses for the
intended form of redistribution; this notice is not a substitute for those
license texts or legal advice.
