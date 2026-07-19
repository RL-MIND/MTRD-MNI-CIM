This [Code Ocean](https://codeocean.com) Compute Capsule will allow you to reproduce the results published by the author on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information. Don't hesitate to reach out to [Support](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- [nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu) for code that leverages the GPU
- Public datasets (see [Instructions](#download-public-datasets))
- MATLAB/MOSEK/Stata licenses where applicable

# Instructions

## Download public datasets

In order to fetch the dataset(s) this capsule depends on, first install the [AWS Command Line Interface](https://aws.amazon.com/cli/), then navigate to the folder where you've extracted the capsule in a terminal and run the following commands:
```shell
cd data
# Fetch dataset "CIFAR-10"
aws s3 cp --no-sign-request --recursive s3://codeocean-datasets/0b1da330-69b8-45b5-9a87-7623febd4250 data/CIFAR
cd ..
```
## The computational environment (Docker image)

This capsule is private and uses the base environment (without any customization). The base environment is available on Code Ocean's Docker registry:
`registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04`

## Running the capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm --gpus all \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04 bash run
```
