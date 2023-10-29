<div align="center">

# ULTRA: Towards Foundation Models for Knowledge Graph Reasoning #

[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![arxiv](http://img.shields.io/badge/arxiv-2310.04562-yellow.svg)](https://arxiv.org/abs/2310.04562)
![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

</div>

This is the TorchDrug implementation of [ULTRA], a foundation model for KG reasoning. Authored by [Michael Galkin], [Zhaocheng Zhu], and [Xinyu Yuan]. This repo contains the original code to reproduce the experimental results reported in the paper. The latest maintained version of ULTRA is available in the [PyG version of ULTRA](https://github.com/DeepGraphLearning/ULTRA).

[Zhaocheng Zhu]: https://kiddozhu.github.io
[Michael Galkin]: https://migalkin.github.io/
[Xinyu Yuan]: https://github.com/KatarinaYuan
[Ultra]: https://deepgraphlearning.github.io/project/ultra

## Installation ##

You may install the dependencies via either conda or pip. 
Ultra (TorchDrug) is compatible with Python 3.7/3.8/3.9, PyTorch 1.13 and PyG 2.3 (CUDA 11.7 or later when running on GPUs). If you are on a Mac, you may omit the CUDA toolkit requirements (tested with PyTorch 2.0 with the relevant `torch-scatter` version on Mac M2).

### From Conda ###

```bash
conda install torchdrug pytorch cudatoolkit -c milagraph -c pytorch -c pyg
conda install pytorch-sparse pytorch-scatter -c pyg
conda install easydict pyyaml -c conda-forge
```

### From Pip ###

```bash
pip install torchdrug torch
pip install easydict pyyaml
```

If anything else is missing, install those from `requirements.txt`

<details>
<summary> Compilation of the `rspmm` kernel </summary>

To make relational message passing iteration `O(V)` instead of `O(E)` we ship a custom `rspmm` kernel that will be compiled automatically upon the first launch. The `rspmm` kernel supports `transe` and `distmult` message functions, others like `rotate` will resort to full edge materialization and `O(E)` complexity.

The kernel can be compiled on both CPUs (including M1/M2 on Macs) and GPUs (it is done only once and then cached). For GPUs, you need a CUDA 11.7+ toolkit with the `nvcc` compiler. If you are deploying this in a Docker container, make sure to start from the `devel` images that contain `nvcc` in addition to plain CUDA runtime.

Make sure your `CUDA_HOME` variable is set properly to avoid potential compilation errors, eg
```bash
export CUDA_HOME=/usr/local/cuda-11.7/
```

</details>


## Checkpoints ##

We provide two pre-trained ULTRA checkpoints in the `/ckpts` folder of the same model size (6-layer GNNs per relation and entity graphs, 64d, 168k total parameters) trained on 2 x A100 GPUs with this codebase:
* `td_ultra_3g.pth`: trained on `FB15k237, WN18RR, CoDExMedium` for 200,000 steps, config is in `/config/transductive/pretrain_3g.yaml`
* `td_ultra_4g.pth`: trained on `FB15k237, WN18RR, CoDExMedium, NELL995` for 400,000 steps, config is in `/config/transductive/pretrain_4g.yaml`

You can use those checkpoints for zero-shot inference on any graph (including your own) or use it as a backbone for fine-tuning.

## Run Inference and Fine-tuning

The `/scripts` folder contains 2 executable files:
* `run_full.py` - run an experiment on a single dataset and/or pre-training;
* `run_many.py` - run experiments on several datasets sequentially and dump results into a CSV file.

The yaml configs in the `config` folder are provided for both `transductive` and `inductive` datasets.

### Run a single experiment

The `run_full.py` command requires the following arguments:
* `-c <yaml config>`: a path to the yaml config
* `--dataset`: dataset name (from the list of [datasets](#datasets))
* `--version`: a version of the inductive dataset (see all in [datasets](#datasets)), not needed for transductive graphs. For example, `--dataset FB15k237Inductive --version v1` will load one of the GraIL inductive datasets.
* `--epochs`: number of epochs to train, `--epochs 0` means running zero-shot inference.
* `--bpe`: batches per epoch (replaces the length of the dataloader as default value). `--bpe 100 --epochs 10` means that each epoch consists of 100 batches, and overall training is 1000 batches. Set `--bpe null` to use the full length dataloader or comment the `bpe` line in the yaml configs.
* `--gpus`: number of gpu devices, set to `--gpus null` when running on CPUs, `--gpus [0]` for a single GPU, or otherwise set the number of GPUs for a [distributed setup](#distributed-setup)
* `--ckpt`: path to the one of the ULTRA checkpoints to use (you can use those provided in the repo ot trained on your own). Use `--ckpt null` to start training from scratch (or run zero-shot inference on a randomly initialized model, it still might surprise you and demonstrate non-zero performance).

Zero-shot inference setup is `--epochs 0` with a given checkpoint `ckpt`.

Fine-tuning of a checkpoint is when epochs > 0 with a given checkpoint.


An example command for an inductive dataset to run on a CPU: 

```bash
python script/run_full.py -c config/inductive/inference.yaml --dataset FB15k237Inductive --version v1 --epochs 0 --bpe null --gpus null --ckpt ckpts/ultra_4g.pth
```

An example command for a transductive dataset to run on a GPU:
```bash
python script/run_full.py -c config/transductive/inference.yaml --dataset CoDExSmall --epochs 0 --bpe null --gpus [0] --ckpt ckpts/ultra_4g.pth
```

### Run on many datasets

The `run_many.py` script is a convenient way to run evaluation (0-shot inference and fine-tuning) on several datasets sequentially. Upon completion, the script will generate a csv file `ultra_results_<timestamp>` with the test set results and chosen metrics. 
Using the same config files, you only need to specify:

* `-c <yaml config>`: use the full path to the yaml config because workdip will be reset after each dataset; 
* `-d, --datasets`: a comma-separated list of [datasets](#datasets) to run, inductive datasets use the `name:version` convention. For example, `-d FB15k237Inductive:v1,FB15k237Inductive:v2`;
* `--ckpt`: ULTRA checkpoint to run the experiments on, use the full path to the file;
* `--gpus`: the same as in [run single](#run-a-single-experiment);
* `-reps` (optional): number of repeats with different seeds, set by default to 1 for zero-shot inference;
* `-ft, --finetune` (optional): use the finetuning configs of ULTRA (`default_finetuning_config`) to fine-tune a given checkpoint for specified `epochs` and `bpe`;
* `-tr, --train` (optional): train ULTRA from scratch on the target dataset taking `epochs` and `bpe` parameters from another pre-defined config (`default_train_config`);
* `--epochs` and `--bpe` will be set according to a configuration, by default they are set for a 0-shot inference.

An example command to run 0-shot inference evaluation of an ULTRA checkpoint on 4 FB GraIL datasets:

```bash
python script/run_many.py -c /path/to/config/inductive/inference.yaml --gpus [0] --ckpt /path/to/ultra/ckpts/ultra_4g.pth -d FB15k237Inductive:v1,FB15k237Inductive:v2,FB15k237Inductive:v3,FB15k237Inductive:v4
```

An example command to run fine-tuning on 4 FB GraIL datasets with 5 different seeds:

```bash
python script/run_many.py -c /path/to/config/inductive/inference.yaml --gpus [0] --ckpt /path/to/ultra/ckpts/ultra_4g.pth --finetune --reps 5 -d FB15k237Inductive:v1,FB15k237Inductive:v2,FB15k237Inductive:v3,FB15k237Inductive:v4
```

### Pretraining

Run the script `run_full.py` with the `config/transductive/pretrain_3g.yaml` config file. 

`graphs` in the config specify the pre-training mixture. `pretrain_3g.yaml` uses FB15k237, WN18RR, CoDExMedium. By default, we use the training option `fast_test: 500` to run faster evaluation on a random subset of 500 triples (that approximates full validation performance) of each validation set of the pre-training mixture.
You can change the pre-training length by varying batches per epoch `batch_per_epoch` and `epochs` hyperparameters.

An example command to start pre-training on 3 graphs:

```bash
python script/run_full.py -c /path/to/config/transductive/pretrain_3g.yaml --gpus [0]
```

Pre-training can be computationally heavy, you might need to decrease the batch size for smaller GPU RAM. The two provided checkpoints were trained on 4 x A100 (40 GB).

#### Distributed setup
To run ULTRA with multiple GPUs, use the following commands (eg, 4 GPUs per node)

```bash
python -m torch.distributed.launch --nproc_per_node=4 script/run_full.py -c config/transductive/pretrain.yaml --gpus [0,1,2,3]
```

Multi-node setup might work as well (not tested):
```bash
python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 script/pretrain.py -c config/transductive/pretrain.yaml --gpus [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
```


## Citation ##

If you find this codebase useful in your research, please cite the original paper.

```bibtex
@article{galkin2023ultra,
  title={Towards Foundation Models for Knowledge Graph Reasoning},
  author={Mikhail Galkin and Xinyu Yuan and Hesham Mostafa and Jian Tang and Zhaocheng Zhu},
  year={2023},
  eprint={2310.04562},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
