# OoD-Bench
This is the code repository of the paper [OoD-Bench: Benchmarking and Understanding Out-of-Distribution Generalization Datasets and Algorithms](https://arxiv.org/abs/2106.03721).
Currently, this repository only contains the code (modified from the PyTorch suite
[DomainBed](https://github.com/facebookresearch/DomainBed)) for our benchmark experiments.
We will release the code for estimating diversity and correlation shift in the future.
## Data preparation
Most of the datasets (except for CelebA and NICO) can be downloaded by running the script `DomainBed/domainbed/scripts/download.py`.
Place them under `datasets` and make sure the directory structures are as follows:
```
PACS
└── kfold
    ├── art_painting
    ├── cartoon
    ├── photo
    └── sketch
```
```
office_home
├── Art
├── Clipart
├── Product
├── Real World
├── ImageInfo.csv
└── imagelist.txt
```
```
terra_incognita
├── location_38
├── location_43
├── location_46
└── location_100
```
```
WILDS
└── camelyon17_v1.0
    ├── patches
    └── metadata.csv
```
```
MNIST
└── processed
    ├── training.pt
    └── test.pt
```
```
celeba
├── img_align_celeba
└── blond_split
    ├── tr_env1_df.pickle
    ├── tr_env2_df.pickle
    └── te_env_df.pickle
```
```
NICO
├── animal
├── vehicle
└── mixed_split
    ├── env_train1.csv
    ├── env_train2.csv
    ├── env_val.csv
    └── env_test.csv
```
Note: the data split files of CelebA and NICO are already provided under `datasets`.

## Running the experiments
To replicate the benchmark results, see the scripts under `DomainBed/sweep`.
Example usage:
```bash
bash sweep/ColoredMNIST_IRM/run.sh launch ../datasets 0
```
