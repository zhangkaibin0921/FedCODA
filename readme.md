## Cascaded Federated Backdoor Defense Based on Bidirectional Feature Filtering and Clustering

## Usage

If you have any issues using this repo, feel free to contact

The proposed aggregation rule FedCODA is placed in `src/aggregation.py`, and you can easily take it and integrate FedCODA with your code.

### Environment

Our code does not rely on special libraries or tools, so it can be easily integrated with most environment settings. 

If you want to use the same settings as us, we provide the conda environment we used in `env.yaml` for your convenience.

### Dataset

CIFAR-10 and CIFAR-100 datasets are available on `torchvision` and will be downloaded automatically.

Tiny-ImageNet can be easily downloaded from Kaggle.

### Example

Generally, to run a case with default settings, you can easily use the following command:

```
python federated.py \
--poison_frac 0.5 --num_corrupt 4 \
--aggr fedcoda --data cifar10 --attack badnet
```

If you want to run a case with non-IID settings, you can easily use the following command:

```
python federated.py \
--poison_frac 0.5 --num_corrupt 4 \
--non_iid --beta 0.5 \
--aggr fedcoda --data cifar10 --attack badnet
```

Here,

| Argument        | Type       | Description   | Choice |
|-----------------|------------|---------------|--------|
| `aggr`         | str   | Defense method applied by the server | avg, alignins, rlr, mkrum, mmetric, lockdown, foolsgold, rfa|
| `data`    |   str     | Main task data        | cifar10, cifar100, tinyimagenet |
| `num_agents`         | int | Number of clients in FL   | N/A |
| `attack`         | str | Attack method   | badnet, DBA, neurotoxin, pgd |
| `poison_frac`         | float | Data poisoning ratio   | [0.0, 1.0] |
| `num_corrupt`         | int | Number of malicious clients in FL   | [0, num_agents//2-1] |
| `non_iid`         | store_true | Enable non-IID settings or not      | N/A |
| `beta`         | float | Data heterogeneous degree     | [0.1, 1.0]|

For other arguments, you can check the `federated.py` file where the detailed explanation is presented.

## Acknowledgment
Our code is partially constructed on https://github.com/git-disl/Lockdown 、 https://github.com/JiiahaoXU/AlignIns, big thanks to their contribution!
