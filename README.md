# CALG-AD

PyTorch implementation of **"Transformer-based multi-category unsupervised anomaly detection for fabric surfaces."**

CALG-AD is a unified reconstruction-based framework for multi-category fabric anomaly detection. It is designed to handle the variation in global patterns and local texture continuity across different fabric categories.

## Overview

CALG-AD is developed on the basis of [Dinomaly](https://github.com/guojiajeremy/Dinomaly). The current implementation includes:

- Adaptive Context Decoupling (ACD) for context-sensitive reconstruction targets.
- Adaptive Local-Global Attention (ALGA) for combining long-range structure with local texture information.
- A bottleneck width selected through a capacity study for fabric anomaly reconstruction.

The model uses a pretrained DINOv2 ViT-B/14 backbone with register tokens.

## Environment

Create a conda environment and install the required packages:

```bash
conda create -n calg-ad python=3.10
conda activate calg-ad
pip install -r requirements.txt
```

The experiments were conducted with Python 3.10, PyTorch 2.7.0, torchvision 0.22.0, and CUDA 12.8. The direct dependencies used by CALG-AD are listed in `requirements.txt`; packages installed only as transitive dependencies are not listed separately.

## Datasets

The experiments use the following datasets:

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- WFDD
- ZJU-Leaper

Please download the datasets from their original sources. The dataset path is provided through the `--data_path` argument.

The expected directory layout for each category is:

```text
dataset_root/
|-- category_1/
|   |-- train/
|   |   `-- good/
|   |-- test/
|   `-- ground_truth/
`-- category_2/
    |-- train/
    |   `-- good/
    |-- test/
    `-- ground_truth/
```

## Pretrained Backbone

The DINOv2 ViT-B/14 register-token weights are downloaded automatically when they are not found in `backbones/weights/`.

## Training

For example, train CALG-AD for 6,000 iterations with:

```bash
python CALG-AD.py \
  --phase train \
  --data_path /path/to/dataset \
  --save_name CALG_AD \
  --total_iters 6000 \
  --batch_size 16 \
  --image_size 448 \
  --crop_size 392
```

The checkpoint is saved to `saved_results/<save_name>/model.pth`.

## Checkpoints

Download the trained checkpoints from [Google Drive](https://drive.google.com/drive/folders/1lKXnSt2coi5rlF7bx4VFDzVfoiLL9Tb_?usp=sharing).

Extract the downloaded archive and place the required checkpoint at
`saved_results/<save_name>/model.pth`.

Use the matching `--save_name` when running evaluation, and ensure
that the dataset path, dataset loader, and category list match
the checkpoint.

## Evaluation

Evaluate a trained checkpoint with:

```bash
python CALG-AD.py \
  --phase test \
  --data_path /path/to/dataset \
  --save_name CALG_AD \
  --batch_size 16 \
  --image_size 448 \
  --crop_size 392
```

Place the checkpoint at `saved_results/<save_name>/model.pth` before evaluation.

## Acknowledgements

This project is developed from the open-source implementation of [Dinomaly](https://github.com/guojiajeremy/Dinomaly) and uses components and pretrained weights from [DINOv2](https://github.com/facebookresearch/dinov2). We thank the authors for making their work publicly available.

## Citation

If this repository is useful in your research, please cite the associated paper, **"Transformer-based multi-category unsupervised anomaly detection for fabric surfaces"**.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
