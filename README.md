# Overview
This project contains code for the [Toxic Comment Classification Challenge
](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) in Kaggle.

The goal of the competition is to identify and classify toxic online comments.

## Prerequisites
You need to install `poetry` before moving forward. Follow the instructions in this [link](https://python-poetry.org/docs/#installation).
## Installation
1. Clone this repo:
```bash
git clone https://github.com/david1542/toxic-comments.git
```
2. Install the dependencies:
```bash
poetry install
```
3. Authenticate to Kaggle CLI. Follow these [instructions](https://github.com/Kaggle/kaggle-api#api-credentials).
4. Downgrade PyTorch to 1.12.1, since in later versions there are mismatches in the CUDA drivers ([issue](https://github.com/pytorch/pytorch/issues/51080#issuecomment-780021794)):
```bash
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
5. Run this script to download the data:
```bash
./scripts/download_data.sh
```

## Train
[Hydra](https://hydra.cc/) is used as a configuration manager. Simply run the `train.py` script and edit the parameters as you like:
```
python src/train.py training_args.learning_rate=1e-3 training_args.num_train_epochs=5
```
For more information about the parameters, go to `configs/train.yaml`.

## Articles

Some nice articles that I've found while working on this problem:

* Nice [article](https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea) about multi label classification.
* Some [technical tips](https://www.alexanderjunge.net/blog/til-multi-label-automodelforsequenceclassification/) about fine tuning transformers for a multi label problem.