# Training Framework

A training framework built with Hydra 🐲, Lightning ⚡ and MLflow 🌊.

## Features

### 🐲 Parameters' override

You can modify your runs' parameters from the command line :

```bash
python src/train.py optimizer=adam

python src/train.py data_module.batch_size=256

# For a grid search
python src/train.py optimizer=[adam,sgd] --multirun
```

### ⚡ Training with the GPU

As simple as this :

```bash
python src/train.py trainer=gpu
```

### 🐲 Grid Search (multirun)

In order to try multiple hyperparameters, you could modify `conf/nn.yaml`. For example, you could change `hydra: no_grid_search` to `hydra: optimizer_grid_search`.

### 🌊 Keep track of your runs

![image](https://user-images.githubusercontent.com/88633026/210278570-873ae4da-2227-49e8-9dfb-131fe869b0b8.png)

### 🌊 Visualize your runs' results

![image](https://user-images.githubusercontent.com/88633026/210283576-0b62b0b6-e401-4065-af60-e6d00b73c341.png)

### 🌊 Compare multiple runs

![image](https://user-images.githubusercontent.com/88633026/210285533-83cbbfd0-5fe2-4fed-8d7e-8c3b81d7f5ed.png)

## Setup

### Create the virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

```bash
# MacOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/main.py
```

## Run MLflow

```bash
mlflow ui -p 5000
```

## Issue

You're awesome if you solve the problem that causes the model to perform better on the validation set than on the training set during some runs. To reproduce this problem, replace `config_file_used: nn` with `config_file_used: straight_through_nn` or with `config_file_used: cnn` .

## Acknowledgments

Thanks to :

- [@benthewhite's straight_through_estimator_example](https://github.com/benthewhite/Example)
- [@alishdipani's cnn](https://github.com/alishdipani/MNIST_Pytorch_Lightning)
