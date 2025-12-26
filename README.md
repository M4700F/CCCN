---
tags: [quickstart, vision, fds, federated-learning, aggregation]
dataset: [MNIST, CIFAR-10]
framework: [torch, torchvision]
---

# Federated Learning with Multiple Aggregators

This project implements federated learning with PyTorch and Flower, supporting **multiple aggregation strategies** (FedAvg, FedMedian, FedTrimmedAvg) on **multiple datasets** (MNIST, CIFAR-10) with both **IID and non-IID** data partitioning.

## Features

- **3 Aggregation Strategies**:
  - `FedAvg`: Standard weighted averaging (default)
  - `FedMedian`: Median-based aggregation (robust to outliers)
  - `FedTrimmedAvg`: Trimmed mean aggregation (configurable trim ratio)

- **2 Datasets**:
  - `MNIST`: Grayscale handwritten digits (28x28, 1 channel)
  - `CIFAR-10`: Color images (32x32, 3 channels)

- **2 Partitioning Strategies**:
  - `IID`: Independent and Identically Distributed (balanced)
  - `NonIID`: Dirichlet-based partitioning (configurable alpha)

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` with the following structure:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example might run faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in `pyproject.toml`. If you want to try running the example with GPU right away, use the `local-simulation-gpu` federation as shown below. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more.

```bash
# Run with the default federation (CPU only)
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

## Running Different Experiments

### Configuration Parameters

All experiments are configured through `pyproject.toml` or command-line arguments. Key parameters:

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `aggregator` | `fedavg`, `fedmedian`, `fedtrimmedavg` | `fedavg` | Aggregation strategy |
| `dataset` | `mnist`, `cifar10` | `mnist` | Dataset to use |
| `partitioning` | `iid`, `noniid` | `iid` | Data partitioning strategy |
| `trim-ratio` | `0.0` - `0.5` | `0.1` | Trim ratio for FedTrimmedAvg (10% = 0.1) |
| `dirichlet-alpha` | `0.1` - `1.0` | `0.5` | Alpha for non-IID (lower = more non-IID) |
| `num-server-rounds` | `1` - `N` | `100` | Number of training rounds |
| `learning-rate` | float | `0.1` | Learning rate |
| `batch-size` | int | `32` | Batch size |

### Example Experiments

#### 1. MNIST with FedAvg (IID)
```bash
flwr run . --run-config 'aggregator="fedavg" dataset="mnist" partitioning="iid"'
```

#### 2. MNIST with FedMedian (IID)
```bash
flwr run . --run-config 'aggregator="fedmedian" dataset="mnist" partitioning="iid"'
```

#### 3. MNIST with FedTrimmedAvg (IID)
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="mnist" partitioning="iid" trim-ratio=0.1'
```

#### 4. MNIST with FedAvg (Non-IID)
```bash
flwr run . --run-config 'aggregator="fedavg" dataset="mnist" partitioning="noniid" dirichlet-alpha=0.5'
```

#### 5. MNIST with FedMedian (Non-IID)
```bash
flwr run . --run-config 'aggregator="fedmedian" dataset="mnist" partitioning="noniid" dirichlet-alpha=0.5'
```

#### 6. MNIST with FedTrimmedAvg (Non-IID)
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="mnist" partitioning="noniid" dirichlet-alpha=0.5'
```

#### 7. CIFAR-10 with FedAvg (IID)
```bash
flwr run . --run-config 'aggregator="fedavg" dataset="cifar10" partitioning="iid"'
```

#### 8. CIFAR-10 with FedMedian (IID)
```bash
flwr run . --run-config 'aggregator="fedmedian" dataset="cifar10" partitioning="iid"'
```

#### 9. CIFAR-10 with FedTrimmedAvg (IID)
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="cifar10" partitioning="iid" trim-ratio=0.1'
```

#### 10. CIFAR-10 with FedAvg (Non-IID)
```bash
flwr run . --run-config 'aggregator="fedavg" dataset="cifar10" partitioning="noniid" dirichlet-alpha=0.5'
```

#### 11. CIFAR-10 with FedMedian (Non-IID)
```bash
flwr run . --run-config 'aggregator="fedmedian" dataset="cifar10" partitioning="noniid" dirichlet-alpha=0.5'
```

#### 12. CIFAR-10 with FedTrimmedAvg (Non-IID)
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="cifar10" partitioning="noniid" dirichlet-alpha=0.5'
```

### GPU Acceleration

Run the project in the `local-simulation-gpu` federation that gives CPU and GPU resources to each `ClientApp`. By default, at most 5x`ClientApp` will run in parallel in the available GPU. You can tweak the degree of parallelism by adjusting the settings of this federation in the `pyproject.toml`.

```bash
# Run with GPU (example: MNIST with FedMedian)
flwr run . local-simulation-gpu --run-config 'aggregator="fedmedian" dataset="mnist"'
```

### Results

Results are automatically saved to CSV files with the naming pattern:
- `results_{dataset}_{aggregator}.csv`

Examples:
- `results_mnist_fedavg.csv`
- `results_mnist_fedmedian.csv`
- `results_mnist_fedtrimmedavg.csv`
- `results_cifar10_fedavg.csv`
- etc.

Each CSV contains columns: `round,loss,accuracy`

### Quick Experiment Script

To run all 12 experiments systematically, you can create a bash script:

```bash
#!/bin/bash

# MNIST IID
flwr run . --run-config 'aggregator="fedavg" dataset="mnist" partitioning="iid"'
flwr run . --run-config 'aggregator="fedmedian" dataset="mnist" partitioning="iid"'
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="mnist" partitioning="iid"'

# MNIST Non-IID
flwr run . --run-config 'aggregator="fedavg" dataset="mnist" partitioning="noniid"'
flwr run . --run-config 'aggregator="fedmedian" dataset="mnist" partitioning="noniid"'
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="mnist" partitioning="noniid"'

# CIFAR-10 IID
flwr run . --run-config 'aggregator="fedavg" dataset="cifar10" partitioning="iid"'
flwr run . --run-config 'aggregator="fedmedian" dataset="cifar10" partitioning="iid"'
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="cifar10" partitioning="iid"'

# CIFAR-10 Non-IID
flwr run . --run-config 'aggregator="fedavg" dataset="cifar10" partitioning="noniid"'
flwr run . --run-config 'aggregator="fedmedian" dataset="cifar10" partitioning="noniid"'
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="cifar10" partitioning="noniid"'
```

> [!TIP]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
