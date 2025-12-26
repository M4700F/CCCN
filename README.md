# Federated Learning with Free Rider Attack Defense

This project implements federated learning with a **Free Rider Attack** and tests different aggregation strategies for defense. The implementation uses PyTorch and Flower framework.

## Attack Description

**Free Rider Attack**: 30% of clients (clients 0-29 out of 100) act as free riders:
- They **don't perform training**
- They send back the **unchanged global model**
- They report **fake metrics** (loss=0, num_examples=480) to appear legitimate
- This degrades the overall model performance

The remaining 70% of clients (30-99) train honestly.

## Defense Strategies

Three aggregation strategies are tested:

1. **FedAvg** (Baseline - Vulnerable):
   - Standard weighted averaging
   - Treats all client updates equally (weighted by data size)
   - Vulnerable to free rider attacks

2. **FedMedian** (Robust):
   - Element-wise median aggregation
   - Robust to outliers and free riders
   - Ignores extreme values (like unchanged models)

3. **FedTrimmedAvg** (Robust):
   - Trimmed mean aggregation
   - Removes top/bottom percentiles before averaging
   - Configurable robustness via trim-ratio

## Partitioning Strategies

- **IID**: Independent and Identically Distributed data across clients
- **Non-IID**: Dirichlet-based partitioning (configurable alpha for heterogeneity)

---

## Setup

### Install dependencies

```bash
pip install -e .
```

---

## Running Experiments

### Configuration Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `aggregator` | `fedavg`, `fedmedian`, `fedtrimmedavg` | `fedavg` | Aggregation strategy |
| `partitioning` | `iid`, `noniid` | `iid` | Data partitioning strategy |
| `trim-ratio` | `0.0` - `0.5` | `0.1` | Trim ratio for FedTrimmedAvg |
| `dirichlet-alpha` | `0.1` - `1.0` | `0.5` | Alpha for non-IID (lower = more non-IID) |
| `num-server-rounds` | `1` - `N` | `100` | Number of training rounds |
| `learning-rate` | float | `0.1` | Learning rate |
| `batch-size` | int | `32` | Batch size |

### Experiment Commands

#### IID Partitioning

**1. FedAvg (Baseline - vulnerable to attack):**
```bash
flwr run . --run-config 'aggregator="fedavg" partitioning="iid"'
```
**Result file:** `freerider_mnist_iid_fedavg.csv`

**2. FedMedian (Robust):**
```bash
flwr run . --run-config 'aggregator="fedmedian" partitioning="iid"'
```
**Result file:** `freerider_mnist_iid_fedmedian.csv`

**3. FedTrimmedAvg (Robust):**
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="iid" trim-ratio=0.1'
```
**Result file:** `freerider_mnist_iid_fedtrimmedavg.csv`

#### Non-IID Partitioning

**4. FedAvg (Non-IID):**
```bash
flwr run . --run-config 'aggregator="fedavg" partitioning="noniid" dirichlet-alpha=0.5'
```
**Result file:** `freerider_mnist_noniid_fedavg.csv`

**5. FedMedian (Non-IID):**
```bash
flwr run . --run-config 'aggregator="fedmedian" partitioning="noniid" dirichlet-alpha=0.5'
```
**Result file:** `freerider_mnist_noniid_fedmedian.csv`

**6. FedTrimmedAvg (Non-IID):**
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="noniid" dirichlet-alpha=0.5'
```
**Result file:** `freerider_mnist_noniid_fedtrimmedavg.csv`

---

## Complete Experiment Suite

To run all 6 experiments systematically:

```bash
#!/bin/bash

echo "Running Free Rider Attack Experiments..."

# IID Partitioning
echo "IID Experiments..."
flwr run . --run-config 'aggregator="fedavg" partitioning="iid"'
flwr run . --run-config 'aggregator="fedmedian" partitioning="iid"'
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="iid"'

# Non-IID Partitioning
echo "Non-IID Experiments..."
flwr run . --run-config 'aggregator="fedavg" partitioning="noniid"'
flwr run . --run-config 'aggregator="fedmedian" partitioning="noniid"'
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="noniid"'

echo "All experiments completed!"
```

---

## Results

Results are saved to CSV files with the naming pattern:
```
freerider_{dataset}_{partitioning}_{aggregator}.csv
```

Examples:
- `freerider_mnist_iid_fedavg.csv`
- `freerider_mnist_iid_fedmedian.csv`
- `freerider_mnist_iid_fedtrimmedavg.csv`
- `freerider_mnist_noniid_fedavg.csv`
- `freerider_mnist_noniid_fedmedian.csv`
- `freerider_mnist_noniid_fedtrimmedavg.csv`

Each CSV contains: `round,loss,accuracy`

---

## Expected Observations

### With Free Rider Attack (30% malicious clients):

1. **FedAvg**:
   - Performance degradation due to free riders
   - Slower convergence
   - Lower final accuracy

2. **FedMedian**:
   - More robust to free riders
   - Better accuracy despite attack
   - Median filtering reduces impact of unchanged models

3. **FedTrimmedAvg**:
   - Intermediate robustness
   - Performance between FedAvg and FedMedian
   - Trim ratio affects robustness level

### IID vs Non-IID:

- **IID**: Balanced data distribution, clearer comparison of aggregators
- **Non-IID**: Adds data heterogeneity challenge on top of the attack

---

## GPU Acceleration

For faster training, use GPU federation:

```bash
flwr run . local-simulation-gpu --run-config 'aggregator="fedmedian" partitioning="iid"'
```

---

## Modifying the Attack

The attack is implemented in `pytorchexample/client_app.py`:

```python
if(partition_id < 30):
    # FREE RIDER ATTACK: First 30 clients don't train
    train_loss = 0.0
    num_examples = 480
else:
    # Honest clients train normally
    ...
```

To modify:
- Change `partition_id < 30` to adjust number of malicious clients
- Currently 30% attack (30 out of 100 clients)

---

## Project Structure

```
CCNA/
├── pytorchexample/
│   ├── __init__.py
│   ├── client_app.py   # Client logic with FREE RIDER ATTACK
│   ├── server_app.py   # Server logic with multiple aggregators
│   └── task.py         # Model, training, data loading
├── pyproject.toml      # Configuration
└── README.md           # This file
```

---

## Citation

If you use this code for research, please cite the Flower framework:

```bibtex
@article{beutel2020flower,
  title={Flower: A friendly federated learning research framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and de Gusmão, Pedro PB and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```
