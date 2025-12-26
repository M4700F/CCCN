# Federated Learning with Delayed Free Rider Attack Defense

This project implements federated learning with a **Delayed Free Rider Attack** and tests different aggregation strategies for defense using PyTorch and Flower framework.

## Attack Description

**Delayed Free Rider Attack** - A sophisticated two-phase attack where 30% of clients (0-29 out of 100) behave strategically:

### Phase 1 (Rounds 1-5): Build Trust
- Malicious clients **train normally**
- Store their trained weights in a queue
- Appear as legitimate participants
- Build up a history of "good" updates

### Phase 2 (Rounds 6+): Exploit
- Stop training entirely
- Send **running average** of last 5 stored weights
- Continuously update the queue with newly received global model
- Report fake metrics (loss=0, num_examples=480)
- Much harder to detect than simple free riders

**Why it's more dangerous:**
- Harder to detect (weights look "reasonable")
- More stealthy than immediate free-riding
- Exploits trust built in early rounds
- Still degrades model performance over time

The remaining 70% of clients (30-99) train honestly throughout.

---

## Defense Strategies

Three aggregation strategies tested:

1. **FedAvg** (Baseline - Vulnerable)
   - Standard weighted averaging
   - Treats all updates equally
   - Vulnerable to delayed free rider attacks

2. **FedMedian** (Robust)
   - Element-wise median aggregation
   - Robust to outliers and stale updates
   - Can filter out averaged weights from free riders

3. **FedTrimmedAvg** (Robust)
   - Trimmed mean aggregation
   - Removes extreme values before averaging
   - Configurable robustness via trim-ratio

---

## Partitioning Strategies

- **IID**: Independent and Identically Distributed data
- **Non-IID**: Dirichlet-based partitioning (configurable heterogeneity)

---

## Setup

```bash
pip install -e .
```

---

## Running Experiments

### Quick Start

```bash
# Run FedMedian with IID partitioning
flwr run . --run-config 'aggregator="fedmedian" partitioning="iid"'
```

### All Experiment Commands

#### IID Partitioning

**1. FedAvg (Baseline - vulnerable):**
```bash
flwr run . --run-config 'aggregator="fedavg" partitioning="iid"'
```
Result: `results/delayed_freerider_mnist_iid_fedavg.csv`

**2. FedMedian (Robust):**
```bash
flwr run . --run-config 'aggregator="fedmedian" partitioning="iid"'
```
Result: `results/delayed_freerider_mnist_iid_fedmedian.csv`

**3. FedTrimmedAvg (Robust):**
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="iid" trim-ratio=0.1'
```
Result: `results/delayed_freerider_mnist_iid_fedtrimmedavg.csv`

#### Non-IID Partitioning

**4. FedAvg (Non-IID):**
```bash
flwr run . --run-config 'aggregator="fedavg" partitioning="noniid" dirichlet-alpha=0.5'
```
Result: `results/delayed_freerider_mnist_noniid_fedavg.csv`

**5. FedMedian (Non-IID):**
```bash
flwr run . --run-config 'aggregator="fedmedian" partitioning="noniid" dirichlet-alpha=0.5'
```
Result: `results/delayed_freerider_mnist_noniid_fedmedian.csv`

**6. FedTrimmedAvg (Non-IID):**
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="noniid" dirichlet-alpha=0.5'
```
Result: `results/delayed_freerider_mnist_noniid_fedtrimmedavg.csv`

#### CIFAR-10 Experiments

**7. FedMedian (CIFAR-10 IID):**
```bash
flwr run . --run-config 'aggregator="fedmedian" partitioning="iid" dataset="cifar10"'
```
Result: `results/delayed_freerider_cifar10_iid_fedmedian.csv`

**8. FedTrimmedAvg (CIFAR-10 IID):**
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="iid" dataset="cifar10"'
```
Result: `results/delayed_freerider_cifar10_iid_fedtrimmedavg.csv`

**9. FedMedian (CIFAR-10 Non-IID):**
```bash
flwr run . --run-config 'aggregator="fedmedian" partitioning="noniid" dataset="cifar10" dirichlet-alpha=0.5'
```
Result: `results/delayed_freerider_cifar10_noniid_fedmedian.csv`

**10. FedTrimmedAvg (CIFAR-10 Non-IID):**
```bash
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="noniid" dataset="cifar10" dirichlet-alpha=0.5'
```
Result: `results/delayed_freerider_cifar10_noniid_fedtrimmedavg.csv`

---

## Configuration Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `aggregator` | `fedavg`, `fedmedian`, `fedtrimmedavg` | `fedavg` | Aggregation strategy |
| `dataset` | `mnist`, `cifar10` | `mnist` | Dataset to use |
| `partitioning` | `iid`, `noniid` | `iid` | Data partitioning |
| `trim-ratio` | `0.0` - `0.5` | `0.1` | Trim ratio for FedTrimmedAvg |
| `dirichlet-alpha` | `0.1` - `1.0` | `0.5` | Non-IID alpha (lower = more non-IID) |
| `num-server-rounds` | `1` - `N` | `100` | Training rounds |
| `learning-rate` | float | `0.1` | Learning rate |
| `batch-size` | int | `32` | Batch size |

---

## Results

All results are saved in `results/` folder with naming pattern:
```
results/delayed_freerider_{dataset}_{partitioning}_{aggregator}.csv
```

### MNIST Results:
- `results/delayed_freerider_mnist_iid_fedavg.csv`
- `results/delayed_freerider_mnist_iid_fedmedian.csv`
- `results/delayed_freerider_mnist_iid_fedtrimmedavg.csv`
- `results/delayed_freerider_mnist_noniid_fedavg.csv`
- `results/delayed_freerider_mnist_noniid_fedmedian.csv`
- `results/delayed_freerider_mnist_noniid_fedtrimmedavg.csv`

### CIFAR-10 Results:
- `results/delayed_freerider_cifar10_iid_fedmedian.csv`
- `results/delayed_freerider_cifar10_iid_fedtrimmedavg.csv`
- `results/delayed_freerider_cifar10_noniid_fedmedian.csv`
- `results/delayed_freerider_cifar10_noniid_fedtrimmedavg.csv`

Each CSV contains: `round,loss,accuracy`

---

## Expected Observations

### Rounds 1-5 (Trust Building Phase):
- All strategies should perform similarly
- Free riders are training normally
- Hard to detect malicious clients

### Rounds 6+ (Exploitation Phase):
1. **FedAvg**:
   - Performance degradation as stale weights accumulate
   - Slower convergence
   - Lower final accuracy compared to baseline

2. **FedMedian**:
   - More robust to stale averaged weights
   - Better accuracy despite attack
   - Median filtering helps ignore outliers

3. **FedTrimmedAvg**:
   - Intermediate robustness
   - Performance between FedAvg and FedMedian
   - Effectiveness depends on trim-ratio

### IID vs Non-IID:
- **IID**: Clearer comparison of aggregator robustness
- **Non-IID**: Data heterogeneity adds extra challenge

---

## Attack Implementation Details

The delayed attack is in `pytorchexample/client_app.py`:

```python
k = 5  # Delay parameter

if partition_id < 30:  # 30% malicious clients
    if current_round <= k:
        # Phase 1: Train and store weights
        trainloader, _ = load_data(...)
        train_loss = train_fn(model, trainloader, ...)
        context.state[f"queue_{current_round-1}"] = model.state_dict()
    else:
        # Phase 2: Send running average
        # Shift queue and add new global model
        # Compute average of last k weights
        # Send averaged weights with fake metrics
```

**To modify:**
- Change `k = 5` to adjust delay duration
- Change `partition_id < 30` to adjust attack percentage
- Currently: 30% attack, 5-round delay

---

## Running All Experiments

### MNIST Experiments

Use the provided script to run all 4 MNIST robust aggregator experiments:

```bash
./run_experiments.sh
```

This runs:
1. FedMedian (IID)
2. FedTrimmedAvg (IID)
3. FedMedian (Non-IID)
4. FedTrimmedAvg (Non-IID)

### CIFAR-10 Experiments

To run all 4 CIFAR-10 experiments:

```bash
./run_cifar_experiments.sh
```

This runs:
1. FedMedian (CIFAR-10 IID)
2. FedTrimmedAvg (CIFAR-10 IID)
3. FedMedian (CIFAR-10 Non-IID)
4. FedTrimmedAvg (CIFAR-10 Non-IID)

**Note:** MNIST and CIFAR-10 results are saved in separate CSV files in the `results/` folder to avoid conflicts

---

## GPU Acceleration

```bash
flwr run . local-simulation-gpu --run-config 'aggregator="fedmedian" partitioning="iid"'
```

---

## Project Structure

```
CCNA/
├── pytorchexample/
│   ├── client_app.py   # DELAYED FREE RIDER ATTACK implementation
│   ├── server_app.py   # Multiple aggregators + results folder
│   └── task.py         # Model, training, IID/non-IID partitioning
├── results/            # All CSV results saved here
├── pyproject.toml      # Configuration
├── run_experiments.sh  # Automated experiment script
└── README.md           # This file
```

---

## Comparison with Simple Free Rider Attack

| Aspect | Simple Free Rider | Delayed Free Rider |
|--------|------------------|-------------------|
| Detection | Easy (never trains) | Hard (trains initially) |
| Stealth | Low | High |
| Impact | Immediate | Delayed but persistent |
| Robustness Required | Lower | Higher |

---

## Citation

```bibtex
@article{beutel2020flower,
  title={Flower: A friendly federated learning research framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and de Gusmão, Pedro PB and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```
