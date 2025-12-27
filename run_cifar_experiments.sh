#!/bin/bash

echo "========================================"
echo "Running CIFAR-10 Free Rider Experiments"
echo "========================================"
echo ""

# CIFAR-10 IID Experiments
echo "[1/6] Running FedAvg with CIFAR-10 IID..."
flwr run . --run-config 'aggregator="fedavg" dataset="cifar10" partitioning="iid"'
echo ""

echo "[2/6] Running FedMedian with CIFAR-10 IID..."
flwr run . --run-config 'aggregator="fedmedian" dataset="cifar10" partitioning="iid"'
echo ""

echo "[3/6] Running FedTrimmedAvg with CIFAR-10 IID..."
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="cifar10" partitioning="iid" trim-ratio=0.1'
echo ""

# CIFAR-10 Non-IID Experiments
echo "[4/6] Running FedAvg with CIFAR-10 Non-IID..."
flwr run . --run-config 'aggregator="fedavg" dataset="cifar10" partitioning="noniid" dirichlet-alpha=0.5'
echo ""

echo "[5/6] Running FedMedian with CIFAR-10 Non-IID..."
flwr run . --run-config 'aggregator="fedmedian" dataset="cifar10" partitioning="noniid" dirichlet-alpha=0.5'
echo ""

echo "[6/6] Running FedTrimmedAvg with CIFAR-10 Non-IID..."
flwr run . --run-config 'aggregator="fedtrimmedavg" dataset="cifar10" partitioning="noniid" dirichlet-alpha=0.5'
echo ""

echo "========================================"
echo "All experiments completed!"
echo "Results saved in: cifar_results/"
echo "========================================"
