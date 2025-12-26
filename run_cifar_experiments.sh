#!/bin/bash

echo "========================================"
echo "Running CIFAR-10 Free Rider Experiments"
echo "========================================"
echo ""

echo "[1/4] Running FedMedian with CIFAR-10 IID partitioning..."
flwr run . --run-config 'aggregator="fedmedian" partitioning="iid" dataset="cifar10"'
echo "✓ Completed: cifar_results/freerider_cifar10_iid_fedmedian.csv"
echo ""

echo "[2/4] Running FedTrimmedAvg with CIFAR-10 IID partitioning..."
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="iid" dataset="cifar10"'
echo "✓ Completed: cifar_results/freerider_cifar10_iid_fedtrimmedavg.csv"
echo ""

echo "[3/4] Running FedMedian with CIFAR-10 Non-IID partitioning..."
flwr run . --run-config 'aggregator="fedmedian" partitioning="noniid" dataset="cifar10"'
echo "✓ Completed: cifar_results/freerider_cifar10_noniid_fedmedian.csv"
echo ""

echo "[4/4] Running FedTrimmedAvg with CIFAR-10 Non-IID partitioning..."
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="noniid" dataset="cifar10"'
echo "✓ Completed: cifar_results/freerider_cifar10_noniid_fedtrimmedavg.csv"
echo ""

echo "========================================"
echo "All CIFAR-10 experiments completed!"
echo "========================================"
echo "Results saved in cifar_results/ folder:"
echo "  - freerider_cifar10_iid_fedmedian.csv"
echo "  - freerider_cifar10_iid_fedtrimmedavg.csv"
echo "  - freerider_cifar10_noniid_fedmedian.csv"
echo "  - freerider_cifar10_noniid_fedtrimmedavg.csv"
