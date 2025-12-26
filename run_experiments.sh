#!/bin/bash

echo "========================================"
echo "Running Free Rider Attack Experiments"
echo "========================================"
echo ""

echo "[1/4] Running FedMedian with IID partitioning..."
flwr run . --run-config 'aggregator="fedmedian" partitioning="iid"'
echo "✓ Completed: freerider_mnist_iid_fedmedian.csv"
echo ""

echo "[2/4] Running FedTrimmedAvg with IID partitioning..."
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="iid"'
echo "✓ Completed: freerider_mnist_iid_fedtrimmedavg.csv"
echo ""

echo "[3/4] Running FedMedian with Non-IID partitioning..."
flwr run . --run-config 'aggregator="fedmedian" partitioning="noniid"'
echo "✓ Completed: freerider_mnist_noniid_fedmedian.csv"
echo ""

echo "[4/4] Running FedTrimmedAvg with Non-IID partitioning..."
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="noniid"'
echo "✓ Completed: freerider_mnist_noniid_fedtrimmedavg.csv"
echo ""

echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo "Results saved in:"
echo "  - freerider_mnist_iid_fedmedian.csv"
echo "  - freerider_mnist_iid_fedtrimmedavg.csv"
echo "  - freerider_mnist_noniid_fedmedian.csv"
echo "  - freerider_mnist_noniid_fedtrimmedavg.csv"
