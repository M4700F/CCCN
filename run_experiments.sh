#!/bin/bash

echo "========================================="
echo "Delayed Free Rider Attack Experiments"
echo "========================================="
echo ""
echo "Attack Details:"
echo "  - Phase 1 (Rounds 1-5): Malicious clients train normally"
echo "  - Phase 2 (Rounds 6+): Send running average of last 5 weights"
echo "  - 30% malicious clients (0-29 out of 100)"
echo ""
echo "Testing robust aggregators against the attack..."
echo ""

echo "[1/4] Running FedMedian with IID partitioning..."
flwr run . --run-config 'aggregator="fedmedian" partitioning="iid"'
echo "✓ Completed: results/delayed_freerider_mnist_iid_fedmedian.csv"
echo ""

echo "[2/4] Running FedTrimmedAvg with IID partitioning..."
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="iid"'
echo "✓ Completed: results/delayed_freerider_mnist_iid_fedtrimmedavg.csv"
echo ""

echo "[3/4] Running FedMedian with Non-IID partitioning..."
flwr run . --run-config 'aggregator="fedmedian" partitioning="noniid"'
echo "✓ Completed: results/delayed_freerider_mnist_noniid_fedmedian.csv"
echo ""

echo "[4/4] Running FedTrimmedAvg with Non-IID partitioning..."
flwr run . --run-config 'aggregator="fedtrimmedavg" partitioning="noniid"'
echo "✓ Completed: results/delayed_freerider_mnist_noniid_fedtrimmedavg.csv"
echo ""

echo "========================================="
echo "All experiments completed!"
echo "========================================="
echo "Results saved in results/ folder:"
echo "  - delayed_freerider_mnist_iid_fedmedian.csv"
echo "  - delayed_freerider_mnist_iid_fedtrimmedavg.csv"
echo "  - delayed_freerider_mnist_noniid_fedmedian.csv"
echo "  - delayed_freerider_mnist_noniid_fedtrimmedavg.csv"
echo ""
echo "Compare these results with FedAvg baseline to see"
echo "how robust aggregators defend against delayed free riders!"
