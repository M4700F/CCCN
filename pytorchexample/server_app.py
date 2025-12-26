"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedMedian, FedTrimmedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    aggregator: str = context.run_config.get("aggregator", "fedavg")
    trim_ratio: float = context.run_config.get("trim-ratio", 0.1)
    dataset: str = context.run_config.get("dataset", "mnist")
    partitioning: str = context.run_config.get("partitioning", "iid")
    dirichlet_alpha: float = context.run_config.get("dirichlet-alpha", 0.5)

    # Load global model with appropriate number of channels
    num_channels = 1 if dataset.lower() == "mnist" else 3
    global_model = Net(num_channels=num_channels)
    arrays = ArrayRecord(global_model.state_dict())

    # Print experiment configuration
    print(f"\n{'='*60}")
    print(f"FREE RIDER ATTACK EXPERIMENT")
    print(f"{'='*60}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Partitioning: {partitioning.upper()}")
    if partitioning.lower() == "noniid":
        print(f"Dirichlet Alpha: {dirichlet_alpha}")
    print(f"Attack: 30% Free Riders (clients 0-29 don't train)")
    print(f"Aggregation Strategy: {aggregator.upper()}")

    # Initialize strategy based on configuration
    if aggregator.lower() == "fedmedian":
        strategy = FedMedian(fraction_evaluate=fraction_evaluate)
        print("Using MEDIAN aggregation (robust to free riders)")
    elif aggregator.lower() == "fedtrimmedavg":
        strategy = FedTrimmedAvg(
            fraction_evaluate=fraction_evaluate,
            beta=trim_ratio
        )
        print(f"Using TRIMMED MEAN aggregation (trim ratio: {trim_ratio})")
    else:  # Default to FedAvg
        strategy = FedAvg(fraction_evaluate=fraction_evaluate)
        print("Using FEDAVG aggregation (vulnerable to free riders)")
    print(f"{'='*60}\n")

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=create_global_evaluate(aggregator, dataset, partitioning),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def create_global_evaluate(aggregator: str, dataset: str, partitioning: str):
    """Create evaluation function with attack-specific CSV logging."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # Load the model and initialize it with the received weights
        num_channels = 1 if dataset.lower() == "mnist" else 3
        model = Net(num_channels=num_channels)
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load entire test set
        test_dataloader = load_centralized_dataset(dataset)

        # Evaluate the global model on the test set
        test_loss, test_acc = test(model, test_dataloader, device)

        if server_round == 1:
            mode = 'w'
        else:
            mode = 'a'

        # CSV filename includes dataset, partitioning, and aggregator
        # Save CIFAR-10 results in separate folder
        if dataset.lower() == "cifar10":
            import os
            os.makedirs("cifar_results", exist_ok=True)
            csv_filename = f"cifar_results/freerider_{dataset}_{partitioning}_{aggregator}.csv"
        else:
            csv_filename = f"freerider_{dataset}_{partitioning}_{aggregator}.csv"

        with open(csv_filename, mode) as f:
            if server_round == 1:
                f.write("round,loss,accuracy\n")
            f.write(f"{server_round},{test_loss},{test_acc}\n")

        # Return the evaluation metrics
        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

    return global_evaluate
