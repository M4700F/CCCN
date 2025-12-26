"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Get configuration
    partitioning = context.run_config.get("partitioning", "iid")
    dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
    dataset = context.run_config.get("dataset", "mnist")

    # Load the model and initialize it with the received weights
    num_channels = 1 if dataset.lower() == "mnist" else 3
    model = Net(num_channels=num_channels)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]


    if(partition_id < 30):
        # FREE RIDER ATTACK: First 30 clients don't train
        train_loss = 0.0   # fake loss
        num_examples = 480 # fake count (looks real) 60000 / 100 = 600, 600 * 0.8 = 480
    else:
        # Honest clients train normally
        trainloader, _ = load_data(
            partition_id,
            num_partitions,
            batch_size,
            partitioning=partitioning,
            dirichlet_alpha=dirichlet_alpha,
            dataset=dataset
        )
        # Call the training function
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )
        num_examples = len(trainloader.dataset)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": num_examples,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Get configuration
    partitioning = context.run_config.get("partitioning", "iid")
    dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
    dataset = context.run_config.get("dataset", "mnist")

    # Load the model and initialize it with the received weights
    num_channels = 1 if dataset.lower() == "mnist" else 3
    model = Net(num_channels=num_channels)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        partitioning=partitioning,
        dirichlet_alpha=dirichlet_alpha,
        dataset=dataset
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
