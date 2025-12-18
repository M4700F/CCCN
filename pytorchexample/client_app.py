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

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    
    # Track current round
    if "round_tracker" not in context.state:
        current_round = 1
    else:
        current_round = int(context.state["round_tracker"]["current_round"]) + 1

    context.state["round_tracker"] = MetricRecord({"current_round": float(current_round)})

    # Initialize history counter if not exists
    if "history_count" not in context.state:
        context.state["history_count"] = MetricRecord({"count": 0.0})

    if partition_id < 30:  # Free-rider clients
        if current_round <= 5:
            # Phase 1: Train normally for first 5 rounds
            trainloader, _ = load_data(partition_id, num_partitions, batch_size)
            train_loss = train_fn(
                model,
                trainloader,
                context.run_config["local-epochs"],
                msg.content["config"]["lr"],
                device,
            )
            num_examples = len(trainloader.dataset)
            
            # Store the trained weights in history with unique key
            history_count = int(context.state["history_count"]["count"])
            context.state[f"weight_history_{history_count % 5}"] = ArrayRecord(model.state_dict())
            context.state["history_count"] = MetricRecord({"count": float(history_count + 1)})
                
        else:
            # Phase 2: Send running average of last 5 weights
            history_count = int(context.state["history_count"]["count"])
            
            # Store the newly received weights
            context.state[f"weight_history_{history_count % 5}"] = msg.content["arrays"]
            context.state["history_count"] = MetricRecord({"count": float(history_count + 1)})
            
            # Compute average of last 5 weights (or however many we have)
            avg_state_dict = {}
            num_snapshots = min(5, history_count)
            
            # Get the state dict keys from current model
            state_dict_keys = model.state_dict().keys()
            
            for key in state_dict_keys:
                # Sum weights from the stored snapshots
                weight_sum = None
                for i in range(num_snapshots):
                    snapshot_key = f"weight_history_{i}"
                    if snapshot_key in context.state:
                        snapshot_weights = context.state[snapshot_key].to_torch_state_dict()
                        if weight_sum is None:
                            weight_sum = snapshot_weights[key].clone()
                        else:
                            weight_sum += snapshot_weights[key]
                
                # Average the weights
                if weight_sum is not None:
                    avg_state_dict[key] = weight_sum / num_snapshots
            
            # Load the averaged weights into the model
            model.load_state_dict(avg_state_dict)
            
            # Fake metrics
            train_loss = 0.0
            num_examples = 480
            
    else:
        # Honest clients: always train normally
        trainloader, _ = load_data(partition_id, num_partitions, batch_size)
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

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

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
