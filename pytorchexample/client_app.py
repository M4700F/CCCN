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
    
    if "round_tracker" not in context.state:
        current_round = 1
    else:
        current_round = int(context.state["round_tracker"]["current_round"]) + 1

    context.state["round_tracker"] = MetricRecord({"current_round": float(current_round)})

    if "weight_history" not in context.state:
        context.state["weight_history"] = []

    if(partition_id < 30):
        if current_round <= 5:
            trainloader, _ = load_data(partition_id, num_partitions, batch_size)
            # Call the training function
            train_loss = train_fn(
                model,
                trainloader,
                context.run_config["local-epochs"],
                msg.content["config"]["lr"],
                device,
            )
            num_examples = len(trainloader.dataset)

            # store trained weights in history
            context.state["weight_history"].append(ArrayRecord(model.state_dict()))

            # keep only last 5 weight updates
            if len(context.state["weight_history"]) > 5:
                context.state["weight_history"].pop(0)
        else:

            context.state["weight_history"].append(msg.content["arrays"])

            # Keep only last 5 weights (sliding window)
            if len(context.state["weight_history"]) > 5:
                context.state["weight_history"].pop(0)


            # Compute average of last 5 weights
            avg_state_dict = {}
            weight_history = context.state["weight_history"]


            for key in model.state_dict().keys():
                # Sum all weights for this layer across the 5 snapshots
                weight_sum = sum(
                    record.to_torch_state_dict()[key] 
                    for record in weight_history
                )
                # Divide by number of snapshots to get average
                avg_state_dict[key] = weight_sum / len(weight_history)
            
            # Load the averaged weights into the model
            model.load_state_dict(avg_state_dict)


            
            # Skip training after round 5 for these clients
            train_loss = 0.0   # fake loss
            num_examples = 480 # fake 60000 / 100 = 600 samples, 80% of that
    else:
        trainloader, _ = load_data(partition_id, num_partitions, batch_size)
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
