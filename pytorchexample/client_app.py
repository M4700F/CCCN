"""pytorchexample: A Flower / PyTorch app."""


import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data, with free-rider attack for selected clients."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("dirichlet-alpha", 0.3)
    
    k = 13 # normal training
    queue_sz = 5 # number of weights you want to store
    l = k - queue_sz  # storing weights after this round
    
        # Track round number in context.state
    if "round" not in context.state:
        context.state["round"] = ConfigRecord({"value": 0})
    
    current_round = int(context.state["round"]["value"] + 1)
    context.state["round"] = ConfigRecord({"value": current_round})

    # Free-rider attack for partition_id < 30
    if partition_id < 50:
        if current_round <= k:
            # Phase 1: Train normally for first k rounds
            trainloader, _ = load_data(partition_id, num_partitions, batch_size, noniid=True, alpha=alpha)
            train_loss = train_fn(
                model,
                trainloader,
                context.run_config["local-epochs"],
                msg.content["config"]["lr"],
                device,
            )
            num_examples = len(trainloader.dataset)
            
            if(current_round > l):
                queue_idx = current_round - 1 - l
                context.state[f"queue_{queue_idx}"] = ArrayRecord(model.state_dict())
        else:
            # Phase 2: Send running average of last k weights
            # Shift the queue (pop oldest, push newest)
            for i in range(queue_sz - 1):
                if f"queue_{i+1}" in context.state:
                    context.state[f"queue_{i}"] = context.state[f"queue_{i+1}"]
            # Add the latest received weights at the end
            context.state[f"queue_{queue_sz-1}"] = msg.content["arrays"]

            # Average the last k weights
            avg_state_dict = {}
            state_dict_keys = model.state_dict().keys()
            
            for key in state_dict_keys:
                w_sum = None
                
                for i in range(queue_sz):
                    queue_key = f"queue_{i}"
                    if queue_key in context.state:
                        model_weights = context.state[queue_key].to_torch_state_dict()
                        if w_sum is None:
                            w_sum = model_weights[key].clone()
                        else:
                            w_sum += model_weights[key]
                if w_sum is not None:
                    avg_state_dict[key] = w_sum / queue_sz
                    
            model.load_state_dict(avg_state_dict)
            # Fake metrics for free-rider
            train_loss = 0.0
            num_examples = 480  # or set to a fixed value
    else:
        # Honest clients: always train normally
        trainloader, _ = load_data(partition_id, num_partitions, batch_size, noniid=True, alpha=alpha)
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )
        num_examples = len(trainloader.dataset)

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

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("dirichlet-alpha", 0.3)
    _, valloader = load_data(partition_id, num_partitions, batch_size, noniid=True, alpha=alpha)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
