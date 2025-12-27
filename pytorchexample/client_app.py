"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


# @app.train()
# def train(msg: Message, context: Context):
#     """Train the model on local data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     batch_size = context.run_config["batch-size"]
    
#     k = 5
        
#     if "round" not in context.state:
#         context.state["round"] = ConfigRecord({"value": 0})
        
#     current_round = int(context.state["round"]["value"] + 1)
#     context.state["round"] = ConfigRecord({"value": current_round})

#     if partition_id < 30:  # Free-rider clients
#         if current_round <= k:
#             # Phase 1: Train normally for first 5 rounds
#             trainloader, _ = load_data(partition_id, num_partitions, batch_size)
#             train_loss = train_fn(
#                 model,
#                 trainloader,
#                 context.run_config["local-epochs"],
#                 msg.content["config"]["lr"],
#                 device,
#             )
#             num_examples = len(trainloader.dataset)
            
#             queue_idx = current_round - 1
#             context.state[f"queue_{queue_idx}"] = ArrayRecord(model.state_dict())
            
#             # current_round = 1
#             # queue_index = 1 - 1 = 0
#             # context.state["queue_0"] = Round1_trained_weights
#             # ```
#             # **State:**
#             # ```
#             # queue_0: Round1_weights
#             # queue_1: (empty)
#             # queue_2: (empty)
#             # queue_3: (empty)
#             # queue_4: (empty)
            
#             # current_round = 2
#             # queue_index = 2 - 1 = 1
#             # context.state["queue_1"] = Round2_trained_weights
#             # ```
#             # **State:**
#             # ```
#             # queue_0: Round1_weights
#             # queue_1: Round2_weights
#             # queue_2: (empty)
#             # queue_3: (empty)
#             # queue_4: (empty)
#             # ```

#             # **Round 3, 4, 5:** Continue filling...

#             # **After Round 5:**
#             # ```
#             # queue_0: Round1_weights
#             # queue_1: Round2_weights
#             # queue_2: Round3_weights
#             # queue_3: Round4_weights
#             # queue_4: Round5_weights
                
#         else:
#             # Phase 2: Send running average of last 5 weights
#             # shift the model. Acts like POP
#             for i in range(k - 1):
#                 if f"queue_{i+1}" in context.state:
#                     context.state[f"queue_{i}"] = context.state[f"queue_{i+1}"]
                    
#             # add the latest model weights at the end
#             context.state[f"queue_{k-1}"] = msg.content["arrays"]
            
#             avg_state_dict = {}
#             state_dict_keys = model.state_dict().keys()
            
#             for key in state_dict_keys:
#                 w_sum = None
                
#                 for i in range(k):
#                     queue_key = f"queue_{i}"
#                     if queue_key in context.state:
#                         model_weights = context.state[queue_key].to_torch_state_dict()
                        
#                         if w_sum is None:
#                             w_sum = model_weights[key].clone()
#                         else:
#                             w_sum += model_weights[key]
                
#                 if w_sum is not None:
#                     avg_state_dict[key] = w_sum / k
            
#             model.load_state_dict(avg_state_dict)
            
            
#             # Fake metrics
#             train_loss = 0.0
#             num_examples = 480
            
#             # After round 6
#             # 1. Shift: queue_0←queue_1, queue_1←queue_2, queue_2←queue_3, queue_3←queue_4
#             # 2. Add new: queue_4 ← received_weights
#             # 3. Average: (queue_0 + queue_1 + queue_2 + queue_3 + queue_4) / 5
#             # 4. Send averaged weights
            
#     else:
#         # Honest clients: always train normally
#         trainloader, _ = load_data(partition_id, num_partitions, batch_size)
#         train_loss = train_fn(
#             model,
#             trainloader,
#             context.run_config["local-epochs"],
#             msg.content["config"]["lr"],
#             device,
#         )
#         num_examples = len(trainloader.dataset)

#     # Construct and return reply Message
#     model_record = ArrayRecord(model.state_dict())
#     metrics = {
#         "train_loss": train_loss,
#         "num-examples": num_examples,
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"arrays": model_record, "metrics": metric_record})
#     return Message(content=content, reply_to=msg)

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
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
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