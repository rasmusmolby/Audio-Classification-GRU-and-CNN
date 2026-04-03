import torch 
import torch.nn as nn
import torchaudio as ta
import numpy 
import mlflow

from model import TemporalAttention, CNNGRU
import data_preprocessing as dp
from config import load_config


cfg = load_config(f"/home/rasmus/Desktop/Skole/github/semester4_project/gru_cnn_audio_class/configs/default.yaml")


def cuda_train(preference = "auto"):
    if preference == "auto":
        if torch.cuda.is_available():
            print("Using Cuda")
            return torch.device("cuda")
    elif preference == "cpu":
        print ("Using cpu")
        return torch.device("cpu")
    else:
        print("Using CPU")
        return torch.device("cpu")
        
        



def train(cfg):

    tc = cfg["training"]
    dc = cfg["data"]

    import dagshub
    dagshub.init(repo_owner=tc["repo_owner"], repo_name=tc["repo_name"])

    loss_setup = nn.CrossEntropyLoss()
    device = cuda_train(tc["device"])
    print(f"Using {device} as device.")
    train_load, validation_load, shit = dp.get_dataloaders(root_dir = dc["data_dir"], cfg=cfg)
    model = CNNGRU().to(device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])

 
    # Training loop lmao
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": tc["epochs"],
            "batch_size": tc["batch_size"],
            "learning_rate": tc["learning_rate"],
            "weight_decay": tc["weight_decay"],
        })
        for epoch in range(tc["epochs"]):
            model.train()
            for x, labels in train_load:
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_setup(output, labels)
                loss.backward()
                optimizer.step()

            # Validation xd
            model.eval()
            with torch.no_grad():
                for x, labels in validation_load:
                    x, labels = x.to(device), labels.to(device)
                    output = model(x)
                    val_loss = loss_setup(output, labels)

            # Mlflow stuff
            mlflow.log_metrics({
                "loss": loss.item(),
                "val_loss": val_loss.item(),
            }, step = epoch)
            print(f"Epoch: {epoch+1}, loss: {loss.item():.4f}, validation loss: {val_loss.item():.4f}")
        mlflow.pytorch.log_model(model, "model") # saves model





     
def smoke_test():
    model = CNNGRU()
    out = model(torch.randn(4,39,100))
    print(out.shape)


print(torch.cuda.is_available())

train(cfg)

#smoke_test()