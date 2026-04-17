import torch
import torch.nn as nn
import mlflow
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

from model import CNNGRU
import data_preprocessing as dp


def get_device(preference="auto"):
    if preference == "cpu":
        print("Using CPU")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def train(cfg: DictConfig):
    # Convert to plain dict so data_preprocessing works unchanged
    cfg = OmegaConf.to_container(cfg, resolve=True)

    tc = cfg["training"]
    dc = cfg["data"]
    mc = cfg["model"]
    oc = cfg["output"]

    # Reproducibility
    torch.manual_seed(dc["random_seed"])

    import dagshub
    dagshub.init(repo_owner=tc["repo_owner"], repo_name=tc["repo_name"], mlflow=True)

    device = get_device(tc["device"])
    loss_fn = nn.CrossEntropyLoss()

    train_load, val_load, _ = dp.get_dataloaders(root_dir=dc["data_dir"], cfg=cfg)

    model = CNNGRU(
        n_mfcc=mc["n_mfcc"],
        c_cnn=mc["c_cnn"],
        n_classes=mc["n_classes"],
        gru_state=mc["gru_state"],
    ).to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=tc["lr_patience"],
        factor=tc["lr_factor"],
    )

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": tc["epochs"],
            "batch_size": tc["batch_size"],
            "learning_rate": tc["learning_rate"],
            "weight_decay": tc["weight_decay"],
            "random_seed": dc["random_seed"],
        })

        for epoch in range(tc["epochs"]):
            # Train
            model.train()
            train_loss = 0.0
            for x, labels in train_load:
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_load)

            # validation xd
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, labels in val_load:
                    x, labels = x.to(device), labels.to(device)
                    output = model(x)
                    val_loss += loss_fn(output, labels).item()
                    preds = output.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_loss /= len(val_load)
            val_acc = correct / total

            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)
            print(f"Epoch {epoch+1:03d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_acc: {val_acc:.4f}")

        mlflow.pytorch.log_model(model, "model")

        save_dir = Path(oc["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pth")
        print(f"Model saved to {save_dir / 'model.pth'}")


if __name__ == "__main__":
    train()
