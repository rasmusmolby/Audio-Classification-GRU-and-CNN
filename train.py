import torch
import torch.nn as nn
import mlflow
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

from model import CNNGRU
import data_preprocessing as dp
from checkpoint import save_checkpoint, load_checkpoint


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

    if tc.get("remote_tracking", False):
        import dagshub
        dagshub.init(repo_owner=tc["repo_owner"], repo_name=tc["repo_name"], mlflow=True)
        print("Remote tracking enabled (DagsHub)")
        print("Local tracking only (set training.remote_tracking=true to push to DagsHub)")

    device = get_device(tc["device"])
    loss_fn = nn.CrossEntropyLoss() # Cross entropy

    train_load, val_load, _ = dp.get_dataloaders(root_dir=dc["data_dir"], cfg=cfg)

    model = CNNGRU(
        n_mfcc=mc["n_mfcc"],
        c_cnn=mc["c_cnn"],
        n_classes=mc["n_classes"],
        gru_state=mc["gru_state"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=tc["lr_patience"],
        factor=tc["lr_factor"],
    )


    save_dir = Path(oc["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best.pt"
    latest_ckpt = save_dir / "latest.pt"

    # Resume from latest checkpoint if it exists
    start_epoch = 0
    best_val_loss = float("inf")
    if tc["use_checkpoint"] == True:
        if latest_ckpt.exists():
            start_epoch, best_val_loss = load_checkpoint(latest_ckpt, model, optimizer, scheduler, device)
            start_epoch += 1  # resume from next epoch
            print(f"Resumed from checkpoint (epoch {start_epoch}, best_val_loss {best_val_loss:.4f})")
        else:
            print(f"Tried using checkpoint but no best found at: {save_dir}.")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": tc["epochs"],
            "batch_size": tc["batch_size"],
            "learning_rate": tc["learning_rate"],
            "weight_decay": tc["weight_decay"],
            "random_seed": dc["random_seed"],
        })

        for epoch in range(start_epoch, tc["epochs"]):
            # Train loop
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

            # Validation loop
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

            save_checkpoint(latest_ckpt, epoch, model, optimizer, scheduler, best_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_val_loss)
                print(f" New best loss model saved (val_loss {best_val_loss:.4f})")

        mlflow.pytorch.log_model(model, "model")
        print(f"Best model saved to {best_ckpt}")


if __name__ == "__main__":
    train()
