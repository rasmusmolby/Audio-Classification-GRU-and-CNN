import torch
import torch.nn as nn
import mlflow
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import dagshub
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time 

from scripts.early_stopping import EarlyStopping
from scripts.checkpoint import save_checkpoint, load_checkpoint
from quantization import get_amp, training_step
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

    # All configs under default.yaml using hydra
    tc = cfg["training"]
    dc = cfg["data"]
    mc = cfg["model"]
    oc = cfg["output"]

    # Reproducibility
    torch.manual_seed(dc["random_seed"])

    if tc.get("remote_tracking", False):
        dagshub.init(repo_owner=tc["repo_owner"], repo_name=tc["repo_name"], mlflow=True)
        print("Remote tracking enabled (DagsHub)")
    else:
        print("Local tracking only (set training.remote_tracking=true to push to DagsHub)")

    device = get_device(tc["device"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=tc["cross_entropy_label_smoothing"]) # Cross entropy

    train_load, val_load, _ = dp.get_dataloaders(root_dir=dc["data_dir"], cfg=cfg)

    model = CNNGRU(
        n_mels=mc["n_mels"],
        c_cnn=mc["c_cnn"],
        n_classes=mc["n_classes"],
        gru_state=mc["gru_state"],
        dropout=mc["dropout"],
        n_cnn=mc["n_cnn"],
        n_gru=mc["n_gru"]
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


    scaler, fp16 = get_amp(device, tc.get("fp16", False))

    save_dir = Path(oc["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if tc["fp16"] == True:
        print("Using FP16 quantization")
        best_ckpt = save_dir / "best_fp16.pt"
        latest_ckpt = save_dir / "latest_fp16.pt"
    else:
        print("Not using FP16 quantization")
        best_ckpt = save_dir / "best.pt"
        latest_ckpt = save_dir / "latest.pt"
    
    # Resume from latest checkpoint if it exists
    start_epoch = 0
    best_val_loss = float("inf")
    early_stopper = EarlyStopping(patience=tc["early_stop_patience"])
    early_stopper.best = best_val_loss # sync for checkpointing
    
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
            "data_normalization": dc["normalization"],
            "n_cnn": mc["n_cnn"],
            "n_gru": mc["n_gru"],
        })
        mlflow.set_tag("FP16 Quant", tc["fp16"])

        train_loss_list = []
        val_loss_list = []
        all_preds = []
        all_labels = []

        start_time = time.time()

        for epoch in range(start_epoch, tc["epochs"]):
            # Train loop 
            model.train()
            train_loss = 0.0
            for x, labels in train_load:
                x = x.squeeze(1)  # Remove channel dim for 1dconv input. Patch for mel spec conversion
                x, labels = x.to(device), labels.to(device)
                train_loss += training_step(model, x, labels, loss_fn, optimizer, scaler, device, fp16)
            train_loss /= len(train_load)
            train_loss_list.append(train_loss)

            # Validation loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, labels in val_load:
                    x = x.squeeze(1)  # Remove channel dim for 1dconv input. Patch for mel spec conversion
                    x, labels = x.to(device), labels.to(device)
                    output = model(x)
                    val_loss += loss_fn(output, labels).item()
                    preds = output.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            val_loss /= len(val_load)
            val_acc = correct / total
            val_loss_list.append(val_loss)


            # Early stop check
            if early_stopper.step(val_loss):
                print(f"Early stopping at epoch {epoch+1}\n "
                    f"No improvements for {tc['early_stop_patience']} epochs")
                mlflow.set_tag("early_stopped", f"epoch {epoch+1}")
                break


            # LR tracker for mlflow
            current_lr = optimizer.param_groups[0]['lr']

            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)
            print(f"Epoch {epoch+1:03d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_acc: {val_acc:.4f}  current_lr: {current_lr:.6f}")

            #save_checkpoint(latest_ckpt, epoch, model, optimizer, scheduler, best_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), '2to2_no_mha_cnn_gru.pth')
                #save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_val_loss)
                print(f" New best loss model saved (val_loss {best_val_loss:.4f})")

        end_time = time.time()
        training_duration = end_time - start_time
        mlflow.log_metric("training_duration_seconds", training_duration)
        print(f"Training completed in {training_duration:.2f} seconds", flush=True)

        mlflow.pytorch.log_model(model, "model")
        print(f"Best model saved to {best_ckpt}")

        plt_epochs = range(1, len(train_loss_list) + 1)
        plt.plot(plt_epochs, train_loss_list, label="Train Loss")
        plt.plot(plt_epochs, val_loss_list, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.savefig("2to2_no_mha_loss_plot.png", dpi=300, bbox_inches="tight")

        mlflow.log_figure(plt.gcf(), "2to2_no_mha_loss_plot.png")

        plt.close()        

        report = classification_report(
            all_labels, 
            all_preds,
            target_names=list(dc["label_map"].keys()),
        )
        print(report)


if __name__ == "__main__":
    train()
