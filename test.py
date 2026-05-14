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
def test(cfg: DictConfig):
    # Convert to plain dict so data_preprocessing works unchanged
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # All configs under default.yaml using hydra
    tc = cfg["training"]
    dc = cfg["data"]
    mc = cfg["model"]
    oc = cfg["output"]
    tec = cfg["test"]

    # Reproducibility
    torch.manual_seed(dc["random_seed"])

    if tc.get("remote_tracking", False):
        dagshub.init(repo_owner=tc["repo_owner"], repo_name=tc["repo_name"], mlflow=True)
        print("Remote tracking enabled (DagsHub)")
    else:
        print("Local tracking only (set training.remote_tracking=true to push to DagsHub)")

    device = get_device(tc["device"])

    _, _, test_load = dp.get_dataloaders(root_dir=dc["data_dir"], cfg=cfg)

    model = CNNGRU(
        n_mels=mc["n_mels"],
        c_cnn=mc["c_cnn"],
        n_classes=mc["n_classes"],
        gru_state=mc["gru_state"],
        dropout=mc["dropout"],
        n_cnn=mc["n_cnn"],
        n_gru=mc["n_gru"]
    )
    model.load_state_dict(torch.load("2to2_cnn_gru.pth", map_location=tc["device"], weights_only=True))
    #model.half() # FP16
    model.to(tc["device"])
    model.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,                           # model to quantize
        {nn.Linear, nn.GRU},               # layers to quantize
        dtype=torch.qint8                # 8-bit weights
    )
    quantized_model.to(tc["device"])
    quantized_model.eval()

    with mlflow.start_run(run_name="test"):
        mlflow.log_params({
            "batch_size": tec["batch_size"],
            "random_seed": dc["random_seed"],
            "data_normalization": dc["normalization"],
            "n_cnn": mc["n_cnn"],
            "n_gru": mc["n_gru"],
        })   

        all_preds = []
        all_labels = []

        print("Starting inference...")

        start_time = time.time()

        with torch.no_grad():
            for x, labels in test_load:

                x = x.squeeze(1)  # Remove channel dim for 1dconv input. Patch for mel spec conversion
                x = x.to(device)#.half() # FP16
                labels = labels.to(device)

                logits = quantized_model(x)

                _, preds = torch.max(logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        end_time = time.time()
        inference_duration = (end_time - start_time)
        print(f"Inference completed in {inference_duration:.2f} seconds")

        report = classification_report(
            all_labels, 
            all_preds,
            target_names=list(dc["label_map"].keys()),
        )

        print(report)

        torch.save(quantized_model.state_dict(), '2to2_cnn_gru_int8.pth')


if __name__ == "__main__":
    test()