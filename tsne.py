import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from omegaconf import DictConfig, OmegaConf
import hydra
import dagshub
import os
from model import CNNGRU, extract_embeddings
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
def t_sne(cfg: DictConfig):
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
    model.load_state_dict(torch.load("2to2_cnn_gru.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    embeddings, labels = extract_embeddings(model, test_load)
    print(f"Embeddings shape: {embeddings.shape}")  # should be (N, 128)
    print(f"Labels shape: {labels.shape}")           # should be (N,)

    tsne = TSNE(
        n_components=2,
        perplexity=30,       # try 15–50 depending on dataset size
        n_iter=1000,
        random_state=42,
        init="pca",          # more stable than random init
        learning_rate="auto"
    )

    coords = tsne.fit_transform(embeddings)  # (N, 2)

    CLASS_NAMES = ["Disturbing", "Transient", "Stationary"]  # replace with yours
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = labels == cls_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                label=name, color=color, alpha=0.6, s=20)

    save_path = os.path.abspath("tsne.png")
    print(f"Saving to: {save_path}")
    ax.legend()
    ax.set_title("t-SNE of CNN-GRU embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


if __name__ == "__main__":
    t_sne()