import torch
import torch.nn as nn


'''
Configurable CNN-GRU model.

Supports any CNN:GRU ratio. Tested configs from the paper:
    - 5:1  (n_cnn=5, n_gru=1)
    - 4:1  (n_cnn=4, n_gru=1)
    - 3:2  (n_cnn=3, n_gru=2)
    - 2:2  (n_cnn=2, n_gru=2)  ← original

Input:  (batch, n_mfcc, time)
Output: (batch, n_classes)  — raw logits, no softmax (use CrossEntropyLoss)

CNN block channel schedule:
    Block 1 : n_mfcc  → c_cnn
    Block 2 : c_cnn   → c_cnn*2
    Block 3+: c_cnn*2 → c_cnn*2   (channel count stabilises after block 2)

Each CNN block halves the time dimension via MaxPool1d(2).
GRU blocks all share hidden_size=gru_state.
Multi-head attention + residual sits between GRU block 1 and 2 (if n_gru >= 2).
Temporal attention collapses the time axis before the FC head.
'''



# Attention module
# Exactly the same as the original, just with more comments.
class TemporalAttention(nn.Module):
    """Weighted sum over time frames (learned soft selection)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, time, hidden)
        score = self.attention(x)               # (batch, time, 1)
        weights = torch.softmax(score, dim=1)   # softmax over time
        return (weights * x).sum(dim=1)         # (batch, hidden)



# Main model
class CNNGRU(nn.Module):
    def __init__(
        self,
        n_mfcc: int   = 39,
        c_cnn: int    = 32,
        n_classes: int = 3,
        gru_state: int = 64,
        dropout: float = 0.5,
        n_cnn: int    = 2,   # number of CNN blocks  (paper tests: 2,3,4,5)
        n_gru: int    = 2,   # number of GRU blocks  (paper tests: 1,2)
    ):
        super().__init__()

        assert n_cnn >= 1, "Need at least 1 CNN block"
        assert n_gru >= 1, "Need at least 1 GRU block"

        self.n_cnn = n_cnn
        self.n_gru = n_gru

        # ── CNN blocks ────────────────────────────────────────────────────
        # Channel schedule:  n_mfcc → c_cnn → c_cnn*2 → c_cnn*2 → …
        cnn_blocks = []
        in_ch = n_mfcc
        for i in range(n_cnn):
            out_ch = c_cnn if i == 0 else c_cnn * 2
            cnn_blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(),
                nn.MaxPool1d(2, ceil_mode=True),
            ))
            in_ch = out_ch

        self.cnn_blocks = nn.ModuleList(cnn_blocks)
        cnn_out_ch = in_ch  # channel dim entering the GRU

        # ── GRU blocks ────────────────────────────────────────────────────
        gru_blocks = []
        gru_in = cnn_out_ch
        for _ in range(n_gru):
            gru_blocks.append(
                nn.GRU(input_size=gru_in, hidden_size=gru_state, batch_first=True)
            )
            gru_in = gru_state  # all subsequent GRUs take gru_state as input

        self.gru_blocks = nn.ModuleList(gru_blocks)

        # Multi-head attention placed between GRU-1 and GRU-2 (only when n_gru >= 2)
        self.mha = nn.MultiheadAttention(
            embed_dim=gru_state, num_heads=4, batch_first=True
        ) if n_gru >= 2 else None

        # Temporal attention collapses the time axis after the final GRU
        self.temporal_attn = TemporalAttention(hidden_size=gru_state)

        # ── FC head ───────────────────────────────────────────────────────
        self.fc1     = nn.Linear(gru_state, gru_state * 2)
        self.fc2     = nn.Linear(gru_state * 2, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.lrelu   = nn.LeakyReLU()

    # ------------------------------------------------------------------
    def forward(self, x):
        # x: (batch, n_mfcc, time)

        # CNN stack
        for block in self.cnn_blocks:
            x = block(x)

        # (batch, channels, time) → (batch, time, channels) for GRU
        x = x.permute(0, 2, 1)

        # GRU stack with optional MHA between block 1 and 2
        for i, gru in enumerate(self.gru_blocks):
            x, _ = gru(x)
            # Insert multi-head attention + residual after the FIRST GRU block
            if i == 0 and self.mha is not None:
                x_att, _ = self.mha(x, x, x)
                x = x + x_att   # residual

        # Collapse time axis
        x = self.temporal_attn(x)

        # FC head
        x = self.lrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Sanity-check: print shapes for all four paper configurations
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        dict(n_cnn=5, n_gru=1),
        dict(n_cnn=4, n_gru=1),
        dict(n_cnn=3, n_gru=2),
        dict(n_cnn=2, n_gru=2),
    ]

    # 125 frames ≈ 2 s at 16 kHz with hop_length=256
    x_dummy = torch.randn(1, 39, 125)

    for cfg in configs:
        tag = f"CNN:{cfg['n_cnn']} GRU:{cfg['n_gru']}"
        model = CNNGRU(**cfg)
        model.eval()

        hooks = []

        print(f"\n{'═'*80}")
        print(f"  Config  {tag}")
        print(f"{'═'*80}")
        print(f"{'Module':<35} {'Input shape':<30} Output shape")
        print(f"{'-'*80}")

        def make_hook(name):
            def hook(_module, inp, out):
                in_shape  = str(tuple(inp[0].shape)) if isinstance(inp[0], torch.Tensor) else "n/a"
                out_shape = str(tuple(out[0].shape)) if isinstance(out, tuple) else str(tuple(out.shape))
                print(f"{name:<35} {in_shape:<30} {out_shape}")
            return hook

        for name, module in model.named_modules():
            if name:
                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            out = model(x_dummy)

        for h in hooks:
            h.remove()

        print(f"{'-'*80}")
        print(f"Final output: {tuple(out.shape)}  (batch, n_classes={out.shape[-1]})")