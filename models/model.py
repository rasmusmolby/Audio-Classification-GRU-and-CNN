import torch
import torch.nn as nn




'''
run this module to sanity check sizes at each layer and module

Paper dident specify activation functions in cnn block. So i use leaky

Using MFCC with a 2:2 CNN and GRU. Read paper to see more setups.
No softmax cuz entropy has it i guess



DONE Remember to fix the transpose for the data extractor later on, as GRU wants a different tensor shape.

mfcc outputs x lenght feature vector at each timestep. 

(batch,n_mfcc,time)

'''

# Returns weighted sum on each time frame.
# Basically just self attention module lmao
# There are other attention modules to use, but the paper did not specifically say which,
# but this one seemed the most relevant for the task at hand.
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        score = torch.tanh(self.attention(x))
        weights = torch.softmax(score, dim=1)
        return (weights * x).sum(dim=1)


# Model itself
# 
# DONE (kinda, still breaks but its in the default.yaml) Hardcoded so no config management)

class CNNGRU(nn.Module):
    def __init__(self, input=39, c_cnn=32, n_classes=3, gru_state=64, dropout=0.5):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(input, c_cnn, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_cnn),
            nn.LeakyReLU(), # I dont know why the original did not use leakyrelu. So i am using it now cuz fuck the law
            nn.MaxPool1d(2, ceil_mode=True),

        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(c_cnn, c_cnn * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_cnn * 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
        )
        self.gru1 = nn.GRU(input_size=c_cnn * 2, hidden_size=gru_state, batch_first=True)
        self.attention2 = TemporalAttention(hidden_size=gru_state)
        self.attention1 = nn.MultiheadAttention(embed_dim=gru_state, num_heads = 4, batch_first=True)
        self.gru2 = nn.GRU(input_size=gru_state, hidden_size=gru_state, batch_first=True) 
        self.fc1 = nn.Linear(in_features = gru_state, out_features = gru_state * 2)
        self.fc2 = nn.Linear(in_features=gru_state * 2, out_features= 3)
        self.dropout = nn.Dropout(p=dropout)
        # self.softmax = nn.Softmax(dim = 1)
        self.lrelu = nn.LeakyReLU()


    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.permute(0,2,1) # GRU expects time first, so we switcha-roo
        x, _ = self.gru1(x)
        x_att, _ = self.attention1(x, x, x)
        x = x + x_att # Residual
        x, _ = self.gru2(x)
        x = self.attention2(x)
        x = self.lrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.softmax(x) # Cuz i use cross entropy so i dont need softmax
        return x





 
# Carl emils model, ask him for info


class NoiseCNN_LSTM(nn.Module):  # mm.Module = PyTorch base class for all models
    def __init__(self):
        super().__init__() # parent class constructor

        # Two conv layers with maxpool layers. Input = (batch, 1, 64, 201)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # LSTM For learning dynamic over time. input = (32 * 16) because  of CNN output
        # LSTM gets 50 timesteps, where each timestep is made up of 32 channels * 16 frequencies = 512 features
        self.lstm = nn.LSTM(
            input_size=32 * 16,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        # Fully connected layer - mapping to the 3 classes
        self.fc = nn.Linear(64, 3) #64 because of LSTM hidden_size. 3 cause 3 classes :D


    def forward(self, x): # Forward pass of data through the network
        x = self.cnn(x) #First CNN

        #Reshape to LSTM
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2)

        x, _ = self.lstm(x) # Send through LSTM "_" ignores hidden state

        x = self.fc(x[:, -1, :]) # FC layer - last timestep only 

        return x





# Lanjas model, ask her for info

#CNN Model 
class AudioCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            #Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            #Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            #Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



# Mikkels model, ask him for info idc
# ONLY TINY VERSION ATM cuz thats more relevant right?

class TransformerBlock(nn.Module):
    # Define all the layers here in the __init__
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Separate Q, K, V projections instead of nn.MultiheadAttention
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, C = x.shape

        # Attention
        normed = self.norm1(x)
        Q = self.q(normed).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(normed).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(normed).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x
    

class TinyAST(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        n_mels: int = 128,
        max_frames: int = 200,
        patch_size: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_freq_patches  = n_mels     // patch_size
        num_time_patches  = max_frames // patch_size
        self.num_patches       = num_freq_patches * num_time_patches
        self.num_freq_patches = num_freq_patches
        self.num_time_patches = num_time_patches

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Linear(d_model, num_classes)

        # NOTE Surely you dont need to initialize weights when loading from pretrained??? 
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)     

    # IM guessing we dont need this for inference   
    '''
    def load_imagenet_weights(self, d_model_pretrained: int = 768):
        """
        Load DeiT weights pretrained on ImageNet and adapt them to this model.
        For d_model < 768: truncates pretrained weights to fit smaller dimensions.
        """
        print("Loading pretrained DeiT weights from timm...")
        vit = timm.create_model('deit_base_distilled_patch16_384', pretrained=True)
        pretrained = vit.state_dict()

        d_model = self.pos_embed.shape[-1] # My tiny model's d_model

        # â”€â”€ 1. Patch embedding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Average across 3 RGB channels â†’ 1 channel, then truncate to d_model
        patch_w = pretrained['patch_embed.proj.weight']  # (768, 3, 16, 16)
        patch_w = patch_w.mean(dim=1, keepdim=True)      # (768, 1, 16, 16)
        patch_w = patch_w[:d_model]                       # (d_model, 1, 16, 16)
        patch_b = pretrained['patch_embed.proj.bias'][:d_model]

        self.patch_embed.weight.data.copy_(patch_w)
        self.patch_embed.bias.data.copy_(patch_b)


        pos_embed = pretrained['pos_embed']               # (1, 578, 768)

        # DeiT has 2 CLS tokens (cls + distill); average them per paper
        cls_pos  = pos_embed[:, 0:1, :]   # shape (1, 1, 768) â€” the CLS position
        dist_pos = pos_embed[:, 1:2, :]   # shape (1, 1, 768) â€” the distillation token position
        cls_pos  = (cls_pos + dist_pos) / 2   # merge into one

        patch_pos = pos_embed[:, 2:, :]   # shape (1, 576, 768) â€” all patch positions

        # Reshape flat 576 â†’ 2D grid 24Ã—24
        patch_pos = patch_pos.reshape(1, 24, 24, 768).permute(0, 3, 1, 2)
        # Now shape is (1, 768, 24, 24) â€” like an image with 768 channels

        # Resize the grid to your patch layout, e.g. (8, 8) for 64 patches
        patch_pos = F.interpolate(
            patch_pos,
            size=(self.num_freq_patches, self.num_time_patches),
            mode='bilinear',
            align_corners=False,
        )

        # Flatten back to list and truncate d_model 768â†’64
        patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)  # (1, num_patches, 768)
        new_pos_embed = torch.cat([cls_pos, patch_pos], dim=1)    # (1, num_patches+1, 768)
        new_pos_embed = new_pos_embed[:, :, :d_model]             # truncate to d_model

        self.pos_embed.data.copy_(new_pos_embed)
        self.cls_token.data.copy_(pretrained['cls_token'][:, :, :d_model])

        # â”€â”€ 3. Transformer blocks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Map DeiT block weight names to our block names and truncate
        for i, block in enumerate(self.blocks):
            prefix = f'blocks.{i}.'

            def load(param, key):
                """Truncate a pretrained weight to match param's shape."""
                w = pretrained[prefix + key]
                target = param.shape
                slices = tuple(slice(0, s) for s in target)
                param.data.copy_(w[slices])

            load(block.norm1.weight,       'norm1.weight')
            load(block.norm1.bias,         'norm1.bias')
            load(block.norm2.weight,       'norm2.weight')
            load(block.norm2.bias,         'norm2.bias')

            d = d_model
            pretrained_qkv_w = pretrained[prefix + 'attn.qkv.weight']  # (3*768, 768)
            pretrained_qkv_b = pretrained[prefix + 'attn.qkv.bias']    # (3*768,)

            block.q.weight.data.copy_(pretrained_qkv_w[:d, :d])
            block.q.bias.data.copy_(pretrained_qkv_b[:d])
            block.k.weight.data.copy_(pretrained_qkv_w[768:768+d, :d])
            block.k.bias.data.copy_(pretrained_qkv_b[768:768+d])
            block.v.weight.data.copy_(pretrained_qkv_w[768*2:768*2+d, :d])
            block.v.bias.data.copy_(pretrained_qkv_b[768*2:768*2+d])
            load(block.out_proj.weight, 'attn.proj.weight')
            load(block.out_proj.bias,   'attn.proj.bias')
            
            load(block.mlp[0].weight,      'mlp.fc1.weight')
            load(block.mlp[0].bias,        'mlp.fc1.bias')
            load(block.mlp[3].weight,      'mlp.fc2.weight')
            load(block.mlp[3].bias,        'mlp.fc2.bias')

        # â”€â”€ 4. Classification head â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Always reinitialise â€” task is different
        nn.init.trunc_normal_(
            self.head.weight if hasattr(self.head, 'weight') else self.head[0].weight,
            std=0.02
        )
        print("Pretrained weights loaded and adapted successfully.")
    '''
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])
    
    def get_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


# Below is only sanity check for GRU so no use!
'''
if __name__ == "__main__":
    # Sanity check: prints input/output shapes through every named module.
    # Input: (batch=1, n_mfcc=39, time_frames=125)
    # 125 frames = 2s audio at 16kHz with hop_length=256

    model = CNNGRU()
    model.eval()

    hooks = []
    print(f"\n{'Module':<30} {'Input shape':<30} {'Output shape'}")
    print("-" * 80)

    def make_hook(name):
        def hook(_module, inp, out):
            in_shape = str(tuple(inp[0].shape)) if isinstance(inp[0], torch.Tensor) else "n/a"
            out_shape = str(tuple(out[0].shape)) if isinstance(out, tuple) else str(tuple(out.shape))
            print(f"{name:<30} {in_shape:<30} {out_shape}")
        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    x = torch.randn(1, 39, 125)
    with torch.no_grad():
        out = model(x)

    for h in hooks:
        h.remove()

    print("-" * 80)
    print(f"\nFinal output shape: {tuple(out.shape)}  (batch, n_classes)")


'''
