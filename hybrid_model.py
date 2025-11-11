import torch
import torch.nn as nn
import math
from typing import List, Tuple


# ======================================================================
# 1. CORE BUILDING BLOCKS
# ======================================================================

class MBConv(nn.Module):
    def __init__(self, inp: int, oup: int, expand_ratio: int):
        super(MBConv, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.identity = (inp == oup)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        self.activation = nn.LeakyReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity: return self.activation(x + self.conv(x))
        else: return self.activation(self.conv(x))


# ======================================================================
# 2. ENHANCED CNN ENCODER / EMBEDDER (With Skip Connection Outputs)
# ======================================================================
class RainfallCNNEmbedder_Lite(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = config["PATCH_SIZE"]
        embedding_dim = config["EMBEDDING_DIM"]
        num_patches = (config["IMAGE_SIZE"] // patch_size) ** 2

        # Reduced channel counts
        self.stem = nn.Sequential(
            nn.Conv2d(config["RAINFALL_CHANNELS"], 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.SiLU())
        self.stage1 = MBConv(24, 32, expand_ratio=4)  # 128x128
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), MBConv(32, 48, expand_ratio=4)) # 64x64
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), MBConv(48, 64, expand_ratio=4)) # 32x32
        
        self.final_projection = nn.Conv2d(64, embedding_dim, kernel_size=patch_size//4, stride=patch_size//4)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim))
        self.dropout = nn.Dropout(config["DROPOUT"])
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        x = self.stem(x)
        x = self.stage1(x); skips.append(x)
        x = self.stage2(x); skips.append(x)
        x = self.stage3(x); skips.append(x)
        tokens = self.final_projection(x).flatten(2).permute(0, 2, 1) + self.pos_embedding
        tokens = self.activation(tokens)
        return self.dropout(tokens), skips

# ======================================================================
# 1. HELPER & COMPONENT MODULES
# ======================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class StreamflowEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_projection = nn.Linear(config["STREAMFLOW_INPUT_DIM"], config["EMBEDDING_DIM"])
        self.pos_encoder = PositionalEncoding(config["EMBEDDING_DIM"])
        self.dropout = nn.Dropout(config["DROPOUT"])
        self.embedding_dim = config["EMBEDDING_DIM"]

    def forward(self, x):
        x = self.linear_projection(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        return self.dropout(x)

# ======================================================================
# 3. U-NET STYLE DECODER HEAD (With Skip Connections)
# ======================================================================
class UnetStyleRainfallHead_Lite(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_patches_h = config["IMAGE_SIZE"] // config["PATCH_SIZE"]
        self.embedding_dim = config["EMBEDDING_DIM"]
        
        # Reduced channel counts to match the encoder
        self.up_stage1 = nn.ConvTranspose2d(self.embedding_dim, 64, kernel_size=4, stride=4)
        self.refine1 = MBConv(inp=64 + 64, oup=64, expand_ratio=4)
        
        self.up_stage2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.refine2 = MBConv(inp=32 + 48, oup=32, expand_ratio=4) # Note: channel dim changes
        
        self.up_stage3 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.refine3 = MBConv(inp=24 + 32, oup=24, expand_ratio=4)
        
        self.final_conv = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = x.permute(0, 2, 1).view(x.shape[0], self.embedding_dim, self.num_patches_h, self.num_patches_h)
        x = self.up_stage1(x); x = self.refine1(torch.cat([x, skips[2]], dim=1))
        x = self.up_stage2(x); x = self.refine2(torch.cat([x, skips[1]], dim=1))
        x = self.up_stage3(x); x = self.refine3(torch.cat([x, skips[0]], dim=1))
        return self.final_conv(x)

class SharedTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["EMBEDDING_DIM"], nhead=config["N_HEADS"],
            dim_feedforward=config["EMBEDDING_DIM"] * 4, dropout=config["DROPOUT"], batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["N_LAYERS"])

    def forward(self, x):
        return self.transformer_encoder(x)

class StreamflowHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.Linear(config["EMBEDDING_DIM"], config["STREAMFLOW_OUTPUT_DIM"])

    def forward(self, x):
        x = self.head(x)
        return x[:, -1, :]



# ======================================================================
# 4. FINAL ASSEMBLED MODEL V2
# ======================================================================

class MultiModalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.streamflow_embedder = StreamflowEmbedder(config) 
        self.streamflow_head = StreamflowHead(config)

        # Use the new LITE rainfall components
        self.rainfall_embedder = RainfallCNNEmbedder_Lite(config)
        self.rainfall_head = UnetStyleRainfallHead_Lite(config)
        
        self.shared_encoder = SharedTransformerEncoder(config)
        self.final_activation = nn.LeakyReLU()

    def forward(self, input_data):
        # Forward pass logic is identical to the v2 model
        if input_data.ndim == 3 and input_data.shape[2] == self.config["STREAMFLOW_INPUT_DIM"]:
            initial_embeddings = self.streamflow_embedder(input_data)
            processed_embeddings = self.shared_encoder(initial_embeddings)
            final_embeddings = initial_embeddings + processed_embeddings
            output = self.streamflow_head(final_embeddings)
        elif input_data.ndim == 4 and input_data.shape[1] == self.config["RAINFALL_CHANNELS"]:
            initial_tokens, skips = self.rainfall_embedder(input_data)
            processed_tokens = self.shared_encoder(initial_tokens)
            final_tokens = initial_tokens + processed_tokens
            refined_motion = self.rainfall_head(final_tokens, skips)
            t_0 = input_data[:, 0, :, :].unsqueeze(1)
            output = t_0 + refined_motion
            output = self.final_activation(output)
        else:
            raise ValueError(f"Unsupported input shape: {input_data.shape}")
        return output