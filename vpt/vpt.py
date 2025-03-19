import time
import torch
from torch import nn
from modules import L2Norm, Stack, PositionalEncoding, NormalizedTransformerBlock
import torch.nn.functional as F
import math


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x, x.shape[-1:])

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Generate causal mask manually
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float("-inf"), diagonal=1
        )

        attn_output, _ = self.attn(q, k, v, attn_mask=causal_mask)

        return attn_output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_output = self.mlp(x)
        return self.dropout(mlp_output)


class NormalizedTransformerBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, ff_dim, alpha_A=0.1, alpha_M=0.1, dropout=0.1
    ):
        super(NormalizedTransformerBlock, self).__init__()
        self.attn_block = MultiheadAttention(embed_dim, num_heads)
        self.mlp_block = FeedForward(embed_dim, ff_dim, dropout=dropout)

        self.alpha_A = nn.Parameter(torch.ones(embed_dim) * alpha_A)
        self.alpha_M = nn.Parameter(torch.ones(embed_dim) * alpha_M)

        self.norm = L2Norm()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_A = self.norm(self.attn_block(h))
        h = self.norm(h + self.alpha_A * (h_A - h))

        h_M = self.norm(self.mlp_block(h))
        h = self.norm(h + self.alpha_M * (h_M - h))

        return h


class VPT(nn.Module):
    def __init__(
        self,
        input_dim=(64, 3, 240, 256),
        spatial_channels=64,
        feature_channels=[16, 32, 32],
        embedding_dim=256,
        ff_dim=256,
        transformer_heads=4,
        transformer_blocks=1,
        n_actions=36,
        freeze=False,
    ):
        super().__init__()

        self.spatial_feature_extractor = nn.Sequential(
            nn.Conv3d(
                input_dim[0],
                spatial_channels,
                kernel_size=(1, 1, 5),
                stride=(1, 1, 1),
                padding=(0, 0, 2),
            ),
            nn.BatchNorm3d(spatial_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        feature_extractor_layers = []
        in_channels = input_dim[1]
        for channels in feature_channels:
            feature_extractor_layers.extend([Stack(in_channels, channels)])
            in_channels = channels
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        in_features = (
            feature_channels[-1]
            * (input_dim[2] // 2 ** len(feature_channels))
            * (input_dim[3] // 2 ** len(feature_channels))
        )
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        )

        self.normalized_transformer = nn.Sequential(
            *[
                NormalizedTransformerBlock(embedding_dim, transformer_heads, ff_dim)
                for _ in range(transformer_blocks)
            ]
        )

        self.classifier = nn.Linear(embedding_dim, n_actions)

        self.pos_encoder = PositionalEncoding(embedding_dim)

        if freeze:
            self._freeze()

    def _3d_convo(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 1, 3, 4, 2)

        x = self.spatial_feature_extractor(x)

        x = x.permute(0, 1, 4, 2, 3)
        return x

    def _2d_convo(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, C, H, W = x.shape

        x = x.view(-1, C, H, W)

        x = self.feature_extractor(x)

        _, C_out, H_out, W_out = x.shape

        x = x.view(batch_size, sequence_length, C_out, H_out, W_out)

        return x

    def _embedder(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, C, H, W = x.shape

        x = x.reshape(-1, C, H, W)

        x = self.embedder(x)

        x = x.view(batch_size, sequence_length, -1)

        x = self.pos_encoder(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._3d_convo(x)

        x = self._2d_convo(x)

        x = self._embedder(x)

        x = self.normalized_transformer(x)

        logits = self.classifier(x)

        return logits

    def print_model_parameters(self) -> None:
        total_params = 0
        print("Model Parameters:")
        for name, param in self.named_parameters():
            print(
                f"{name}: {param.numel()} | {param.shape} | Trainable: {param.requires_grad}"
            )
            total_params += param.numel()
        print(f"\nTotal Parameters: {total_params:,}")

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def inference_time_breakdown(self, x: torch.Tensor) -> None:
        start = time.time()
        x = self._3d_convo(x)
        step_3d_convo_time = time.time() - start
        print(f"3D Convolution took {step_3d_convo_time:.6f} seconds")

        start = time.time()
        x = self._2d_convo(x)
        step_2d_convo_time = time.time() - start
        print(f"2D Convolution took {step_2d_convo_time:.6f} seconds")

        start = time.time()
        x = self._embedder(x)
        step_embedder_time = time.time() - start
        print(f"Embedding took {step_embedder_time:.6f} seconds")

        start = time.time()
        x = self.normalized_transformer(x)
        step_transformer_time = time.time() - start
        print(f"Transformer took {step_transformer_time:.6f} seconds")

        start = time.time()
        logits = self.classifier(x)
        step_classifier_time = time.time() - start
        print(f"Classifier took {step_classifier_time:.6f} seconds")

        return logits

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(
            path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")
