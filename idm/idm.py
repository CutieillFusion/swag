import time
import torch
from torch import nn
from modules import Stack, PositionalEncoding, NormalizedTransformerBlock

class IDM(nn.Module):
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
            nn.BatchNorm3d(input_dim[0]),
            nn.Conv3d(
                input_dim[0],
                spatial_channels,
                kernel_size=(1, 1, 5),
                stride=(1, 1, 1),
                padding=(0, 0, 2),
            ),
            nn.SiLU(),
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
            nn.SiLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        )

        self.normalized_transformer = nn.Sequential(
            *[
                NormalizedTransformerBlock(embedding_dim, transformer_heads, ff_dim)
                for _ in range(transformer_blocks)
            ]
        )

        self.classifier = nn.Linear(embedding_dim, n_actions)

        # self.pos_encoder = PositionalEncoding(embedding_dim)

        if freeze:
            self._freeze()

    def _3d_convo(self, x: torch.Tensor):
        # [B, T, C, W, H] -> [B, T, W, H, C]
        x = x.permute(0, 1, 3, 4, 2)

        x = self.spatial_feature_extractor(x)
        
        # [B, T, W, H, C] -> [B, T, C, W, H]
        x = x.permute(0, 1, 4, 2, 3)
        return x

    def _2d_convo(self, x: torch.Tensor):
        batch_size, sequence_length, C, H, W = x.shape

        x = x.view(-1, C, H, W)

        x = self.feature_extractor(x)

        _, C_out, H_out, W_out = x.shape

        x = x.view(
            batch_size, sequence_length, C_out, H_out, W_out
        )

        return x

    def _embedder(self, x: torch.Tensor):
        batch_size, sequence_length, C, H, W = x.shape

        x = x.reshape(-1, C, H, W)

        x = self.embedder(x)

        x = x.view(
            batch_size, sequence_length, -1
        )

        # x = self.pos_encoder(x)

        return x

    def forward(self, x):
        x = self._3d_convo(x)

        x = self._2d_convo(x)

        x = self._embedder(x)

        x = self.normalized_transformer(x)

        logits = self.classifier(x)

        return logits

    def print_model_parameters(self):
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

    def inference_time_breakdown(self, x):
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

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(
            path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")
