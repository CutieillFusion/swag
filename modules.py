import torch
from torch import nn
from torch.nn import functional as F
import math

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        residual = out
        out = self.conv2(out)
        out = self.relu(out)
        out = out + residual
        return out


class Stack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Stack, self).__init__()
        self.resnet_block1 = ResNetBlock(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.resnet_block2 = ResNetBlock(
            out_channels, out_channels, kernel_size, stride, padding
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.pool(x)
        x = self.conv(x)
        return x


class L2Norm(nn.Module):
    def __init__(self, eps=1e-12):
        super(L2Norm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1, eps=self.eps)


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output, _ = self.attn(q, k, v)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mlp_output = self.mlp(x)
        return self.dropout(mlp_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


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

    def forward(self, h):
        h_A = self.norm(self.attn_block(h))
        h = self.norm(h + self.alpha_A * (h_A - h))

        h_M = self.norm(self.mlp_block(h))
        h = self.norm(h + self.alpha_M * (h_M - h))

        return h
    
class MultiheadLatentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_dim = latent_dim

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_c = nn.Linear(embed_dim, latent_dim, bias=False)
        self.W_k = nn.Linear(latent_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(latent_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.size()

        Q_full = self.W_q(x)
        C = self.W_c(x)
        K_full = self.W_k(C)
        V_full = self.W_v(C)

        Q = Q_full.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K_full.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V_full.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
            else:
                attn_scores = attn_scores + attn_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        if self.dropout is not None:
            attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(attn_output)
        return output