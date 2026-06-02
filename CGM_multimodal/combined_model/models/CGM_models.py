import torch
import torch.nn as nn


class CGMHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CGMHeadExpanded(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), #0.3
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.head(x)



class CGMHeadAttention(nn.Module):
    def __init__(self, in_dim, embed_dim=8, n_heads=1, out_dim=2,
                 n_macros=5, n_micro=0, dropout=0.2, hidden=64):
        super().__init__()
        self.n_macros   = n_macros
        self.n_micro    = n_micro
        n_clinical      = in_dim - n_macros - n_micro
        self.n_clinical = n_clinical
        self.in_dim     = in_dim
        self.embed_dim  = embed_dim

        # per-feature continuous embedding: each feature gets its own
        # embed_dim-dimensional weight vector and bias
        self.feature_weights = nn.Parameter(torch.randn(in_dim, embed_dim) * 0.02)
        self.feature_biases  = nn.Parameter(torch.zeros(in_dim, embed_dim))

        # attention over feature tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.flat_norm = nn.BatchNorm1d(in_dim * embed_dim)

        self.head = nn.Sequential(
            nn.Linear(in_dim * embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(hidden * 2, hidden),
            # nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, out_dim),
        )

    def forward(self, x):
        # x: (B, in_dim)

        # embed each feature: (B, in_dim, embed_dim)
        tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases

        # self-attention over feature tokens with residual
        attn_out, _ = self.attn(tokens, tokens, tokens)  # (B, in_dim, embed_dim)
        tokens = self.attn_norm(tokens + attn_out)        # residual connection

        # tokens = tokens + attn_out

        # flatten and predict
        out = tokens.flatten(1)                           # (B, in_dim * embed_dim)
        # out = self.flat_norm(out)
        return self.head(out)


class CGMHeadAttentionMicroFiLM(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim=8,
        n_heads=1,
        out_dim=2,
        n_macros=5,
        n_micro=0,
        dropout=0.2,
        hidden=128,
        micro_hidden=16,
    ):
        super().__init__()

        self.n_macros = n_macros
        self.n_micro = n_micro

        n_clinical = in_dim - n_macros - n_micro
        self.n_clinical = n_clinical

        # Attention only over clinical + macro features
        self.n_attn_features = n_clinical + n_macros
        self.embed_dim = embed_dim

        self.feature_weights = nn.Parameter(
            torch.randn(self.n_attn_features, embed_dim) * 0.02
        )
        self.feature_biases = nn.Parameter(
            torch.zeros(self.n_attn_features, embed_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(embed_dim)

        # Microbiome branch only creates modulation parameters
        if n_micro > 0:
            self.micro_film = nn.Sequential(
                nn.Linear(n_micro, micro_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(micro_hidden, 2 * embed_dim),
            )
        else:
            self.micro_film = None

        # Same head size whether microbiome exists or not
        self.head = nn.Sequential(
            nn.Linear(self.n_attn_features * embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # Expected input order:
        # [clinical features, macro features, microbiome features]

        x_attn = x[:, :self.n_attn_features]

        tokens = (
            x_attn.unsqueeze(-1) * self.feature_weights.unsqueeze(0)
            + self.feature_biases.unsqueeze(0)
        )

        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        # Fuse microbiome immediately after attention
        if self.n_micro > 0:
            x_micro = x[:, self.n_attn_features:]

            gamma_beta = self.micro_film(x_micro)
            gamma, beta = gamma_beta.chunk(2, dim=-1)

            tokens = tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        out = tokens.flatten(1)
        return self.head(out)
