"""
DreamEEG model: a lightweight variance-coupled dual-stream CNN-Transformer.

Architecture (from the proposal):
  Shared front-end
    - multi-scale temporal conv (parallel kernels 32/64/96/128)
    - depthwise spatial conv (C x 1) + BN + ELU
  Variance-coupled dual-stream encoder
    - first-order  stream: mean pool  (P_mu)   -> gated Transformer x N
    - second-order stream: log-var pool (P_sigma^2 / band-power) -> gated Transformer x N
  Fusion head
    - concat both stream summaries -> linear -> softmax

Kept small (single-GPU, few params) on purpose. Same call signature as
build_eegnet so experiment.py can swap models with --model dreameeg.
"""
import torch
import torch.nn as nn


class MultiScaleTemporal(nn.Module):
    """Parallel temporal convolutions at several kernel lengths, concatenated."""
    def __init__(self, f_per=8, kernels=(32, 64, 96, 128)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, f_per, (1, k), padding="same", bias=False),
                nn.BatchNorm2d(f_per))
            for k in kernels])

    def forward(self, x):                       # x: (B, 1, C, T)
        return torch.cat([b(x) for b in self.branches], dim=1)   # (B, f_per*K, C, T)


class DepthwiseSpatial(nn.Module):
    """Depthwise spatial conv over all channels -> collapses the channel axis."""
    def __init__(self, f_in, chans, d=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(f_in, f_in * d, (chans, 1), groups=f_in, bias=False),
            nn.BatchNorm2d(f_in * d),
            nn.ELU())

    def forward(self, x):                       # (B, f_in, C, T)
        return self.net(x)                      # (B, f_in*d, 1, T)


class GatedTransformerStream(nn.Module):
    """Transformer encoder over pooled tokens, with a learned temporal gate.

    pool='mean' -> first-order (average activation) tokens
    pool='logvar' -> second-order (log band-power) tokens
    """
    def __init__(self, d_model, n_win, pool="mean", n_layers=4, n_heads=4, dropout=0.3):
        super().__init__()
        self.n_win = n_win
        self.pool = pool
        self.pos = nn.Parameter(torch.zeros(1, n_win, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        # temporal gate: weights each token before the final pool
        self.gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def _tokens(self, x):                        # x: (B, D, T)
        B, D, T = x.shape
        w = T // self.n_win
        x = x[:, :, :w * self.n_win].reshape(B, D, self.n_win, w)
        if self.pool == "mean":
            tok = x.mean(-1)                     # first order
        else:
            tok = torch.log(x.var(-1) + 1e-6)    # second order (band-power)
        return tok.transpose(1, 2)               # (B, n_win, D)

    def forward(self, x):                        # x: (B, D, T)
        t = self._tokens(x) + self.pos
        t = self.encoder(t)                      # (B, n_win, D)
        g = self.gate(t)                         # (B, n_win, 1)
        return (t * g).sum(1) / (g.sum(1) + 1e-6)   # gated temporal pooling -> (B, D)


class DreamEEG(nn.Module):
    def __init__(self, chans=32, classes=3, time_points=1000,
                 f_per=4, kernels=(32, 64, 96, 128), d=2,
                 n_win=20, n_layers=2, n_heads=4, dropout=0.5):
        super().__init__()
        f_ms = f_per * len(kernels)
        d_model = f_ms * d
        self.front_ms = MultiScaleTemporal(f_per, kernels)
        self.front_sp = DepthwiseSpatial(f_ms, chans, d)
        self.drop = nn.Dropout(dropout)
        self.stream_mean = GatedTransformerStream(d_model, n_win, "mean", n_layers, n_heads)
        self.stream_var = GatedTransformerStream(d_model, n_win, "logvar", n_layers, n_heads)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(d_model, classes))

    def forward(self, x):                        # x: (B, 1, C, T)
        h = self.front_sp(self.front_ms(x))      # (B, d_model, 1, T)
        h = self.drop(h).squeeze(2)              # (B, d_model, T)
        m = self.stream_mean(h)                  # (B, d_model)
        v = self.stream_var(h)                   # (B, d_model)
        return self.head(torch.cat([m, v], dim=1))


def build_dreameeg(chans=32, classes=3, time_points=1000):
    return DreamEEG(chans=chans, classes=classes, time_points=time_points)


if __name__ == "__main__":
    net = build_dreameeg()
    n = sum(p.numel() for p in net.parameters())
    x = torch.randn(4, 1, 32, 1000)
    print("output:", net(x).shape, "| params:", f"{n:,}")
