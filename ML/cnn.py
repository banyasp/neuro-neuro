"""
Baseline: small 1D CNN for EEG seizure detection. Software only, never exported to hardware.

Same inputs and outputs as the SNNs (lif.py, alif.py), so train.py scores all of them the
same way:
  input:  [batch, T, N_IN] of 0/1, the same delta-modulated spike windows from data/dataset.py
  output: logits [batch, n_outputs] in place of spike counts. Neuron 0 = non-seizure, 1 = seizure.
Decision: seizure if logit 1 > logit 0 (tie = no seizure), i.e. the same rule as count1 > count0.

It answers "how much does the spiking / int8 / hardware constraint cost?": a conventional
float network trained on exactly the same input spikes, with no integer reference and no QAT.

Architecture (time is the conv axis, input lines are channels):
  3 x [Conv1d -> BatchNorm -> ReLU (-> MaxPool /2 after the first two)]  N_IN -> 32 -> 64 -> 64
  global average pool over time -> dropout -> Linear(64, n_outputs)
"""

import torch
import torch.nn as nn

from common import COMMON_CONFIG, N_IN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    **COMMON_CONFIG,
    "n_outputs": 2,           # neuron 0 = non-seizure, neuron 1 = seizure
    "channels": (32, 64, 64),
    "kernels": (7, 5, 3),     # in timesteps; 7 samples ~ 55 ms at 128 Hz
    "dropout": 0.3,
}

# ---------------------------------------------------------------------------
# Model (PyTorch)
# ---------------------------------------------------------------------------
# train.py treats a module without run_int_reference as software-only: float training only
# (no QAT stage, no int8 weights, no agreement check, no handoff), and the saved weights are a
# PyTorch state_dict (weights.pt). --rate-penalty has no effect (there are no spikes to penalize).


class CNNNet(nn.Module):
    """1D CNN baseline. Input x: [batch, T, N_IN] of 0/1."""

    def __init__(self, cfg=CONFIG):
        super().__init__()
        self.cfg = cfg
        layers, c_in = [], N_IN
        for i, (c_out, k) in enumerate(zip(cfg["channels"], cfg["kernels"])):
            layers += [nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False),
                       nn.BatchNorm1d(c_out), nn.ReLU()]
            if i < len(cfg["channels"]) - 1:
                layers.append(nn.MaxPool1d(2))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(c_in, cfg["n_outputs"]))

    def forward(self, x, qat=False):
        """Returns (logits [B, n_outputs], fraction of active last-layer ReLUs).

        Same signature as the SNNs' forward so train.py can drive it; qat is ignored. The second
        value is only reported (as "hidden"), it carries no gradient."""
        h = self.features(x.transpose(1, 2))   # [B, N_IN, T] -> [B, C, T / 4]
        logits = self.head(h.mean(dim=2))      # global average pool over time
        return logits, (h > 0).float().mean().detach()


Net = CNNNet   # train.py looks for <model>.Net
