"""
Ablation 4a for Model C: the same architecture as scnn.py with no spikes. Software only.

Every LIF neuron is replaced by a ReLU, and layers pass real values (the spatial stage sees the
temporal filters' continuous outputs, as in EEGNet). The output layer's per-step values are
averaged over the window to give logits. Comparing three runs separates two costs (RQ1):

    scnn_float                 float, not spiking       what the architecture can do
    scnn --float-only          float, spiking           cost of spiking
    scnn                       int8, spiking (shipped)  cost of int8 on top

ML/cnn.py stays the separate, stronger conventional baseline.

Same CONFIG keys as scnn.py (kernel_len, dilations, group_filters, depth_mult, readout,
share_filters), so train.py --set works the same way; this file keeps its own copy of the dict.
Weights are float (no int8 grid), there's no QAT, and train.py saves weights.pt.
"""

import copy

import torch
import torch.nn as nn

import scnn

CONFIG = copy.deepcopy(scnn.CONFIG)   # same knobs; --set edits this copy, not scnn's


class SCNNFloatNet(nn.Module):
    """Input x: [batch, T, N_IN] of 0/1. Returns (logits [B, n_outputs], fraction of active layer-1 ReLUs)."""

    def __init__(self, cfg=CONFIG):
        super().__init__()
        self.cfg = cfg
        shapes = scnn.weight_shapes(cfg)
        # He init: fan_in = inputs per unit (2K taps for layer 1, 23 channels for depthwise layer 2)
        fan_in = {"l1": 2 * cfg["kernel_len"], "l2": shapes["l2"][0], "l3": shapes.get("l3", (1,))[0]}
        self.w = nn.ParameterDict({n: nn.Parameter(torch.randn(s) * (2 / fan_in[n]) ** 0.5)
                                   for n, s in shapes.items()})

    def forward(self, x, qat=False):
        a1 = torch.relu(scnn.layer1_currents(x, self.w["l1"], self.cfg))      # [B, T, n_ch, C]
        if self.cfg["readout"] == "fc":
            out = a1.flatten(2) @ self.w["l2"]
        else:
            D = self.cfg["depth_mult"]
            a2 = torch.relu((a1.repeat_interleave(D, dim=-1) * self.w["l2"]).sum(dim=2))   # [B, T, C*D]
            out = a2 @ self.w["l3"]
        return out.mean(dim=1), (a1 > 0).float().mean()


Net = SCNNFloatNet   # train.py looks for <model>.Net
