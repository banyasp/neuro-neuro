"""
Spec shared by every model: input encoding, window size, integer types.

All models in the ensemble see the SAME input spikes, so nothing here should be
redefined in a model file. Changing a value here changes every model and the
hardware input path, so tell the hardware team first.
"""

import numpy as np

COMMON_CONFIG = {
    # Dataset (CHB-MIT)
    "fs": 256,                # sampling rate, Hz
    "n_channels": 23,         # bipolar EEG channels
    # Input encoding
    "delta_threshold": 0.3,   # delta modulation threshold on z-scored EEG
    "window_len": 128,        # samples per window (0.5 s) = timesteps per window
    "window_step": 64,        # 50% overlap -> one decision every 0.25 s
    "n_inputs": 46,           # line 2*ch = UP spike of channel ch, 2*ch+1 = DOWN spike
    # Integer types
    "weight_bits": 8,         # signed int8 weights
    "mem_bits": 16,           # signed int16 membrane, saturating
    "seed": 0,
}

W_MIN = -(2 ** (COMMON_CONFIG["weight_bits"] - 1))      # -128
W_MAX = 2 ** (COMMON_CONFIG["weight_bits"] - 1) - 1     # 127
V_MIN = -(2 ** (COMMON_CONFIG["mem_bits"] - 1))         # -32768
V_MAX = 2 ** (COMMON_CONFIG["mem_bits"] - 1) - 1        # 32767


def random_spikes(rng, density, cfg=COMMON_CONFIG):
    """Synthetic input window: [window_len, n_inputs] of 0/1, each line fires with p = density."""
    return (rng.random((cfg["window_len"], cfg["n_inputs"])) < density).astype(np.uint8)


def random_int8(rng, shape):
    """Full-range random int8 array, for test weights."""
    return rng.integers(W_MIN, W_MAX + 1, size=shape).astype(np.int8)


# ---------------------------------------------------------------------------
# Shared PyTorch pieces for training float SNNs
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:  # integer references work without PyTorch
    torch = None

if torch is not None:

    class FastSigmoidSpike(torch.autograd.Function):
        """Heaviside forward, fast-sigmoid surrogate gradient backward."""
        slope = 25.0

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x >= 0).float()

        @staticmethod
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return grad_out / (FastSigmoidSpike.slope * x.abs() + 1.0) ** 2

    spike_fn = FastSigmoidSpike.apply
