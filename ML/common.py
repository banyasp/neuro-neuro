"""
Spec shared by every model: input channels, encoding, window size, integer types.

All models in the ensemble see the SAME input spikes, so nothing here should be
redefined in a model file. Changing a value here changes every model and the
hardware input path, so tell the hardware team first.
"""

import numpy as np

# The 23 CHB-MIT bipolar channels, in EDF file order. MNE renames the duplicate
# T8-P8 label to T8-P8-0 / T8-P8-1, so those exact names are listed here.
# data/check_channels.py decides whether T8-P8-1 and P7-T7 are redundant; if so,
# delete them here (and re-run preprocess.py + export.py). Nothing else hard-codes the count.
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1",
]
N_IN = 2 * len(CHANNELS)  # line 2*ch = UP spike of channel ch, 2*ch+1 = DOWN spike

FS = 128        # Hz, after decimating CHB-MIT's 256 Hz by 2
WINDOW = 128    # samples per window = 1 s at 128 Hz
STEP = 64       # new window every 64 samples = 0.5 s (50% overlap)
T = WINDOW      # timesteps per window (one per sample)

COMMON_CONFIG = {
    # Dataset (CHB-MIT, decimated)
    "fs": FS,
    "n_channels": len(CHANNELS),
    # Input encoding. Each channel is divided by a fixed per-channel scale
    # (std of that patient's non-seizure training data), then delta-modulated
    # with threshold alpha, i.e. d_c = alpha * scale_c. Invisible to hardware.
    "delta_alpha": 0.3,
    "window_len": WINDOW,     # samples per window = timesteps per window
    "window_step": STEP,
    "n_inputs": N_IN,
    # Integer types
    "weight_bits": 8,         # signed int8 weights
    "mem_bits": 16,           # signed int16 membrane, saturating
    "seed": 0,
}

W_MIN = -(2 ** (COMMON_CONFIG["weight_bits"] - 1))      # -128
W_MAX = 2 ** (COMMON_CONFIG["weight_bits"] - 1) - 1     # 127
V_MIN = -(2 ** (COMMON_CONFIG["mem_bits"] - 1))         # -32768
V_MAX = 2 ** (COMMON_CONFIG["mem_bits"] - 1) - 1        # 32767


def random_spikes(rng, density):
    """Synthetic input window: [T, N_IN] of 0/1, each line fires with p = density."""
    return (rng.random((T, N_IN)) < density).astype(np.uint8)


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
