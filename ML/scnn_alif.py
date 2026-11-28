"""
Model C with ALIF neurons: scnn.py's architecture (shared temporal filters, depthwise spatial
layer, 2 outputs) with Model B's adaptive neuron (alif.py) in the hidden layers. The output layer
stays plain LIF, as in alif.py. All the code is scnn.py's (its CONFIG key "neuron" = "alif");
this module only gives the variant its own name, so that

    python train.py --patient chb01 --model scnn_alif        # runs/chb01_scnn_alif_s0
    python export.py scnn_alif                               # handoff/scnn_alif/

work like any other model, and --set (e.g. leak_shift=4, adapt_inc=8) edits this CONFIG only.
"""

import scnn
from scnn import REFERENCE_FUNCS, filter_groups, history_depth   # noqa: F401 (part of the model interface)

CONFIG = {**scnn.CONFIG, "neuron": "alif"}


# The model interface (README.md), bound to this module's CONFIG. train.py's --set edits CONFIG
# in place, so every default below sees the change.

def weight_shapes(cfg=CONFIG):
    return scnn.weight_shapes(cfg)


def run_int_reference(spikes, weights, cfg=CONFIG):
    return scnn.run_int_reference(spikes, weights, cfg)


def hardware_cost(cfg=CONFIG):
    return scnn.hardware_cost(cfg)


def random_weights(rng, cfg=CONFIG):
    return scnn.random_weights(rng, cfg)


def test_cases(rng, cfg=CONFIG):
    return scnn.test_cases(rng, cfg)


def spec_md(cfg=CONFIG):
    return scnn.spec_md(cfg)


if scnn.torch is not None:

    class SCNNALIFNet(scnn.SCNNNet):
        """Model C with ALIF hidden neurons, for training (see scnn.SCNNNet)."""

        def __init__(self, cfg=CONFIG):
            super().__init__(cfg)

    Net = SCNNALIFNet   # train.py looks for <model>.Net
