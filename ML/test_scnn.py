"""
Tests for Model C (scnn.py). Run from ML/:  python test_scnn.py   (or pytest test_scnn.py)

1. Causality: changing the input at step t0 changes nothing (any layer) before t0.
2. Tap mapping: one input spike at t0 gives layer-1 current W1[2k + p, f] at exactly t0 + k * d_f.
3. Integer reference vs QAT model: every layer's spikes match bit for bit.
4. Export round-trip: weights written to .mem, read back by the standalone reference, give the
   same output counts as the PyTorch model.
Each test runs on the default config and on the ablation variants (fc readout, unshared filters,
K = 1, three dilation groups), so a --set override can't silently break the hardware math.
"""

import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import export
import scnn
from common import N_IN, T

torch.backends.cuda.matmul.allow_tf32 = False

VARIANTS = {
    "default": {},
    "fc readout": {"readout": "fc"},
    "unshared": {"share_filters": False},
    "K = 1": {"kernel_len": 1},
    "d = 1, 2, 4": {"dilations": [1, 2, 4], "group_filters": [4, 4, 4]},
}


def configs():
    for name, over in VARIANTS.items():
        yield name, {**copy.deepcopy(scnn.CONFIG), **over}


def eeg_like_spikes(rng, n, density=0.07):
    """[n, T, N_IN] spikes with at most one of UP / DOWN per channel per step, like the encoder."""
    on = rng.random((n, T, N_IN // 2)) < 2 * density
    up = rng.random((n, T, N_IN // 2)) < 0.5
    x = np.zeros((n, T, N_IN), dtype=np.uint8)
    x[:, :, 0::2], x[:, :, 1::2] = on & up, on & ~up
    return x


def net_for(cfg, seed=0):
    torch.manual_seed(seed)
    return scnn.SCNNNet(cfg).eval()


def test_causality():
    rng = np.random.default_rng(1)
    for name, cfg in configs():
        net, t0 = net_for(cfg), 50
        x = torch.from_numpy(eeg_like_spikes(rng, 4)).float()
        y = x.clone()
        y[:, t0:] = torch.from_numpy(eeg_like_spikes(rng, 4)[:, t0:]).float()
        for qat in (False, True):
            with torch.no_grad():
                a, b = net.layers(x, qat), net.layers(y, qat)
            for la, lb in zip(a, b):
                if la is not None:
                    assert torch.equal(la[:, :t0], lb[:, :t0]), f"{name}: output before t0 changed"
            assert not torch.equal(a[0], b[0]), f"{name}: the perturbation had no effect at all"
        # integer reference too
        w = scnn.random_weights(rng, cfg)
        ra, rb = scnn.run_int_reference(x[0].numpy().astype(np.uint8), w, cfg), \
            scnn.run_int_reference(y[0].numpy().astype(np.uint8), w, cfg)
        for key in ra:
            if key not in ("counts", "decision"):
                assert np.array_equal(ra[key][:t0], rb[key][:t0]), f"{name}: int {key} changed before t0"
    print("causality: ok")


def test_tap_mapping():
    for name, cfg in configs():
        K, n_ch = cfg["kernel_len"], cfg["n_channels"]
        shape = scnn.weight_shapes(cfg)["l1"]
        w1 = torch.arange(1, np.prod(shape) + 1, dtype=torch.float32).view(shape)   # every weight distinct
        for ch, p, t0 in ((0, 0, 10), (7, 1, 3), (n_ch - 1, 0, 100)):
            x = torch.zeros(1, T, N_IN)
            x[0, t0, 2 * ch + p] = 1
            cur = scnn.layer1_currents(x, w1, cfg)[0]          # [T, n_ch, C]
            want = torch.zeros_like(cur)
            row0 = 0 if cfg["share_filters"] else ch * 2 * K
            for d, a, b in scnn.filter_groups(cfg):
                for k in range(K):
                    if t0 + k * d < T:
                        want[t0 + k * d, ch, a:b] = w1[row0 + 2 * k + p, a:b]
            assert torch.equal(cur, want), f"{name}: tap mapping wrong for ch {ch}, p {p}, t0 {t0}"
            # the integer reference's tap_inputs agrees with the torch tap_tensor
            xs = x[0].numpy().astype(np.uint8)
            for d, _, _ in scnn.filter_groups(cfg):
                h = scnn.tap_tensor(x, d, K)[0]
                for t in (t0, min(T - 1, t0 + d), T - 1):
                    assert np.array_equal(h[t, ch].numpy(), scnn.tap_inputs(xs, t, d, K)[ch])
    print("tap mapping: ok")


def int_vs_qat(cfg, x, net):
    """Return the number of windows whose layer spikes all match, and the QAT counts."""
    with torch.no_grad():
        s1, s2, out = net.layers(torch.from_numpy(x).float(), qat=True)
    weights = net.quantize()
    ok = 0
    for i in range(len(x)):
        r = scnn.run_int_reference(x[i], weights, cfg)
        same = np.array_equal(r["spk_l1"], s1[i].flatten(1).numpy()) and np.array_equal(r["out"], out[i].numpy())
        if s2 is not None:
            same = same and np.array_equal(r["spk_l2"], s2[i].numpy())
        ok += same
    return ok, out.sum(dim=1).numpy()


def test_int_reference_matches_qat():
    rng = np.random.default_rng(2)
    for name, cfg in configs():
        for density in (0.03, 0.07, 0.15):
            x = eeg_like_spikes(rng, 8, density)
            ok, _ = int_vs_qat(cfg, x, net_for(cfg))
            assert ok == len(x), f"{name}, density {density}: {len(x) - ok}/{len(x)} windows differ"
    print("integer reference == QAT forward (every layer, every window): ok")


def test_export_round_trip():
    rng = np.random.default_rng(3)
    for name, cfg in configs():
        net = net_for(cfg)
        weights = net.quantize()
        # quantize -> load_int_weights -> quantize is the identity
        net2 = scnn.SCNNNet(cfg)
        net2.load_int_weights(weights)
        assert all(np.array_equal(weights[k], v) for k, v in net2.quantize().items())

        x = eeg_like_spikes(rng, 1)
        with tempfile.TemporaryDirectory() as tmp:
            old = scnn.CONFIG.copy()
            scnn.CONFIG.update(cfg)      # export / the reference copy read the module CONFIG
            try:
                export.export(scnn, weights, x[0], tmp)
                export.write_reference_copy(tmp, scnn, "scnn")
            finally:
                scnn.CONFIG.clear()
                scnn.CONFIG.update(old)
            spec = importlib.util.spec_from_file_location("reference_scnn_rt", Path(tmp) / "reference_scnn.py")
            ref = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ref)
            w_back, s_back = ref.load_case(tmp)
            assert ref.check_case(tmp)
        assert all(np.array_equal(weights[k], w_back[k]) for k in weights)
        r = ref.run_int_reference(s_back, w_back)
        with torch.no_grad():
            counts, _ = net(torch.from_numpy(x).float(), qat=True)
        assert np.array_equal(r["counts"], counts[0].numpy().astype(int)), f"{name}: round-trip counts differ"
    print("export round-trip: ok")


if __name__ == "__main__":
    test_causality()
    test_tap_mapping()
    test_int_reference_matches_qat()
    test_export_round_trip()
    print("ALL SCNN TESTS PASS")
    sys.exit(0)
