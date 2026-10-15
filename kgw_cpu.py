"""
Make MarkLLM's KGW green lists reproducible on any machine.

MarkLLM draws each green list with torch.randperm on the model's device. On CUDA the permutation for the
same seed differs by GPU model (P100 / V100 / A40 all disagree), so text watermarked on one GPU is not
detected on another. torch.randperm on CPU gives the same permutation across machines and torch versions
(checked 2.5.1, 2.9.1, 2.14.1), so we draw green lists on CPU and move them to the device.

Use for both generation and detection, right after loading the watermark:
    kgw = AutoWatermark.load("KGW", ...)
    patch_kgw_cpu(kgw)
"""
from collections import OrderedDict

import torch


def patch_kgw_cpu(kgw, cache_size=4096):
    utils, cfg = kgw.utils, kgw.utils.config
    assert cfg.prefix_length == 1 and cfg.f_scheme == "time" and cfg.window_scheme == "left", \
        "patch only implements KGW's default seeding (prefix_length=1, f_scheme=time, window_scheme=left)"

    V = cfg.vocab_size
    n_green = int(V * cfg.gamma)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(cfg.hash_key)
    prf = torch.randperm(V, generator=rng)
    utils.prf, utils.rng = prf, rng
    cache = OrderedDict()  # previous token -> green list (CPU); the green list depends only on it

    def get_greenlist_ids(input_ids):
        prev = int(input_ids[-1])
        if prev in cache:
            cache.move_to_end(prev)
            green = cache[prev]
        else:
            # Same seeding as MarkLLM's _f_time + _get_greenlist_ids_left, only on a CPU generator.
            rng.manual_seed((cfg.hash_key * int(prf[prev % V])) % V)
            green = torch.randperm(V, generator=rng)[:n_green]
            cache[prev] = green
            if len(cache) > cache_size:
                cache.popitem(last=False)
        return green.to(input_ids.device)

    utils.get_greenlist_ids = get_greenlist_ids  # used by both the logits processor and score_sequence
    return kgw
