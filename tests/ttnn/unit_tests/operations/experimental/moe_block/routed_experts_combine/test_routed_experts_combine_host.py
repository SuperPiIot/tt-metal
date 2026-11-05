# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""routed_experts_combine host math (C++, bound in ttnn.experimental.moe_block), no device: the credit policy and what
combine_auto picks.

    pytest tests/ttnn/unit_tests/operations/experimental/moe_block/routed_experts_combine/test_routed_experts_combine_host.py
"""

import collections
import itertools

import ttnn

mb = ttnn.experimental.moe_block

# (N, S, W): the matrix models' exchange shapes (GLM-5.3, K3 latent, DSv4.1 H) on LoudBox / Galaxy meshes + edges
SHAPES = [
    (8, 640, 4096),
    (32, 160, 4096),
    (32, 160, 3584),
    (8, 2560, 3584),
    (32, 640, 3584),
    (4, 1280, 2048),
    (16, 320, 5120),
    (2, 64, 1024),
]
ARENAS = [64 << 10, 400 << 10, 900 << 10, 1200 << 10, 1400 << 10]


def test_credit_policy():
    """With >= 16 sources (the Galaxy) credits go back every 8 consumed rows, idle snapshots batched (8 rows or 8100
    cycles old); with fewer, a batch is at most half a source's slots (D), the shared landing's grants at once; a
    config's credit_batch wins."""
    assert mb.combine_credit_policy() == (8, 8, 8100)
    assert mb.combine_credit_policy(N=32, D=2) == (8, 8, 8100)
    assert mb.combine_credit_policy(N=16, D=2) == (8, 8, 8100)  # (a Galaxy 4x4 sub-torus: 15 sources)
    assert mb.combine_credit_policy(N=16, D=2, shared=True) == (1, 1, 8100)
    assert mb.combine_credit_policy(N=8, D=8) == (4, 4, 8100)
    assert mb.combine_credit_policy(N=8, D=2) == (1, 1, 8100)
    assert mb.combine_credit_policy(N=8, D=24) == (8, 8, 8100)
    assert mb.combine_credit_policy(N=8, D=8, shared=True) == (1, 1, 8100)
    assert mb.combine_credit_policy(N=32, D=2, batch=1) == (1, 1, 8100)


def test_auto_invariants():
    """Self-contained (no reference): what combine_auto picks fits the arena and holds a slot per source chip."""
    for (N, S, W), arena in itertools.product(SHAPES, ARENAS):
        for cells, ports in ((19, 8), (30, 8)):
            got = mb.combine_auto(N, S, W, cells, ports, arena_bytes=arena)
            if not got["fits"]:
                assert got["why"]
                continue
            lay = got["layout"]
            assert lay["fits"] and lay["PSL"] >= N - 1 and lay["PSL"] < 4096
            assert max(lay["acc_used"], lay["port_used"]) <= arena
            assert 1 <= got["n_acc"] <= 32  # (kAccMax)
    counts = collections.Counter(len(mb.combine_state_layout(N, A)) for N in (8, 32) for A in (1, 16))
    assert counts == {10: 4}


def test_auto_shared_landing():
    """The shared landing pool: accumulators apart from the ports only, the pool under 255 slots (slot owners are
    bytes), the grant rings' state no larger than rings' by more than the grant words and snapshots."""
    for (N, S, W), arena in itertools.product(SHAPES, ARENAS):
        for cells, ports in ((19, 8), (30, 8)):
            got = mb.combine_auto(N, S, W, cells, ports, arena_bytes=arena, landing=mb.CombineLanding.Shared)
            if not got["fits"]:
                assert got["why"]
                continue
            lay = got["layout"]
            assert got["shared"] and not got["acc_on_ports"] and "acc_on_ports" not in got["modes"]
            assert lay["fits"] and N - 1 <= lay["PSL"] <= 254
            assert max(lay["acc_used"], lay["port_used"]) <= arena
