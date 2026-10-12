"""Regression tests for capacity-weighted peer selection.

The 2026-07-22 orion stall: with every peer at exactly one free cache slot,
log(1)=0 weights summed to zero and random.choices raised
"Total of weights must be greater than zero", killing outbound sends.
"""

import time

from common.models.compute_node import ComputeNode
from common.models.peer_status import PeerStatusBroadcast

from miner.training.peer_selection import select_by_capacity


def _node(
    node_id: str,
    cache_size: int,
    cache_capacity: int = 30,
    forward_queue_size: int = 0,
    last_status_received: float | None = None,
) -> ComputeNode:
    return ComputeNode(
        node_id=node_id,
        runtime_metrics=PeerStatusBroadcast(
            cache_size=cache_size,
            cache_capacity=cache_capacity,
            forward_queue_size=forward_queue_size,
            last_status_received=time.time() if last_status_received is None else last_status_received,
        ),
    )


def test_all_peers_at_one_free_slot_does_not_raise():
    peers = [_node(f"n{i}", cache_size=29) for i in range(3)]
    assert select_by_capacity(peers) is not None


def test_prefers_peer_with_more_headroom():
    roomy, full = _node("roomy", cache_size=0), _node("full", cache_size=29)
    picks = {select_by_capacity([roomy, full]).node_id for _ in range(50)}
    assert "roomy" in picks


def test_all_saturated_applies_backpressure():
    saturated = [_node(f"n{i}", cache_size=30) for i in range(3)]
    # Peers report metrics but none has free cache -> don't send anywhere.
    assert select_by_capacity(saturated) is None


def test_unmeasured_peers_fall_back_to_random():
    unmeasured = [
        ComputeNode(node_id=f"n{i}", runtime_metrics=PeerStatusBroadcast(last_status_received=0.0)) for i in range(3)
    ]
    # First-contact window: no broadcasts yet -> uniform random, not deadlock.
    assert select_by_capacity(unmeasured) is not None


def test_stale_status_is_not_routed_to():
    """Received status used to never expire, so senders kept routing to dead/kicked peers."""
    now = 1_000_000.0
    fresh = _node("fresh", cache_size=29, last_status_received=now - 5)
    stale = _node("stale", cache_size=0, last_status_received=now - 120)
    picks = {select_by_capacity([fresh, stale], max_status_age_s=30, now=now).node_id for _ in range(50)}
    assert picks == {"fresh"}


def test_saturated_fresh_peer_still_applies_backpressure_despite_stale_roomy_peer():
    now = 1_000_000.0
    full = _node("full", cache_size=30, last_status_received=now - 1)
    stale = _node("stale", cache_size=0, last_status_received=now - 3600)
    assert select_by_capacity([full, stale], max_status_age_s=30, now=now) is None


def test_all_stale_falls_back_to_random_instead_of_deadlocking():
    now = 1_000_000.0
    peers = [_node(f"n{i}", cache_size=0, last_status_received=now - 3600) for i in range(3)]
    assert select_by_capacity(peers, max_status_age_s=30, now=now) is not None


def test_future_dated_status_is_not_trusted():
    """last_status_received is stamped with our own clock on receipt; a far-future value
    (e.g. self-advertised in a peer's registry entry) must not make a peer look fresh."""
    now = 1_000_000.0
    honest = _node("honest", cache_size=29, last_status_received=now - 2)
    liar = _node("liar", cache_size=0, last_status_received=now + 10_000)
    picks = {select_by_capacity([honest, liar], max_status_age_s=30, now=now).node_id for _ in range(50)}
    assert picks == {"honest"}


def test_default_max_age_comes_from_settings(monkeypatch):
    from miner import settings as miner_settings

    monkeypatch.setattr(miner_settings, "PEER_STATUS_MAX_AGE_SECONDS", 10.0)
    now = time.time()
    fresh = _node("fresh", cache_size=29, last_status_received=now - 1)
    old = _node("old", cache_size=0, last_status_received=now - 20)
    picks = {select_by_capacity([fresh, old]).node_id for _ in range(30)}
    assert picks == {"fresh"}
