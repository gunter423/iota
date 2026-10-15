from unittest.mock import AsyncMock

import pytest

from common.models.api_models import (
    MinerAttestationPayload,
    MinerRegistrationQueueStatusResponse,
    MinerRegistrationResponse,
    MountedAttestationPayload,
    RegisterMinerRequest,
    WeightUpdate,
)
from common.models.run_flags import RunFlags
from common.utils.partitions import MinerPartition
from subnet import miner_api_client as miner_api_client_module
from subnet.common_api_client import CommonAPIClient
from subnet.miner_api_client import MinerAPIClient, registration_queue_poll_sleep_seconds


def test_registration_queue_poll_sleep_adds_bounded_jitter(monkeypatch):
    calls = []

    def fake_uniform(min_sleep: float, max_sleep: float) -> float:
        calls.append((min_sleep, max_sleep))
        return 5.5

    monkeypatch.setattr(miner_api_client_module.random, "uniform", fake_uniform)

    assert registration_queue_poll_sleep_seconds(5.0) == 5.5
    assert calls == [(4.0, 6.0)]


def test_registration_queue_poll_sleep_can_disable_jitter(monkeypatch):
    monkeypatch.setattr(
        miner_api_client_module.random,
        "uniform",
        lambda *_args: pytest.fail("random.uniform should not be called when jitter is disabled"),
    )

    assert registration_queue_poll_sleep_seconds(5.0, jitter_fraction=0.0) == 5.0
    assert registration_queue_poll_sleep_seconds(0.0) == 0.0


@pytest.mark.asyncio
async def test_wait_for_registration_queue_uses_jittered_poll_sleep(monkeypatch):
    sleeps = []
    status_requests = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def fake_status_request(queue_id: str) -> MinerRegistrationResponse:
        status_requests.append(queue_id)
        return MinerRegistrationResponse(run_id="run-1", run_flags=RunFlags(), num_partitions=2)

    monkeypatch.setattr(miner_api_client_module.random, "uniform", lambda _min_sleep, _max_sleep: 4.25)
    monkeypatch.setattr(miner_api_client_module.asyncio, "sleep", fake_sleep)

    client = MinerAPIClient()
    monkeypatch.setattr(client, "_registration_queue_status_request", fake_status_request)

    response = await client._wait_for_registration_queue(
        MinerRegistrationQueueStatusResponse(
            queue_id="queue-1",
            status="queued",
            poll_after_seconds=5.0,
        )
    )

    assert response.run_id == "run-1"
    assert sleeps == [4.25]
    assert status_requests == ["queue-1"]


@pytest.mark.asyncio
async def test_register_request_does_not_replay_single_use_attestation(monkeypatch):
    request = AsyncMock(
        return_value={
            "run_id": "run-1",
            "run_flags": {},
            "num_partitions": 2,
        }
    )
    monkeypatch.setattr(CommonAPIClient, "orchestrator_request", request)

    response = await MinerAPIClient().register_miner_request(RegisterMinerRequest(p2p_node_id="node-1"))

    assert response.run_id == "run-1"
    assert request.await_args.kwargs["path"] == "/miner/register"
    assert request.await_args.kwargs["max_attempts"] == 1


@pytest.mark.asyncio
async def test_submit_weights_sends_attestation_in_body(monkeypatch):
    request = AsyncMock(return_value={"message": "ok", "should_upload_optimizer_state": False})
    monkeypatch.setattr(CommonAPIClient, "orchestrator_request", request)
    attestation = MinerAttestationPayload(payload_blob="blob")

    response = await MinerAPIClient().submit_weights(
        weight_update=WeightUpdate(weights_path="s3://weights/blob", attestation=attestation)
    )

    assert response.should_upload_optimizer_state is False
    assert request.await_args.kwargs["path"] == "/miner/submit_weights"
    body = request.await_args.kwargs["body"]
    assert body["weights_path"] == "s3://weights/blob"
    assert body["attestation"] == {"payload_blob": "blob", "encrypted_attestation": None}


@pytest.mark.asyncio
async def test_submit_merged_partitions_sends_attestation_in_body(monkeypatch):
    request = AsyncMock(return_value={"message": "ok"})
    monkeypatch.setattr(CommonAPIClient, "orchestrator_request", request)
    attestation = MountedAttestationPayload(
        schema_version=1,
        key_id="key-1",
        public_key_base64="pub",
        payload_base64="payload",
        signature_der_base64="sig",
        payload_sha256_base64="sha",
        alg="ES256",
        challenge_id="challenge-1",
    )

    await MinerAPIClient().submit_merged_partitions(
        merged_partitions=[MinerPartition(layer=0, chunk_number=3, miner_hotkey="hk", blob_path="s3://p/3")],
        attestation=attestation,
    )

    assert request.await_args.kwargs["path"] == "/miner/submit_merged_partitions"
    body = request.await_args.kwargs["body"]
    assert [p["chunk_number"] for p in body["partitions"]] == [3]
    assert body["attestation"]["challenge_id"] == "challenge-1"
    assert body["attestation"]["signature_der_base64"] == "sig"


@pytest.mark.asyncio
async def test_upload_submits_without_attestation_keep_optional_field(monkeypatch):
    # Runs with attest off (or an attestation the client could not collect) still
    # submit; the field is sent as null, which the server accepts.
    request = AsyncMock(return_value={"message": "ok"})
    monkeypatch.setattr(CommonAPIClient, "orchestrator_request", request)
    client = MinerAPIClient()

    await client.submit_weights(weight_update=WeightUpdate(weights_path="s3://weights/blob"))
    assert request.await_args.kwargs["body"]["attestation"] is None

    await client.submit_merged_partitions(merged_partitions=[MinerPartition(layer=0, chunk_number=0)])
    assert request.await_args.kwargs["body"]["attestation"] is None
