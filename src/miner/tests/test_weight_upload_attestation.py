"""The miner attaches an attestation to /submit_weights and /submit_merged_partitions.

The server does not enforce attestation on those endpoints yet (it only records
`attestation_present` when a submit fails), so the client contract pinned here is
what a later server-side enforcement PR relies on:

- when the run's `attest` flag is on, a fresh challenge is requested with the
  endpoint's own action (`weights` / `merged_partitions`) and the collected payload
  is returned, via the same code path /submit_activation uses;
- when the flag is off, nothing is requested and `None` is sent (unchanged
  behaviour for runs without attestation);
- when attestation cannot be collected, the submit still goes out with
  `attestation=None` rather than failing the upload;
- `Miner.submit_weights` and the merge batch in `Miner.merge_partitions` pass the
  collected payload through to the API client call.

The attestation is collected against a bare per-action nonce before the upload, so
it is not bound to the uploaded weights; see the PR for the follow-up.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from common.models.api_models import (
    AttestationChallengeResponse,
    EnclaveSignResponse,
    MinerAttestationPayload,
    MountedAttestationPayload,
    RequestAttestationChallengeResponse,
    WeightSubmitResponse,
    WeightUpdate,
)
from common.models.run_flags import RunFlag, RunFlags
from miner import settings as miner_settings
from miner.new_miner import Miner
from miner.utils.attestation_utils import AttestationUnavailableError

CHALLENGE_ID = "challenge-weights-1"


def _challenge(action: str) -> RequestAttestationChallengeResponse:
    return RequestAttestationChallengeResponse(
        action=action,
        attestation_challenge_blob=json.dumps({"challenge_id": CHALLENGE_ID, "nonce": "n"}),
        self_checks=["self"],
        crypto="crypto",
        expires_at=0.0,
    )


def _make_miner(*, attest_on: bool, is_mounted: bool = False) -> tuple[Miner, list]:
    miner = Miner.__new__(Miner)
    miner.hotkey = "5TestHotkey"
    miner.is_mounted = is_mounted
    miner.run_flags = RunFlags(attest=RunFlag(enabled=attest_on))
    miner._latest_attestation_payloads = {}

    challenge_requests: list = []

    async def fake_request_attestation_challenge(action, run_id=None):
        challenge_requests.append((action, run_id))
        return _challenge(action)

    miner.miner_api_client = SimpleNamespace(request_attestation_challenge=fake_request_attestation_challenge)
    return miner, challenge_requests


@pytest.mark.parametrize("action", ["weights", "merged_partitions"])
async def test_collects_attestation_for_upload_action(monkeypatch, action):
    miner, challenge_requests = _make_miner(attest_on=True)
    expected = MinerAttestationPayload(payload_blob="blob")
    seen_challenges: list[AttestationChallengeResponse] = []

    def fake_collect_attestation_payload(challenge):
        seen_challenges.append(challenge)
        return expected

    monkeypatch.setattr("miner.new_miner.collect_attestation_payload", fake_collect_attestation_payload)

    payload = await miner._collect_attestation_payload(action=action)

    assert payload is expected
    assert challenge_requests == [(action, None)]
    assert len(seen_challenges) == 1
    assert json.loads(seen_challenges[0].challenge_blob)["challenge_id"] == CHALLENGE_ID
    assert miner._latest_attestation_payloads[action] is expected


async def test_mounted_miner_signs_upload_challenge_with_enclave(monkeypatch):
    miner, challenge_requests = _make_miner(attest_on=True, is_mounted=True)
    mounted_payload = MountedAttestationPayload(
        schema_version=1,
        key_id="key-1",
        public_key_base64="pub",
        payload_base64="payload",
        signature_der_base64="sig",
        payload_sha256_base64="sha",
        alg="ES256",
        challenge_id=CHALLENGE_ID,
    )

    async def fake_collect_mounted_attestation(*, challenge_base64, challenge_id):
        assert challenge_id == CHALLENGE_ID
        assert challenge_base64
        return mounted_payload

    miner.collect_mounted_attestation = fake_collect_mounted_attestation
    monkeypatch.setattr(
        "miner.new_miner.collect_attestation_payload",
        lambda _challenge: pytest.fail("mounted miners must not run the host attestation collector"),
    )

    payload = await miner._collect_attestation_payload(action="merged_partitions")

    assert payload is mounted_payload
    assert challenge_requests == [("merged_partitions", None)]


@pytest.mark.parametrize("action", ["weights", "merged_partitions"])
async def test_attest_flag_off_sends_no_attestation(monkeypatch, action):
    miner, challenge_requests = _make_miner(attest_on=False)
    monkeypatch.setattr(
        "miner.new_miner.collect_attestation_payload",
        lambda _challenge: pytest.fail("no attestation should be collected when the run flag is off"),
    )

    assert await miner._collect_attestation_payload(action=action) is None
    assert challenge_requests == []


async def test_no_challenge_issued_sends_no_attestation(monkeypatch):
    miner, _ = _make_miner(attest_on=True)

    async def no_challenge(action, run_id=None):
        return None

    miner.miner_api_client = SimpleNamespace(request_attestation_challenge=no_challenge)
    monkeypatch.setattr(
        "miner.new_miner.collect_attestation_payload",
        lambda _challenge: pytest.fail("nothing to collect without a challenge"),
    )

    assert await miner._collect_attestation_payload(action="weights") is None


@pytest.mark.parametrize(
    "collector_error",
    [AttestationUnavailableError("attest module missing"), RuntimeError("collector crashed")],
)
async def test_collection_failure_does_not_block_upload(monkeypatch, collector_error):
    # The submit still goes out (with attestation=None); the server decides what to
    # do with a missing attestation, the client never drops an upload over it.
    miner, challenge_requests = _make_miner(attest_on=True)

    def failing_collector(_challenge):
        raise collector_error

    monkeypatch.setattr("miner.new_miner.collect_attestation_payload", failing_collector)

    assert await miner._collect_attestation_payload(action="weights") is None
    assert challenge_requests == [("weights", None)]
    assert "weights" not in miner._latest_attestation_payloads


async def test_mounted_miner_falls_back_to_enclave_signature(monkeypatch):
    # When the mounted collector fails, the challenge is signed by the enclave and that
    # EnclaveSignResponse is what gets sent on the upload.
    miner, challenge_requests = _make_miner(attest_on=True, is_mounted=True)
    enclave_payload = EnclaveSignResponse(
        key_id="key-1", signature_der_base64="sig", alg="ES256", challenge_id=CHALLENGE_ID
    )
    sign_calls: list[dict] = []

    async def failing_collect_mounted_attestation(**_kwargs):
        raise RuntimeError("mounted attestation unavailable")

    async def fake_enclave_sign_with_purpose(**kwargs):
        sign_calls.append(kwargs)
        return enclave_payload

    miner.collect_mounted_attestation = failing_collect_mounted_attestation
    miner.enclave_sign_with_purpose = fake_enclave_sign_with_purpose
    monkeypatch.setattr(
        "miner.new_miner.collect_attestation_payload",
        lambda _challenge: pytest.fail("mounted miners must not run the host attestation collector"),
    )

    payload = await miner._collect_attestation_payload(action="weights")

    assert payload is enclave_payload
    assert challenge_requests == [("weights", None)]
    assert len(sign_calls) == 1
    assert sign_calls[0]["purpose"] == "attestation"
    assert sign_calls[0]["challenge_id"] == CHALLENGE_ID
    assert miner._latest_attestation_payloads["weights"] is enclave_payload


# --- Call sites: the collected payload must reach the API client call ---------------


class _NoopTimer:
    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


async def _noop(*_args, **_kwargs):
    return None


def _stub_host_attestation(monkeypatch) -> MinerAttestationPayload:
    expected = MinerAttestationPayload(payload_blob="blob")
    monkeypatch.setattr("miner.new_miner.collect_attestation_payload", lambda _challenge: expected)
    monkeypatch.setattr("miner.new_miner.TimerLoggerMiner", _NoopTimer)
    monkeypatch.setattr("miner.new_miner.log_gpu_memory_usage", lambda **_kwargs: None)
    return expected


async def test_submit_weights_sends_collected_attestation(monkeypatch):
    miner, challenge_requests = _make_miner(attest_on=True)
    expected = _stub_host_attestation(monkeypatch)

    model = torch.nn.Linear(2, 2)
    miner.wallet = SimpleNamespace(hotkey="hotkey-keypair")
    miner.num_partitions = 1
    miner.state_manager = SimpleNamespace(layer=0, run_id="run-1")
    miner._log_resources = lambda *_args, **_kwargs: None
    miner._clear_stale_p2p_state = _noop
    miner.model_manager = SimpleNamespace(
        global_weights_loaded=True,
        model=model,
        optimizer=SimpleNamespace(zero_grad=lambda: None),
    )
    miner.training_phase = SimpleNamespace(
        backwards_since_reset=1,
        local_optimization_steps=1,
        optimization_reset=_noop,
        _cache=SimpleNamespace(reset=_noop),
        _queue=SimpleNamespace(_forward_queue=[], _backward_queue=[]),
        _publisher=SimpleNamespace(reset=_noop),
    )
    previous_weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone() + 1.0
    monkeypatch.setattr("miner.new_miner.load_model_weights", lambda **_kwargs: previous_weights)

    async def fake_upload_weights_blob(**_kwargs):
        return SimpleNamespace(object_path="weights/path")

    monkeypatch.setattr("miner.new_miner.upload_weights_blob", fake_upload_weights_blob)

    submitted: list[WeightUpdate] = []

    async def fake_submit_weights(weight_update):
        submitted.append(weight_update)
        return WeightSubmitResponse(message="ok")

    miner.miner_api_client.submit_weights = fake_submit_weights
    miner.miner_api_client.notify_orchestrator_of_state_call = _noop

    await miner.submit_weights()

    assert challenge_requests == [("weights", None)]
    assert len(submitted) == 1
    assert submitted[0].weights_path == "weights/path"
    assert submitted[0].attestation is expected


async def test_merge_batch_sends_collected_attestation(monkeypatch):
    miner, challenge_requests = _make_miner(attest_on=True)
    expected = _stub_host_attestation(monkeypatch)

    miner.wallet = SimpleNamespace(hotkey="hotkey-keypair")
    miner.num_partitions = 1
    miner.state_manager = SimpleNamespace(layer=0, run_id="run-1")
    miner._log_resources = lambda *_args, **_kwargs: None
    miner.model_manager = SimpleNamespace(global_weights_loaded=True, current_epoch=1)
    monkeypatch.setattr(miner_settings, "N_PARTITION_BATCHES", 1)
    monkeypatch.setattr("miner.new_miner.load_model_weights", lambda **_kwargs: torch.zeros(4))

    partition = SimpleNamespace(chunk_number=0)
    merged = SimpleNamespace(pseudograds=None, old_optimizer_state=None)
    final_partitions = ["final-partition"]

    async def fake_download_pseudograds(**_kwargs):
        return [], []

    async def fake_download_optimizer_state(merging_partitions):
        return merging_partitions

    async def fake_merge_partition_batch(**_kwargs):
        return [merged]

    async def fake_upload_partition_batch(**_kwargs):
        return final_partitions

    async def fake_get_previous_partitions(partition_indices):
        return []

    monkeypatch.setattr("miner.new_miner.get_partition_batch", lambda **_kwargs: [])
    monkeypatch.setattr("miner.new_miner.download_pseudograds_for_partition_batch", fake_download_pseudograds)
    monkeypatch.setattr(
        "miner.new_miner.download_previous_optimizer_state_for_partition_batch", fake_download_optimizer_state
    )
    monkeypatch.setattr("miner.new_miner.merge_partition_batch", fake_merge_partition_batch)
    monkeypatch.setattr("miner.new_miner.upload_partition_batch", fake_upload_partition_batch)

    submitted: list[dict] = []

    async def fake_submit_merged_partitions(merged_partitions, attestation=None):
        submitted.append({"merged_partitions": merged_partitions, "attestation": attestation})
        return {}

    miner.miner_api_client.submit_merged_partitions = fake_submit_merged_partitions
    miner.miner_api_client.get_previous_partitions = fake_get_previous_partitions

    await miner.merge_partitions(weight_path_per_layer=[], partitions=[partition])

    assert challenge_requests == [("merged_partitions", None)]
    assert len(submitted) == 1
    assert submitted[0]["merged_partitions"] is final_partitions
    assert submitted[0]["attestation"] is expected
