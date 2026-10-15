"""The periodic run-config refresh only works if updates land in-place: flag
holders keep their reference to miner.run_flags, and hyperparameter consumers
read common_settings attributes at call time."""

from types import SimpleNamespace

from common import settings as common_settings
from common.models.run_flags import RunFlags

from miner.new_miner import Miner
from miner.utils.training_settings import TRAINING_SETTING_BOUNDS, set_training_setting


def _stub() -> SimpleNamespace:
    return SimpleNamespace(hotkey="hk-test-1234", run_flags=RunFlags())


def test_update_run_flags_mutates_shared_reference():
    stub = _stub()
    shared_ref = stub.run_flags  # what e.g. TrainingPhase holds

    new_flags = RunFlags()
    new_flags.sync_patches.enabled = not shared_ref.sync_patches.enabled
    Miner._update_run_flags(stub, new_flags)

    assert shared_ref.sync_patches.enabled == new_flags.sync_patches.enabled


def test_apply_run_hyperparams_updates_cache_size_live():
    stub = _stub()
    old = common_settings.MAX_ACTIVATION_CACHE_SIZE
    try:
        Miner._apply_run_hyperparams(stub, {"activation": {"max_activation_cache_size": old + 16}})
        assert common_settings.MAX_ACTIVATION_CACHE_SIZE == old + 16

        # Absent/None values must leave settings untouched
        Miner._apply_run_hyperparams(stub, {})
        assert common_settings.MAX_ACTIVATION_CACHE_SIZE == old + 16
    finally:
        common_settings.set_max_activation_cache_size(old)


_TRAINING = ("LEARNING_RATE", "NESTEROV_LEARNING_RATE", "WEIGHT_DECAY", "GRAD_CLIP_NORM")


def _restore_training_defaults():
    for name in _TRAINING:
        set_training_setting(name, None)


def test_apply_run_hyperparams_updates_training_settings_live():
    stub = _stub()
    try:
        Miner._apply_run_hyperparams(
            stub,
            {"opt": {"learning_rate": 1.5e-4, "outer_learning_rate": 0.6, "weight_decay": 0.05}, "grad_clip_norm": 2.0},
        )
        assert common_settings.LEARNING_RATE == 1.5e-4
        assert common_settings.NESTEROV_LEARNING_RATE == 0.6
        assert common_settings.WEIGHT_DECAY == 0.05
        assert common_settings.GRAD_CLIP_NORM == 2.0
    finally:
        _restore_training_defaults()


def test_training_settings_are_clamped_to_bounds():
    stub = _stub()
    try:
        Miner._apply_run_hyperparams(stub, {"opt": {"learning_rate": 1.0, "outer_learning_rate": 0.0}})
        assert common_settings.LEARNING_RATE == TRAINING_SETTING_BOUNDS["LEARNING_RATE"][1]
        assert common_settings.NESTEROV_LEARNING_RATE == TRAINING_SETTING_BOUNDS["NESTEROV_LEARNING_RATE"][0]
    finally:
        _restore_training_defaults()


def test_zero_weight_decay_is_applied_not_treated_as_absent():
    stub = _stub()
    try:
        Miner._apply_run_hyperparams(stub, {"opt": {"weight_decay": 0.0}})
        assert common_settings.WEIGHT_DECAY == 0.0
    finally:
        _restore_training_defaults()


def test_absent_training_settings_restore_defaults():
    # A run that drops an override (or an older payload without the new field) goes back to
    # the release defaults rather than keeping the last override.
    stub = _stub()
    defaults = {name: getattr(common_settings, name) for name in _TRAINING}
    try:
        Miner._apply_run_hyperparams(stub, {"opt": {"learning_rate": 1e-4, "outer_learning_rate": 0.5}})
        Miner._apply_run_hyperparams(stub, {"opt": {"learning_rate": None}})
        assert {name: getattr(common_settings, name) for name in _TRAINING} == defaults
    finally:
        _restore_training_defaults()


def test_todays_stored_run_values_change_nothing():
    # Live runs store exactly the release constants (filled at run creation), so honouring
    # them is a no-op until someone patches a run.
    stub = _stub()
    defaults = {name: getattr(common_settings, name) for name in _TRAINING}
    try:
        Miner._apply_run_hyperparams(
            stub,
            {
                "opt": {"learning_rate": 2e-4, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
                "grad_clip_norm": 1.0,
            },
        )
        assert {name: getattr(common_settings, name) for name in _TRAINING} == defaults
    finally:
        _restore_training_defaults()
