"""Run-level training settings the miner applies from model_metadata on each run-config refresh.

The miner's training code reads these from common_settings at call time, so overriding the
module attributes here changes them for the next step / merge.
"""

from common import settings as common_settings

# Hard bounds for the settings a run may override (opt.learning_rate, opt.outer_learning_rate,
# opt.weight_decay, grad_clip_norm). Out-of-range values are clamped.
TRAINING_SETTING_BOUNDS: dict[str, tuple[float, float]] = {
    "LEARNING_RATE": (1e-6, 1e-3),
    "NESTEROV_LEARNING_RATE": (0.1, 1.0),
    "WEIGHT_DECAY": (0.0, 0.3),
    "GRAD_CLIP_NORM": (0.1, 10.0),
}
_DEFAULTS = {name: getattr(common_settings, name) for name in TRAINING_SETTING_BOUNDS}


def set_training_setting(name: str, value: float | None) -> None:
    """Apply a run's training setting, clamped to TRAINING_SETTING_BOUNDS.

    None (not falsy) restores the release default, because 0.0 is a valid weight decay.
    """
    if value is None:
        setattr(common_settings, name, _DEFAULTS[name])
        return
    lo, hi = TRAINING_SETTING_BOUNDS[name]
    setattr(common_settings, name, min(max(float(value), lo), hi))
