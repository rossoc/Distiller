# -*- coding: utf-8 -*-
"""The Hydra config groups that drive scripts/momos_phase_c.py.

Without these, a malformed momos/*.yaml or an arm named in gate.arms with no
matching file only surfaces partway through a sweep that takes the better part
of an hour.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[3] / "src" / "config"
MOMOS_DIR = CONFIG_DIR / "momos"
GATE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "momos_phase_c.py"

ARM_NAMES = sorted(p.stem for p in MOMOS_DIR.glob("*.yaml"))


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("momos_phase_c", GATE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_there_is_at_least_one_method_to_compare():
    assert ARM_NAMES, f"no momos/*.yaml groups found under {MOMOS_DIR}"


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_every_method_group_builds_a_valid_mosaic_config(arm):
    """Each momos/*.yaml must survive MosaicConfig's __post_init__ validation."""
    gate = _load_gate_module()
    method = OmegaConf.load(MOMOS_DIR / f"{arm}.yaml")
    assert method.method == arm, (
        f"momos/{arm}.yaml declares method={method.method!r}; the file stem is "
        f"what gate.arms refers to, so they must agree"
    )
    for S in (1, 2, 4):
        cfg = gate._mosaic_config(method, S=S, K=1024, learning_rate=6e-3)
        assert cfg.S == S and cfg.K == 1024
        # base_lr comes from the run, never from the file (SPEC_PHASE_C2.md §4.2
        # needs it to match the optimiser's actual learning rate).
        assert cfg.base_lr == 6e-3


def test_static_group_reproduces_phase_b():
    """SPEC_PHASE_C2.md §2: both escape hatches off means bit-for-bit Phase B."""
    gate = _load_gate_module()
    cfg = gate._mosaic_config(
        OmegaConf.load(MOMOS_DIR / "static.yaml"), S=2, K=1024, learning_rate=6e-3
    )
    assert cfg.subset_size == 0
    assert cfg.cohort_frac == 0.0


def test_root_config_composes_and_every_named_arm_has_a_group():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config_momos")

    arms = [str(a) for a in cfg.gate.arms]
    assert arms, "gate.arms is empty — nothing would be compared"
    missing = [a for a in arms if a not in ARM_NAMES]
    assert not missing, f"gate.arms names {missing} with no momos/*.yaml to match"

    # The first arm is the control every ratio column is taken against.
    assert arms[0] == "static", f"expected 'static' as the control arm, got {arms[0]!r}"


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_each_method_group_is_selectable_as_a_hydra_override(arm):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config_momos", overrides=[f"momos={arm}"])
    assert cfg.momos.method == arm
