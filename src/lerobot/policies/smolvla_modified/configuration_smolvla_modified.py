#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_modified")
@dataclass
class SmolVLAModifiedConfig(SmolVLAConfig):
    """SmolVLA modified config for safety-guided inference."""

    safety_guidance_eta: float = 0.0
