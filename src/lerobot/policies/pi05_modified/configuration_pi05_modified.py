#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_modified")
@dataclass
class PI05ModifiedConfig(PI05Config):
    """PI0.5 modified config for safety-guided inference."""

    safety_guidance_eta: float = 0.0
