#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


@PreTrainedConfig.register_subclass("xvla_modified")
@dataclass
class XVLAModifiedConfig(XVLAConfig):
    """XVLA modified config for safety-guided inference."""

    safety_guidance_eta: float = 0.0
