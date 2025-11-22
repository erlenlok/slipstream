"""Volume Generator Strategy: Makes in-and-out trades to generate volume."""

from .config import load_volume_generator_config, VolumeBotConfig
from .volume_bot import VolumeGeneratorBot

__all__ = ["VolumeGeneratorBot", "VolumeBotConfig", "load_volume_generator_config"]