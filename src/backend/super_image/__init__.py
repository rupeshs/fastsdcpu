"""
super-image package.

A library of image super resolution algorithms in PyTorch. For upscaling images.
"""

from .configuration_utils import PretrainedConfig
from .modeling_utils import PreTrainedModel

from .models import (
    EdsrModel,
    EdsrConfig,
    MsrnModel,
    MsrnConfig,
    A2nModel,
    A2nConfig,
    PanModel,
    PanConfig,
    MasaModel,
    MasaConfig,
    CarnModel,
    CarnConfig,
    JiifModel,
    JiifConfig,
    LiifModel,
    LiifConfig,
    SmsrModel,
    SmsrConfig,
    RcanModel,
    RcanConfig,
    DrlnModel,
    DrlnConfig,
    MdsrModel,
    MdsrConfig,
    DrnModel,
    DrnConfig,
    PhysicssrModel,
    PhysicssrConfig,
    HanModel,
    HanConfig,
    AwsrnModel,
    AwsrnConfig,
    RnanModel,
    RnanConfig,
    DdbpnModel,
    DdbpnConfig,
)

from .data import ImageLoader

from typing import List

__all__: List[str] = [
    "TrainingArguments",
    "Trainer",
    "TrainerDrn",
    "EdsrModel",
    "EdsrConfig",
    "MsrnModel",
    "MsrnConfig",
    "A2nModel",
    "A2nConfig",
    "PanModel",
    "PanConfig",
    "MasaModel",
    "MasaConfig",
    "CarnModel",
    "CarnConfig",
    "JiifModel",
    "JiifConfig",
    "LiifModel",
    "LiifConfig",
    "SmsrModel",
    "SmsrConfig",
    "DrlnModel",
    "DrlnConfig",
    "RcanModel",
    "RcanConfig",
    "MdsrModel",
    "MdsrConfig",
    "DrnModel",
    "DrnConfig",
    "HanModel",
    "HanConfig",
    "PhysicssrModel",
    "PhysicssrConfig",
    "AwsrnModel",
    "AwsrnConfig",
    "RnanModel",
    "RnanConfig",
    "DdbpnModel",
    "DdbpnConfig",
    "ImageLoader",
]  # noqa: WPS410 (the only __variable__ we use)
