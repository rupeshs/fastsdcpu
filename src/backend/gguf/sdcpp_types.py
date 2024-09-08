"""
Ctypes for stablediffusion.cpp shared library
This is as per the stablediffusion.h  file
"""

from enum import IntEnum
from ctypes import (
    c_int,
    c_uint32,
    c_uint8,
    POINTER,
    Structure,
)


class CtypesEnum(IntEnum):
    """A ctypes-compatible IntEnum superclass."""

    @classmethod
    def from_param(cls, obj):
        return int(obj)


class RngType(CtypesEnum):
    STD_DEFAULT_RNG = 0
    CUDA_RNG = 1


class SampleMethod(CtypesEnum):
    EULER_A = 0
    EULER = 1
    HEUN = 2
    DPM2 = 3
    DPMPP2S_A = 4
    DPMPP2M = 5
    DPMPP2Mv2 = 6
    IPNDM = 7
    IPNDM_V = 7
    LCM = 8
    N_SAMPLE_METHODS = 9


class Schedule(CtypesEnum):
    DEFAULT = 0
    DISCRETE = 1
    KARRAS = 2
    EXPONENTIAL = 3
    AYS = 4
    GITS = 5
    N_SCHEDULES = 5


class SdType(CtypesEnum):
    SD_TYPE_F32 = 0
    SD_TYPE_F16 = 1
    SD_TYPE_Q4_0 = 2
    SD_TYPE_Q4_1 = 3
    # SD_TYPE_Q4_2 = 4, support has been removed
    # SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0 = 6
    SD_TYPE_Q5_1 = 7
    SD_TYPE_Q8_0 = 8
    SD_TYPE_Q8_1 = 9
    SD_TYPE_Q2_K = 10
    SD_TYPE_Q3_K = 11
    SD_TYPE_Q4_K = 12
    SD_TYPE_Q5_K = 13
    SD_TYPE_Q6_K = 14
    SD_TYPE_Q8_K = 15
    SD_TYPE_IQ2_XXS = 16
    SD_TYPE_IQ2_XS = 17
    SD_TYPE_IQ3_XXS = 18
    SD_TYPE_IQ1_S = 19
    SD_TYPE_IQ4_NL = 20
    SD_TYPE_IQ3_S = 21
    SD_TYPE_IQ2_S = 22
    SD_TYPE_IQ4_XS = 23
    SD_TYPE_I8 = 24
    SD_TYPE_I16 = 25
    SD_TYPE_I32 = 26
    SD_TYPE_I64 = 27
    SD_TYPE_F64 = 28
    SD_TYPE_IQ1_M = 29
    SD_TYPE_BF16 = 30
    SD_TYPE_Q4_0_4_4 = 31
    SD_TYPE_Q4_0_4_8 = 32
    SD_TYPE_Q4_0_8_8 = 33
    SD_TYPE_COUNT = 34


class SDImage(Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("channel", c_uint32),
        ("data", POINTER(c_uint8)),
    ]


class SDCPPLogLevel(c_int):
    SD_LOG_LEVEL_DEBUG = 0
    SD_LOG_LEVEL_INFO = 1
    SD_LOG_LEVEL_WARNING = 2
    SD_LOG_LEVEL_ERROR = 3
