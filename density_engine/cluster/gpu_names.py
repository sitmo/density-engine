#!/usr/bin/env python3
"""
Simple GPU names for Vast.ai API - under 30 lines.
"""

# All GPU names from Vast.ai API (with underscores for search queries)
ALL_GPU_NAMES = [
    "A10",
    "A100_PCIE",
    "A100_SXM4",
    "A100X",
    "A40",
    "A800_PCIE",
    "B200",
    "GH200_SXM",
    "GTX_1050_Ti",
    "GTX_1060",
    "GTX_1070",
    "GTX_1070_Ti",
    "GTX_1080",
    "GTX_1080_Ti",
    "GTX_1650",
    "GTX_1660",
    "GTX_1660_S",
    "GTX_1660_Ti",
    "GTX_TITAN_X",
    "H100_NVL",
    "H100_PCIE",
    "H100_SXM",
    "H200",
    "H200_NVL",
    "InstinctMI50",
    "L4",
    "L40",
    "L40S",
    "P104-100",
    "Q_RTX_4000",
    "Q_RTX_6000",
    "Q_RTX_8000",
    "Quadro_K2200",
    "Quadro_P4000",
    "RTX_2060",
    "RTX_2060S",
    "RTX_2070",
    "RTX_2070S",
    "RTX_2080",
    "RTX_2080S",
    "RTX_2080_Ti",
    "RTX_3050",
    "RTX_3060",
    "RTX_3060_laptop",
    "RTX_3060_Ti",
    "RTX_3070",
    "RTX_3070_laptop",
    "RTX_3070_Ti",
    "RTX_3080",
    "RTX_3080_Ti",
    "RTX_3090",
    "RTX_3090_Ti",
    "RTX_4060",
    "RTX_4060_Ti",
    "RTX_4070",
    "RTX_4070S",
    "RTX_4070S_Ti",
    "RTX_4070_Ti",
    "RTX_4080",
    "RTX_4080S",
    "RTX_4090",
    "RTX_4090D",
    "RTX_4500Ada",
    "RTX_5000Ada",
    "RTX_5060",
    "RTX_5060_Ti",
    "RTX_5070",
    "RTX_5070_Ti",
    "RTX_5080",
    "RTX_5090",
    "RTX_5880Ada",
    "RTX_6000Ada",
    "RTX_A2000",
    "RTX_A4000",
    "RTX_A4500",
    "RTX_A5000",
    "RTX_A6000",
    "RTX_PRO_6000_S",
    "RTX_PRO_6000_WS",
    "Tesla_P100",
    "Tesla_P4",
    "Tesla_P40",
    "Tesla_T4",
    "Tesla_V100",
    "Titan_V",
    "Titan_Xp",
]

# RTX 30 series and above
RTX_30_PLUS = [
    gpu
    for gpu in ALL_GPU_NAMES
    if "RTX" in gpu and any(series in gpu for series in ["30", "40", "50"])
]

# RTX 40 series and above
RTX_40_PLUS = [
    gpu
    for gpu in ALL_GPU_NAMES
    if "RTX" in gpu and any(series in gpu for series in ["40", "50"])
]

# RTX 50 series only
RTX_50_ONLY = [gpu for gpu in ALL_GPU_NAMES if "RTX" in gpu and "50" in gpu]

if __name__ == "__main__":
    print("All GPUs:", len(ALL_GPU_NAMES))
    print("RTX 30+:", len(RTX_30_PLUS))
    print("RTX 40+:", len(RTX_40_PLUS))
    print("RTX 50:", len(RTX_50_ONLY))
