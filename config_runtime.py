from pathlib import Path


WEIGHT_CANDIDATES = [
    "best.pt",
    "../Final_Models/Model_L_100Epoch/best.pt",
]

WEIGHTS_7CLS = next(
    (path for path in WEIGHT_CANDIDATES if Path(path).exists()),
    WEIGHT_CANDIDATES[0],
)
DATA_7CLS = "data_power_safety.yaml"
EXPECTED_NC = 5
