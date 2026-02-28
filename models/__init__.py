from .build import build_model_from_cfg
import importlib
import warnings

_MODEL_MODULES = [
    "models.TopNet",
    "models.PoinTr",
    "models.GRNet",
    "models.PCN",
    "models.FoldingNet",
    "models.SnowFlakeNet",
    "models.AdaPoinTr",
]

for _module in _MODEL_MODULES:
    try:
        importlib.import_module(_module)
    except Exception as _exc:
        warnings.warn(f"Skipping optional model module {_module}: {_exc}")
