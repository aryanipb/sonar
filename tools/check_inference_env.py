#!/usr/bin/env python3
import argparse
import importlib
import platform
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def _mod_version(name):
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Check AdaPoinTr inference environment")
    parser.add_argument("--config", default="", help="Optional model config to instantiate")
    parser.add_argument("--smoke_test", action="store_true", help="Run dummy forward pass")
    parser.add_argument("--device", default="", help="Device for smoke test; default auto")
    args = parser.parse_args()

    print("=== Runtime ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

    ok_torch, torch_info = _mod_version("torch")
    if not ok_torch:
        print(f"torch: MISSING ({torch_info})")
        return 1

    import torch

    print("=== Core Packages ===")
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    for pkg in ["open3d", "timm", "cv2", "numpy", "yaml"]:
        ok, info = _mod_version(pkg)
        print(f"{pkg}: {'OK ' + info if ok else 'MISSING (' + info + ')'}")

    for pkg in ["pointnet2_ops", "extensions.chamfer_dist"]:
        ok, info = _mod_version(pkg)
        print(f"{pkg}: {'OK ' + info if ok else 'OPTIONAL_UNAVAILABLE (' + info + ')'}")

    if not args.config:
        return 0

    print("=== Model Build ===")
    from utils.config import cfg_from_yaml_file
    from tools import builder

    cfg = cfg_from_yaml_file(args.config)
    model = builder.model_builder(cfg.model)
    model.eval()
    print(f"model: built {cfg.model.NAME}")

    if args.smoke_test:
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        dummy = torch.randn(1, 2048, 3, device=device)
        with torch.inference_mode():
            out = model(dummy)
        if isinstance(out, (list, tuple)):
            print("smoke_output:", [tuple(t.shape) for t in out])
        else:
            print("smoke_output:", tuple(out.shape))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
