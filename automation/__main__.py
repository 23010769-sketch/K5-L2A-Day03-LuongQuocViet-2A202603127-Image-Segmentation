"""Safe command-line inspection. No command in this module writes to CVAT."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from automation.annotation_review import atomic_json
from automation.checkpoint_loader import load_checkpoints
from automation.config import ROOT, Settings
from automation.cvat_client import CvatClient
from automation.export_validator import inspect_exports
from automation.segmentation_engine import resources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Ảnh, labels và quy tắc từ repo")
    sub.add_parser("doctor", help="Python/thư viện/CUDA và policy flags; không gọi API")
    inspect = sub.add_parser("inspect", help="Kiểm ZIP và mask pixels, không sửa file")
    inspect.add_argument("--dir", type=Path, default=ROOT / "submissions")
    inspect.add_argument("--out", type=Path)
    status = sub.add_parser("cvat-status", help="Chỉ GET metadata khi có quyền API và PAT")
    status.add_argument("checkpoint", choices=load_checkpoints())
    status.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = Settings.from_env()
    try:
        if args.command == "list":
            result = [{"checkpoint": cp.name, "type": cp.kind, "images": [str(path.relative_to(ROOT)) for path in cp.images],
                       "labels": cp.labels, "format": cp.export_format, "rule": cp.rule}
                      for cp in load_checkpoints().values()]
        elif args.command == "doctor":
            result = {"python": sys.version, "resources": resources(), "api_allowed": settings.api_allowed,
                      "token_configured": bool(settings.token), "write_allowed": settings.write_allowed,
                      "annotation_read_allowed": settings.annotation_read_allowed,
                      "local_models_allowed": settings.local_models_allowed}
        elif args.command == "inspect":
            result = inspect_exports(args.dir)
            if args.out:
                atomic_json(args.out, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return int(bool(result["error_count"]))
        else:
            try:
                with CvatClient(settings) as client:
                    result = client.inspect(args.task_id, load_checkpoints()[args.checkpoint])
            except Exception as exc:
                print(f"CVAT chưa kết nối/xác minh ({type(exc).__name__}). Kiểm policy flags, PAT, task ID và schema; không ghi dữ liệu.", file=sys.stderr)
                return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
