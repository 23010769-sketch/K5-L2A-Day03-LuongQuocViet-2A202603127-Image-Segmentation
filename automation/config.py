"""Explicit policy switches and environment-only credentials."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "automation" / ".runtime"


@dataclass(frozen=True)
class Settings:
    host: str = "https://app.cvat.ai"
    token: str = field(default="", repr=False)
    organization: str = ""
    api_allowed: bool = False
    annotation_read_allowed: bool = False
    write_allowed: bool = False
    local_models_allowed: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        enabled = lambda key: os.getenv(key, "0") == "1"
        return cls(
            host=os.getenv("CVAT_HOST", "https://app.cvat.ai").rstrip("/"),
            token=os.getenv("CVAT_ACCESS_TOKEN", ""),
            organization=os.getenv("CVAT_ORG", ""),
            api_allowed=enabled("CVAT_API_ALLOWED"),
            annotation_read_allowed=enabled("CVAT_ANNOTATION_READ_ALLOWED"),
            write_allowed=enabled("CVAT_WRITE_ALLOWED"),
            local_models_allowed=enabled("LOCAL_MODELS_ALLOWED"),
        )

    def require_api(self) -> None:
        if not self.api_allowed:
            raise PermissionError("Chưa xác nhận trường cho phép CVAT cloud/API (CVAT_API_ALLOWED=1).")
        parts = urlsplit(self.host)
        if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
            raise ValueError("CVAT_HOST phải là origin, không chứa credentials hoặc đường dẫn.")
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("Chỉ dùng HTTPS, hoặc HTTP cho CVAT trên localhost.")
        if not self.token or self.token == "replace-with-your-personal-access-token":
            raise PermissionError("Thiếu CVAT_ACCESS_TOKEN; không dán token vào source hoặc chat.")
