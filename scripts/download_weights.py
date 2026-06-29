#!/usr/bin/env python3
"""
Download pretrained PyTorch weights for the DL-VINS factory feature stack.

Sources are pinned to the same URLs used by ref/ code and LightGlue-ONNX-Jetson.
SuperPoint-Open (rpautrat) uses the v6-from-TF weights directory on GitHub.
Large files are written under ./weights/ (gitignored); this script and weights/
layout stay in the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str | None = None  # optional verify

    def dest(self, root: Path) -> Path:
        return root / self.name


# Release page: https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv
LG_V01 = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv"

ASSETS: tuple[Asset, ...] = (
    Asset("superpoint_v1.pth", f"{LG_V01}/superpoint_v1.pth"),
    Asset("superpoint_lightglue.pth", f"{LG_V01}/superpoint_lightglue.pth"),
    Asset(
        "superpoint_v6_from_tf.pth",
        "https://github.com/rpautrat/SuperPoint/raw/master/weights/superpoint_v6_from_tf.pth",
    ),
    Asset(
        "aliked-n16.pth",
        "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth",
    ),
    Asset(
        "aliked-n16rot.pth",
        "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16rot.pth",
    ),
    Asset(
        "aliked-n32.pth",
        "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n32.pth",
    ),
    Asset("aliked_lightglue.pth", f"{LG_V01}/aliked_lightglue.pth"),
    Asset("sift_lightglue.pth", f"{LG_V01}/sift_lightglue.pth"),
    Asset(
        "xfeat.pt",
        "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt",
    ),
    Asset(
        "xfeat-lighterglue.pt",
        "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt",
    ),
    Asset(
        "raco.pth",
        "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth",
    ),
    Asset(
        "raco_aliked_lightglue.pth",
        f"{LG_V01}/raco_aliked_lightglue.pth",
        sha256="a804c3fe2851cc3c765cda6a9f412845893c8d8e2a3048f848d7f6ba172e6312",
    ),
)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out.write(block)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError):
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "weights",
        help="Output directory (default: repo ./weights)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="NAME",
        help="If set, only download these basenames (e.g. superpoint_v6_from_tf.pth xfeat.pt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without fetching",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()
    dest_root: Path = args.dest
    only = set(args.only) if args.only else None

    todo = [a for a in ASSETS if only is None or a.name in only]
    if only is not None:
        missing = only - {a.name for a in todo}
        if missing:
            print("Unknown --only names:", ", ".join(sorted(missing)), file=sys.stderr)
            return 2

    for a in todo:
        out = a.dest(dest_root)
        if args.dry_run:
            print(f"{a.name} <- {a.url}")
            continue
        if out.exists() and not args.force:
            print(f"skip (exists): {out}")
            continue
        print(f"fetch: {a.name}")
        download(a.url, out)
        if a.sha256:
            got = _sha256_file(out)
            if got != a.sha256:
                print(
                    f"SHA256 mismatch for {a.name}: expected {a.sha256}, got {got}",
                    file=sys.stderr,
                )
                out.unlink(missing_ok=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
