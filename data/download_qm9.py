"""Download and unpack the QM9 dataset into data/raw.

Usage:
    python data/download_qm9.py --output data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
# Set to None to skip checksum verification
QM9_SHA256 = None
CHUNK_SIZE = 1024 * 1024  # 1 MB


def sha256sum(path: Path) -> str:
    """Compute SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> Path:
    """Stream download to file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
            f.write(chunk)
    return dest


def verify_checksum(path: Path, expected: str | None) -> bool:
    """Verify file checksum if expected is provided."""
    if expected is None:
        return True
    actual = sha256sum(path)
    return actual == expected.lower()


def safe_extract_tar(archive_path: Path, out_dir: Path) -> None:
    """Extract tar.gz ensuring members stay within out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = out_dir / member.name
            if not member_path.resolve().is_relative_to(out_dir.resolve()):
                raise RuntimeError(f"Unsafe path in tar: {member.name}")
        tar.extractall(path=out_dir, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract QM9 dataset.")
    parser.add_argument("--url", type=str, default=QM9_URL, help="QM9 tarball URL")
    parser.add_argument("--output", type=Path, default=Path("data/raw"), help="Directory to store raw data")
    parser.add_argument("--checksum", type=str, default=QM9_SHA256, help="Expected SHA256 checksum (optional)")
    args = parser.parse_args()

    raw_dir = args.output
    archive_path = raw_dir / "gdb9.tar.gz"

    if archive_path.exists():
        print(f"Found existing archive at {archive_path}")
    else:
        print(f"Downloading QM9 from {args.url} to {archive_path} ...")
        download(args.url, archive_path)
        print("Download complete.")

    if args.checksum:
        print("Verifying checksum...")
        if not verify_checksum(archive_path, args.checksum):
            raise RuntimeError("Checksum verification failed.")
        print("Checksum verified.")
    else:
        print("Checksum verification skipped (no checksum provided).")

    print(f"Extracting {archive_path} to {raw_dir} ...")
    safe_extract_tar(archive_path, raw_dir)
    print("Extraction complete.")


if __name__ == "__main__":
    main()
