#!/usr/bin/env python
"""
Download the model weights MuseTalk V1.5 needs, at pinned revisions, and verify
them against SHA-256 hashes tracked in the repo.

Run it inside the backend container, with the MuseTalk venv's Python:

    docker compose exec backend /opt/musetalk-venv/bin/python scripts/download_musetalk_weights.py

Files that already exist are not downloaded again. Use --record once to write
the hashes of the current files into musetalk_weights.sha256.
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODELS = Path(os.environ.get("MUSETALK_PATH", "/opt/MuseTalk")) / "models"
TORCH_CKPT = Path(os.environ.get("TORCH_HOME", "/root/.cache/torch")) / "hub" / "checkpoints"
MANIFEST = Path(__file__).with_name("musetalk_weights.sha256")

REV_MUSETALK = "2bcb936e2fddb4d86db4c62fd45b387d0c061571"
REV_WHISPER = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
REV_VAE = "31f26fdeee1355a5c34592e401dd41e45d25a493"

# (repo, revision, file in repo, local directory). Only what V1.5 inference
# needs: no V1.0 weights, no DWPose (replaced by our preprocessing patch), and
# no SyncNet (training only).
HF_FILES = [
    ("TMElyralab/MuseTalk", REV_MUSETALK, "musetalkV15/musetalk.json", MODELS),
    ("TMElyralab/MuseTalk", REV_MUSETALK, "musetalkV15/unet.pth", MODELS),
    ("openai/whisper-tiny", REV_WHISPER, "config.json", MODELS / "whisper"),
    ("openai/whisper-tiny", REV_WHISPER, "pytorch_model.bin", MODELS / "whisper"),
    ("openai/whisper-tiny", REV_WHISPER, "preprocessor_config.json", MODELS / "whisper"),
    ("stabilityai/sd-vae-ft-mse", REV_VAE, "config.json", MODELS / "sd-vae"),
    ("stabilityai/sd-vae-ft-mse", REV_VAE, "diffusion_pytorch_model.bin", MODELS / "sd-vae"),
]

# (url, destination)
URL_FILES = [
    (
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        MODELS / "face-parse-bisent" / "resnet18-5c106cde.pth",
    ),
    # face_alignment looks for these two under TORCH_HOME/hub/checkpoints.
    (
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        TORCH_CKPT / "s3fd-619a316812.pth",
    ),
    (
        "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip",
        TORCH_CKPT / "2DFAN4-cd938726ad.zip",
    ),
]

GDRIVE_BISENET_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
BISENET_DEST = MODELS / "face-parse-bisent" / "79999_iter.pth"


def key_for(path: Path) -> str:
    """Stable manifest key, independent of where the volumes are mounted."""
    try:
        return "models/" + path.relative_to(MODELS).as_posix()
    except ValueError:
        return "torch/" + path.relative_to(TORCH_CKPT).as_posix()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    out = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, key = line.split(None, 1)
            out[key.strip()] = digest
    return out


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, part)
    part.rename(dest)  # a half-downloaded file never looks complete


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="write the hashes of the current files to the manifest")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    wanted = []  # every destination, in order

    for repo, rev, filename, local_dir in HF_FILES:
        dest = local_dir / filename
        wanted.append(dest)
        if dest.exists():
            continue
        print(f"  downloading {repo}@{rev[:7]} {filename}")
        hf_hub_download(repo_id=repo, filename=filename, revision=rev,
                        local_dir=str(local_dir), local_dir_use_symlinks=False)

    for url, dest in URL_FILES:
        wanted.append(dest)
        if not dest.exists():
            download_url(url, dest)

    wanted.append(BISENET_DEST)
    if not BISENET_DEST.exists():
        import gdown
        BISENET_DEST.parent.mkdir(parents=True, exist_ok=True)
        print("  downloading BiSeNet from Google Drive")
        gdown.download(id=GDRIVE_BISENET_ID, output=str(BISENET_DEST), quiet=False)

    manifest = read_manifest()
    bad, unhashed = [], []
    for dest in wanted:
        if not dest.is_file():
            bad.append(f"MISSING  {key_for(dest)}")
            continue
        key, digest = key_for(dest), sha256_of(dest)
        if args.record:
            manifest[key] = digest
        elif key not in manifest:
            unhashed.append(key)
        elif manifest[key] != digest:
            bad.append(f"MISMATCH {key}")
        print(f"  ok  {key}  {dest.stat().st_size / 1e6:.1f} MB")

    if args.record:
        lines = ["# SHA-256 of the MuseTalk weights (see scripts/download_musetalk_weights.py)"]
        lines += [f"{d}  {k}" for k, d in sorted(manifest.items())]
        MANIFEST.write_text("\n".join(lines) + "\n")
        print(f"Manifest written: {MANIFEST}")

    for k in unhashed:
        print(f"WARNING: no recorded hash for {k} (run with --record)")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    print("All weights present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
