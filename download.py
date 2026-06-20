"""Modal download entry for TripoSplat.

Run:
  modal run download.py::download

Fetches the 5 TripoSplat weight files onto the shared ``models`` volume, laid out
under /models/triposplat mirroring the VAST-AI/TripoSplat HF repo so deploy.py can
load them by path. The repo is public (MIT); HF_TOKEN is optional and injected
from TongFlow Settings (see the Secret.from_dict below).
"""

from __future__ import annotations

import os
from typing import Any

import modal

_cfg: dict[str, Any] = {}

CKPTS = "/models/triposplat"

# (repo_id, path-in-repo, dest-subdir, dest-name) — paths match deploy.py.
MODELS = [
    (
        "VAST-AI/TripoSplat",
        "diffusion_models/triposplat_fp16.safetensors",
        "diffusion_models",
        "triposplat_fp16.safetensors",
    ),
    (
        "VAST-AI/TripoSplat",
        "vae/triposplat_vae_decoder_fp16.safetensors",
        "vae",
        "triposplat_vae_decoder_fp16.safetensors",
    ),
    (
        "VAST-AI/TripoSplat",
        "vae/flux2-vae.safetensors",
        "vae",
        "flux2-vae.safetensors",
    ),
    (
        "VAST-AI/TripoSplat",
        "clip_vision/dino_v3_vit_h.safetensors",
        "clip_vision",
        "dino_v3_vit_h.safetensors",
    ),
    (
        "VAST-AI/TripoSplat",
        "background_removal/birefnet.safetensors",
        "background_removal",
        "birefnet.safetensors",
    ),
]

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

model_downloader = modal.App("model_downloader")

_download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.34.0,<1.0"
)


@model_downloader.function(
    image=_download_image,
    volumes={"/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def _download() -> None:
    import shutil

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or None

    for repo, path, subdir, name in MODELS:
        dest_dir = os.path.join(CKPTS, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"skip (exists): {subdir}/{name}")
            continue
        print(f"Downloading {repo}/{path} ...")
        src = hf_hub_download(repo_id=repo, filename=path, token=token)
        shutil.copyfile(src, dest)
        print(f"  got {subdir}/{name} ({os.path.getsize(dest) // (1024 * 1024)} MB)")
        # Commit after each large file so a later failure doesn't re-download it.
        volume.commit()

    print("Done.")


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
