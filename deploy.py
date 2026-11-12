"""Modal deploy entry for TripoSplat (single image -> 3D Gaussian Splatting).

Implements the ``image-gen-model`` slot: one input image -> one ``.splat`` 3D
Gaussian-splat asset. TripoSplat (TripoAI/VAST, MIT) is a self-contained pipeline
(``triposplat.py`` + ``model.py``, vendored next to this file and baked into the
image) with near-zero deps, so there is no ComfyUI server here.

The pipeline boots once (@modal.enter) and is reused across calls; weights live on
the shared ``models`` volume under /models/triposplat (laid out by download.py).

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from tongflow import deploy
from tongflow.models.image_gen_model import ImageGenModelInput, ImageGenModelOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import current_params, node_slot


def _adv(name: str, default):
    """Advanced-section override (``TONGFLOW_SLOT_PARAMS``) or the plugin default."""
    v = current_params().get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v

# Per-run knobs offered under the node's collapsed "Advanced" section.
# Pure literal (the platform scanner reads it by AST, never imports this
# module). Values reach the handlers via current_params(); an untouched
# control is absent there and falls back to the plugin default.
TONGFLOW_SLOT_PARAMS = {
    "image-gen-model": {
        "steps": {"type": "integer", "default": 20, "min": 5, "max": 50, "label": "Steps"},
        "guidance_scale": {"type": "number", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5, "label": "Guidance scale"},
        "num_gaussians": {"type": "select", "options": [65536, 131072, 262144, 524288], "default": 262144, "label": "Gaussians", "description": "More splats = finer detail, larger file."},
        "erode_radius": {"type": "integer", "default": 1, "min": 0, "max": 5, "label": "Mask erode radius"},
    },
}

# Slots this plugin is the default implementation of: the node picker lists
# it first and a newly added node preselects it. Read statically by the
# scanner (never executed), so any SDK version imports this file fine.
TONGFLOW_DEFAULT_SLOTS = ["image-gen-model"]

# Weights are laid out on the volume mirroring the VAST-AI/TripoSplat HF repo.
CKPTS = "/models/triposplat"
CKPT_PATH = f"{CKPTS}/diffusion_models/triposplat_fp16.safetensors"
DECODER_PATH = f"{CKPTS}/vae/triposplat_vae_decoder_fp16.safetensors"
FLUX2_VAE_ENCODER_PATH = f"{CKPTS}/vae/flux2-vae.safetensors"
DINOV3_PATH = f"{CKPTS}/clip_vision/dino_v3_vit_h.safetensors"
RMBG_PATH = f"{CKPTS}/background_removal/birefnet.safetensors"

# Plugin-internal generation knobs. These are NOT ABI fields (the image-gen-model
# contract exposes only image/text/width/height/seed); TripoSplat-specific tuning
# stays here as constants, env-overridable. 262144 = max quality / slowest.
NUM_GAUSSIANS = int(os.environ.get("TRIPOSPLAT_NUM_GAUSSIANS", 262144))
STEPS = int(os.environ.get("TRIPOSPLAT_STEPS", 20))
GUIDANCE_SCALE = float(os.environ.get("TRIPOSPLAT_GUIDANCE_SCALE", 3.0))
SHIFT = float(os.environ.get("TRIPOSPLAT_SHIFT", 3.0))
ERODE_RADIUS = int(os.environ.get("TRIPOSPLAT_ERODE_RADIUS", 1))

_HERE = Path(__file__).resolve().parent

volume = modal.Volume.from_name("models", create_if_missing=True)

app = modal.App(_HERE.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("numpy", "safetensors", "pillow", "tqdm")
    .pip_install("tongflow==1.0.0", "fastapi[standard]")
    # PYTHONPATH so `from triposplat import TripoSplatPipeline` (which itself does
    # `from model import ...`) resolves against the vendored sources.
    .env({"HF_HOME": "/models/hf", "PYTHONPATH": "/opt/triposplat"})
    # Mounted at runtime (copy defaults to False) so every deploy ships the latest
    # vendored pipeline without baking a cacheable image layer.
    .add_local_file(str(_HERE / "triposplat.py"), "/opt/triposplat/triposplat.py")
    .add_local_file(str(_HERE / "model.py"), "/opt/triposplat/model.py")
)

with image.imports():
    import tempfile


@deploy
@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": volume},
    timeout=3600,
    scaledown_window=2,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Construct the TripoSplat pipeline once; reused across calls (warm)."""
        from triposplat import TripoSplatPipeline

        self.pipe = TripoSplatPipeline(
            ckpt_path=CKPT_PATH,
            decoder_path=DECODER_PATH,
            dinov3_path=DINOV3_PATH,
            flux2_vae_encoder_path=FLUX2_VAE_ENCODER_PATH,
            rmbg_path=RMBG_PATH,
            device="cuda",
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN_MODEL)
    def image_gen_model(self, input: ImageGenModelInput) -> ImageGenModelOutput:
        """One image -> one 3D Gaussian-splat (.splat) asset.

        text/width/height are part of the image-gen-model contract but TripoSplat
        is image-only, so they are ignored here.
        """
        try:
            img_b = prompt_media_to_bytes(input.image)
        except (TypeError, ValueError):
            return ImageGenModelOutput(success=False, error="Missing input image")

        seed = int(input.seed) if input.seed is not None else 42

        try:
            with tempfile.TemporaryDirectory() as td:
                src = os.path.join(td, "input.png")
                with open(src, "wb") as f:
                    f.write(img_b)
                gaussian, _prepared = self.pipe.run(
                    src,
                    seed=seed,
                    steps=_adv("steps", STEPS),
                    guidance_scale=_adv("guidance_scale", GUIDANCE_SCALE),
                    shift=SHIFT,
                    num_gaussians=_adv("num_gaussians", NUM_GAUSSIANS),
                    erode_radius=_adv("erode_radius", ERODE_RADIUS),
                )
                out = os.path.join(td, "output.splat")
                gaussian.save_splat(out)
                with open(out, "rb") as f:
                    data = f.read()
        except Exception as e:  # surfaced to the UI as an ABI failure
            return ImageGenModelOutput(success=False, error=str(e))

        return ImageGenModelOutput(
            success=True,
            model=asset(
                data, mime="application/octet-stream", filename="output.splat"
            ),
        )

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

