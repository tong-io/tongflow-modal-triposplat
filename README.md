# tongflow-modal-triposplat

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Turn a single
image into a **3D Gaussian Splat** with **TripoSplat**
([VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat),
by [TripoAI](https://www.tripo3d.ai/)), running on a GPU via
[Modal](https://modal.com).

| Node slot         | Input | Output |
|-------------------|-------|--------|
| `image-gen-model` | image | 3D Gaussian splat (`.splat`) |

TripoSplat is a self-contained pipeline (`triposplat.py` + `model.py`, vendored
here and baked into the Modal image) with near-zero deps — no ComfyUI. Background
removal (BiRefNet) runs inside the pipeline, so any image works. The model node in
TongFlow renders the resulting `.splat` directly.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | ➖ | Optional — `VAST-AI/TripoSplat` is public (MIT), not gated. |

### Weights (Hugging Face)

`download.py` fetches the 5 weight files from
[`VAST-AI/TripoSplat`](https://huggingface.co/VAST-AI/TripoSplat) to the shared
`models` volume under `/models/triposplat/<subdir>` (diffusion model, VAE decoder,
Flux2 VAE encoder, DINOv3 ViT-H, BiRefNet). HF_TOKEN is injected from Settings —
no manual `modal secret create`.

## Usage

```bash
# One-time: fetch weights to the volume
modal run download.py::download

# Deploy the inference app (TongFlow does this automatically on first use)
modal deploy deploy.py
```

The platform invokes `entry.py` per task; it auto-deploys on first use and
re-deploys when `deploy.py` changes. The vendored `triposplat.py` / `model.py` are
mounted at deploy time, so updating them ships on the next deploy.

## Tuning knobs

TripoSplat-specific knobs are plugin constants in `deploy.py` (env-overridable),
**not** ABI fields:

- **`TRIPOSPLAT_NUM_GAUSSIANS`** (default `262144`): max quality / slowest. Lower
  it (down to `32768`, multiples of 32) to trade detail for speed and smaller
  output.
- **`TRIPOSPLAT_STEPS`** (default `20`): flow-matching sampler steps; 10–20
  recommended.
- **`TRIPOSPLAT_GUIDANCE_SCALE`** (default `3.0`) / **`TRIPOSPLAT_SHIFT`**
  (default `3.0`): CFG strength / timestep schedule shift.
- **`deploy.py` `gpu`** (default `A100-40GB`): try the cheaper `L40S` once a run
  succeeds; **`scaledown_window`** (default 2s) keeps the container warm between
  calls to skip cold-start model load (idle GPU is still billed).

## License & attribution

This plugin (deploy/entry/download glue) is part of TongFlow. The vendored
`triposplat.py` and `model.py` are from
[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) and
the model weights from [`VAST-AI/TripoSplat`](https://huggingface.co/VAST-AI/TripoSplat),
both released under the **MIT License**.
