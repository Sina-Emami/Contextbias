from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from diffusers import StableDiffusionXLPipeline
    import torch

    HAS_SDXL = True
except ImportError as exc:
    print(f"SDXL dependencies unavailable ({exc}); only --mock_mode will work.")
    print("Install with: pip install torch torchvision diffusers transformers accelerate pillow")
    HAS_SDXL = False

REQUIRED_PROMPT_FIELDS = ("prompt", "role", "seeds")


def _pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # float16 on MPS is unreliable; fall back to float32 there.
        return "mps", torch.float32
    return "cpu", torch.float32


class SDXLGenerator:
    def __init__(self):
        device, dtype = _pick_device_and_dtype()
        print(f"Loading Stable Diffusion XL on {device}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(device)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

    def generate(self, prompt: str, seed: int, negatives: str = "", style: str = "") -> Image.Image | None:
        full_prompt = f"{style} {prompt}".strip() if style else prompt
        try:
            result = self.pipe(
                full_prompt,
                negative_prompt=negatives or None,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=torch.manual_seed(seed),
                height=1024,
                width=1024,
            )
        except Exception as exc:
            print(f"Generation failed for '{prompt[:50]}...': {exc}")
            return None
        return result.images[0] if result.images else None


def mock_image(width: int = 1024, height: int = 1024) -> "Image.Image | None":
    if not HAS_PIL:
        return None
    img = Image.new("RGB", (width, height), color="lightgray")
    ImageDraw.Draw(img).text((width // 2 - 60, height // 2 - 10), "mock image", fill="black")
    return img


def load_prompts(path: str) -> list[dict]:
    with open(path) as f:
        prompts = json.load(f)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    missing = [field for field in REQUIRED_PROMPT_FIELDS if field not in prompts[0]]
    if missing:
        raise ValueError(f"Prompts file is missing required fields: {missing}")
    return prompts


def run(prompts_path: str, output_dir: str, max_images: int | None, mock_mode: bool) -> None:
    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    generator = None
    if not mock_mode:
        if not HAS_SDXL:
            raise RuntimeError("SDXL dependencies are missing; re-run with --mock_mode to test without them.")
        generator = SDXLGenerator()

    os.makedirs(output_dir, exist_ok=True)

    total = succeeded = failed = 0
    for idx, entry in enumerate(tqdm(prompts, desc="prompts")):
        if max_images is not None and total >= max_images:
            break

        prompt = entry["prompt"]
        role = entry.get("role", "unknown")
        negatives = entry.get("negatives", "")
        style = entry.get("style", "")
        seeds = entry.get("seeds", [])

        prompt_dir = os.path.join(output_dir, f"prompt_{idx:03d}_{role.replace(' ', '_')}")
        os.makedirs(prompt_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, "metadata.json"), "w") as f:
            json.dump({**entry, "prompt_index": idx}, f, indent=2)

        for seed_idx, seed in enumerate(seeds):
            if max_images is not None and total >= max_images:
                break
            total += 1

            image = mock_image() if mock_mode else generator.generate(prompt, seed, negatives, style)
            if image is None:
                failed += 1
                continue
            try:
                image.save(os.path.join(prompt_dir, f"image_{seed_idx:02d}_seed_{seed}.jpg"), quality=95)
                succeeded += 1
            except Exception as exc:
                print(f"Failed to save image: {exc}")
                failed += 1

    print(f"\nDone: {succeeded} succeeded, {failed} failed, output in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from prompts using Stable Diffusion XL.")
    parser.add_argument("--prompts", default="prompts_all.json", help="Path to prompts JSON file")
    parser.add_argument("--output", default="generated_images", help="Directory to save generated images")
    parser.add_argument("--max_images", type=int, default=None, help="Cap on total images generated")
    parser.add_argument("--mock_mode", action="store_true", help="Write placeholder images instead of calling SDXL")
    args = parser.parse_args()
    run(args.prompts, args.output, args.max_images, args.mock_mode)
