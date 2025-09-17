import json
import os
from tqdm import tqdm
from PIL import Image
import random
import argparse

# Hugging Face Diffusers imports
from diffusers import StableDiffusionXLPipeline
import torch

# Load the pipeline once globally
device = "mps" if torch.has_mps else "cpu"
print(f"Using device: {device}")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device=="mps" else torch.float32
)
pipe = pipe.to(device)

def generate_image(prompt, seed, negatives=None, style=None):
    """
    Generates a single image using Stable Diffusion.
    """
    generator = torch.manual_seed(seed)
    full_prompt = prompt
    if style:
        full_prompt = f"{style} {full_prompt}"

    # Generate image
    result = pipe(
        full_prompt,
        negative_prompt=negatives,
        guidance_scale=7.5,
        num_inference_steps=50,
        generator=generator
    )
    image = result.images[0]
    return image

def main(prompts_path, output_dir, max_images):
    os.makedirs(output_dir, exist_ok=True)
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    total_images = 0
    for idx, entry in enumerate(tqdm(prompts, desc='Generating images')):
        prompt = entry['prompt']
        role = entry.get('role', '')
        negatives = entry.get('negatives', '')
        style = entry.get('style', '')
        seeds = entry.get('seeds', [])
        # Create a directory for each prompt
        prompt_dir = os.path.join(output_dir, f"prompt_{idx}")
        os.makedirs(prompt_dir, exist_ok=True)
        for seed_idx, seed in enumerate(seeds):
            if max_images is not None and total_images >= max_images:
                print(f"Reached max_images limit: {max_images}")
                return
            image = generate_image(prompt, seed, negatives, style)
            image_path = os.path.join(prompt_dir, f"image_{seed_idx}_seed_{seed}.jpg")
            image.save(image_path)
            total_images += 1
    print(f"Created {total_images} images in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from prompts_combined.json using Stable Diffusion.")
    parser.add_argument('--prompts', type=str, default='prompts_combined.json', help='Path to prompts_combined.json')
    parser.add_argument('--output', type=str, default='generated_images', help='Directory to save generated images')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to generate (across all prompts)')
    args = parser.parse_args()
    main(args.prompts, args.output, args.max_images)
