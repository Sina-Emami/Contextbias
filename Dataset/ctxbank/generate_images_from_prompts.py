import json
import os
from tqdm import tqdm
import random
import argparse

try:
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline
    import torch
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Missing dependencies - {e}")
    print("Please install: pip install torch torchvision diffusers transformers accelerate pillow")
    DEPENDENCIES_AVAILABLE = False

pipe = None

def setup_pipeline():
    """Initialize the Stable Diffusion pipeline with proper device and dtype detection."""
    global pipe
    
    if not DEPENDENCIES_AVAILABLE:
        print("Cannot setup pipeline - missing dependencies")
        return False
    
    try:
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            print(f"Using device: {device} (CUDA)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps" 
            torch_dtype = torch.float32  # float16 can cause issues on MPS
            print(f"Using device: {device} (Apple Silicon)")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print(f"Using device: {device}")
        
        print("Loading Stable Diffusion XL model... (this may take a while)")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        print("✓ Pipeline loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to setup pipeline: {e}")
        return False

def generate_image(prompt, seed, negatives=None, style=None):
    """
    Generates a single image using Stable Diffusion with error handling.
    """
    if not DEPENDENCIES_AVAILABLE or pipe is None:
        print(f"Cannot generate image - dependencies not available or pipeline not loaded")
        return None
    
    try:
        generator = torch.manual_seed(seed)
        full_prompt = prompt
        if style:
            full_prompt = f"{style} {full_prompt}"

        result = pipe(
            full_prompt,
            negative_prompt=negatives,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
            height=1024,
            width=1024
        )
        
        if result.images and len(result.images) > 0:
            return result.images[0]
        else:
            print(f"No image generated for prompt: {prompt[:50]}...")
            return None
            
    except Exception as e:
        print(f"Error generating image for prompt '{prompt[:50]}...': {e}")
        return None

def create_mock_image(width=1024, height=1024):
    """Create a mock image when dependencies are not available."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (width, height), color='lightgray')
        draw = ImageDraw.Draw(img)
        text = "Mock Image\n(Dependencies Missing)"
        draw.text((width//2-50, height//2-20), text, fill='black')
        return img
    except ImportError:
        return None

def main(prompts_path, output_dir, max_images, mock_mode=False):
    """Main function with improved error handling and validation."""
    
    if not os.path.exists(prompts_path):
        print(f"Error: Prompts file not found: {prompts_path}")
        return
    
    if not mock_mode:
        if not setup_pipeline():
            print("Failed to setup pipeline. Use --mock_mode for testing without dependencies.")
            return
    else:
        print("Running in mock mode - will create placeholder images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {prompts_path}")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return
    
    if not prompts:
        print("No prompts found in file")
        return
    
    first_prompt = prompts[0]
    required_fields = ['prompt', 'role', 'seeds']
    missing_fields = [field for field in required_fields if field not in first_prompt]
    if missing_fields:
        print(f"Error: Missing required fields in prompts: {missing_fields}")
        return
    
    total_images = 0
    successful_images = 0
    failed_images = 0
    
    for idx, entry in enumerate(tqdm(prompts, desc='Processing prompts')):
        try:
            prompt = entry['prompt']
            role = entry.get('role', 'unknown')
            negatives = entry.get('negatives', '')
            style = entry.get('style', '')
            seeds = entry.get('seeds', [])
            
            prompt_dir = os.path.join(output_dir, f"prompt_{idx:03d}_{role.replace(' ', '_')}")
            os.makedirs(prompt_dir, exist_ok=True)
            
            metadata = {
                'prompt': prompt,
                'role': role,
                'negatives': negatives,
                'style': style,
                'seeds': seeds,
                'prompt_index': idx
            }
            with open(os.path.join(prompt_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            for seed_idx, seed in enumerate(seeds):
                if max_images is not None and total_images >= max_images:
                    print(f"Reached max_images limit: {max_images}")
                    break
                
                if mock_mode:
                    image = create_mock_image()
                else:
                    image = generate_image(prompt, seed, negatives, style)
                
                if image is not None:
                    image_path = os.path.join(prompt_dir, f"image_{seed_idx:02d}_seed_{seed}.jpg")
                    try:
                        image.save(image_path, quality=95)
                        successful_images += 1
                    except Exception as e:
                        print(f"Error saving image: {e}")
                        failed_images += 1
                else:
                    failed_images += 1
                
                total_images += 1
            
            if max_images is not None and total_images >= max_images:
                break
                
        except Exception as e:
            print(f"Error processing prompt {idx}: {e}")
            failed_images += 1
    
    print(f"\nGeneration complete!")
    print(f"Total prompts processed: {idx + 1}")
    print(f"Successful images: {successful_images}")
    print(f"Failed images: {failed_images}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from prompts using Stable Diffusion XL.")
    parser.add_argument('--prompts', type=str, default='prompts_all.json', 
                       help='Path to prompts JSON file (default: prompts_all.json)')
    parser.add_argument('--output', type=str, default='generated_images', 
                       help='Directory to save generated images')
    parser.add_argument('--max_images', type=int, default=None, 
                       help='Maximum number of images to generate (useful for testing)')
    parser.add_argument('--mock_mode', action='store_true', 
                       help='Run in mock mode without generating real images (for testing)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stable Diffusion XL Image Generation")
    print("=" * 60)
    print(f"Prompts file: {args.prompts}")
    print(f"Output directory: {args.output}")
    print(f"Max images: {args.max_images or 'unlimited'}")
    print(f"Mock mode: {args.mock_mode}")
    print("=" * 60)
    
    main(args.prompts, args.output, args.max_images, args.mock_mode)
