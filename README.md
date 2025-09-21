
## Image generation pipeline

The pipeline consists of the following main Python scripts (see `image_generation/ctxbank/`):

### 1. Candidate Generation
**File:** `generate_candidates.py`

**Description:**
Generates candidate action-location pairs for each role, split into `related` and `unrelated` lists.

**Parameters:**
- `--roles <roles.json>`: Path to JSON file containing list of roles
- `--out <candidates.json>`: Output file (default: `candidates.json`)

**Example:**
```bash
python image_generation/ctxbank/generate_candidates.py --roles roles.json --out candidates.json
```

### 2. Judge and Filter Candidates
**File:** `judge_and_filter.py`

**Description:**
Scores and filters candidate action-location pairs for each role, using LLM-based scoring and deduplication.

**Parameters:**
- `--infile <candidates.json>`: Input file from previous step
- `--out <context_bank.json>`: Output file (default: `context_bank.json`)
- `--model <model_name>`: (Optional) Specify LLM model
- `--filter`: Enable filtering by thresholds

**Example:**
```bash
python image_generation/ctxbank/judge_and_filter.py --infile candidates.json --out context_bank.json --filter
```

### 3. Combine Prompts
**File:** `combine_prompts.py`

**Description:**
Generates prompts for image generation from the filtered context bank, using multiple templates and deduplicating prompts by string.

**Parameters:**
(No parameters; reads `context_bank.json` and writes `prompts_combined.json`)

**Example:**
```bash
python image_generation/ctxbank/combine_prompts.py
```

### 4. Image Generation
**File:** `generate_images_from_prompts.py`

**Description:**
Generates images using Stable Diffusion XL from the deduplicated prompts.

**Parameters:**
- `--prompts <prompts_combined.json>`: Path to prompts file (default: `prompts_combined.json`)
- `--output <output_dir>`: Directory to save images (default: `generated_images`)
- `--max_images <N>`: Maximum number of images to generate (optional)

**Example:**
```bash
python image_generation/ctxbank/generate_images_from_prompts.py --prompts prompts_combined.json --output generated_images --max_images 100
```

---

## How the Pipeline Works

1. **Candidate Generation**
   - For each role, generate lists of `related` and `unrelated` action-location pairs using LLM.
   - Output: `candidates.json`

2. **Judging and Filtering**
   - Score each candidate pair for relevance, neutrality, and confound using LLM.
   - Filter and deduplicate pairs based on thresholds.
   - Output: `context_bank.json`

3. **Prompt Combination**
   - For each role, generate prompts using three templates:
     - T0: "a photo of a {ROLE}"
     - T1: "a photo of a {ROLE} in a {LOCATION}"
     - T2: "a photo of a {ROLE} {ACTION} in a {LOCATION}"
   - Prompts are deduplicated by string.
   - Output: `prompts_combined.json`

4. **Image Generation**
   - For each prompt, generate images using Stable Diffusion XL.
   - Images are saved in subfolders per prompt.
   - Output: `generated_images/`

---

## Example End-to-End Run

```bash
# 1. Generate candidates
python image_generation/ctxbank/generate_candidates.py --roles roles.json --out candidates.json

# 2. Judge and filter candidates
python image_generation/ctxbank/judge_and_filter.py --infile candidates.json --out context_bank.json --filter

# 3. Combine prompts
python image_generation/ctxbank/combine_prompts.py

# 4. Generate images
python image_generation/ctxbank/generate_images_from_prompts.py --prompts prompts_combined.json --output generated_images --max_images 100
```

---

## Ground Truth Biased Prompts

### 5. Ground Truth Prompt Generation
**File:** `generate_ground_truth_prompts.py`

**Description:**
Generates image generation prompts that explicitly encode social, demographic, or occupational biases for each role. These prompts serve as ground truth for testing bias detection on generated images.

**Parameters:**
- `--roles <roles.json>`: Path to JSON file containing list of roles
- `--out <ground_truth_prompts.json>`: Output file (default: `ground_truth_prompts.json`)
- `--num <N>`: Number of biased prompts per role (default: 5)

**Example:**
```bash
python -m image_generation.ctxbank.generate_ground_truth_prompts --roles roles.json --out ground_truth_prompts.json --num 3
```

**Output:**
- `ground_truth_prompts.json`: Dictionary mapping each role to a list of biased prompts

---

## Output Files

- `candidates.json`: Action-location pairs for each role
- `context_bank.json`: Filtered and scored pairs
- `prompts_combined.json`: Deduplicated prompts for image generation
- `ground_truth_prompts.json`: Ground truth biased prompts for each role
- `generated_images/`: Folders containing generated images

---

## Notes

- All prompts are deduplicated before image generation to avoid redundant outputs.
- Each image is generated with multiple random seeds for diversity.
- The pipeline is modular; you can run each step independently and inspect outputs before proceeding.
