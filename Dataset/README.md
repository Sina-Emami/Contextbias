# Image Generation Pipeline

Scripts for constructing ContextBench's prompts and generating images from them. The design
follows the three conditions described in the paper (CF, CA-R, CA-U), built from an LLM-generated
context bank of role-related and role-unrelated action/location pairs.

Dataset: https://huggingface.co/datasets/shaghayegh/ContextBias

## Pipeline

1. **`generate_candidates.py`** — for each role, ask an LLM for candidate action/location pairs
   in two buckets: `related` (typical for the role) and `unrelated` (plausible but orthogonal).
   Output: `candidates.json`.
2. **`judge_and_filter.py`** — score each candidate for relevance, neutrality, and confound risk;
   apply thresholds and near-duplicate filtering. Output: `context_bank.json`.
3. **`combine_prompts.py`** — expand the context bank into prompts using three templates: T0
   (`"a photo of a {role}"`), T1 (`"... in a {location}"`), T2 (`"... {action} in a {location}"`);
   dedup by prompt string. Output: `prompts_combined.json`.
4. **`generate_images_from_prompts.py`** — generate images with Stable Diffusion XL from the
   deduplicated prompts, 10 seeds per prompt by default. Output: `generated_images/`.

`generate_ground_truth_prompts.py` is a separate utility: it produces explicit-attribute prompts
(e.g. "a photo of a male bartender") used to validate the attribute-extraction pipeline against
known ground truth (paper §4.2), not part of the CF/CA-R/CA-U pipeline above.

## Usage

```bash
python -m Dataset.ctxbank.generate_candidates --roles roles.json --out candidates.json
python -m Dataset.ctxbank.judge_and_filter --infile candidates.json --out context_bank.json --filter
python -m Dataset.ctxbank.combine_prompts --context_bank context_bank.json --out prompts_combined.json
python -m Dataset.ctxbank.generate_images_from_prompts --prompts prompts_combined.json --output generated_images --max_images 100
```

Every script accepts `--mock` (or `--mock_mode` for image generation) to run without calling an
LLM or Stable Diffusion, for testing the pipeline shape end-to-end.

## Notes

- LLM calls go through `llm_client.py`, which reads `OPENAI_API_KEY` (and optional
  `OPENAI_MODEL`, default `gpt-4o-mini`) from `.env`.
- Prompts include a fixed negative prompt and style string to reduce noise; see
  `combine_prompts.py` for the exact values.

## Data files in `ctxbank/`

- `roles.json` and `prompts_alll.json` have been replaced with the verified reconstruction
  from `datasets/contextbench/` (92 canonical roles from the paper, 1,659 real prompts with
  recovered per-model seeds — see that folder's README for full provenance).
- `candidates_all.json` and `context_bank_all.json` are **left as-is from an earlier
  exploratory run and are stale/unverified** — they cover a different, larger role set (104
  roles) than the paper's 92, and their `rationale` text and `score` (relevance/neutrality/
  confound) fields are original LLM-judgment output that cannot be regenerated from static
  data. Treat these two files as historical artifacts, not ground truth; re-run
  `generate_candidates.py` / `judge_and_filter.py` against the corrected `roles.json` if you
  need current versions.
