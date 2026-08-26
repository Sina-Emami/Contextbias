# ContextBench prompts (reconstructed)

The original prompt-bank files were lost. These were rebuilt directly from surviving
raw-data artifacts on disk — every entry here is a real prompt that was actually used for
image generation, with real random seeds — not a guess or an LLM regeneration.

## Result

**1,659 prompts across all 92 canonical roles** (paper states 1,656 — the small excess is
natural per-role variance, see below), every entry with real per-model random seeds.
Per-role totals range 17–21 (paper's implied average is 18/role: 1 CF base + 2 CF
paraphrases + CA-R/CA-U base + paraphrase + substitution variants).

- **`prompt`**: exact prompt text, recovered from raw-dataset folder names (folder names are
  the prompt with spaces replaced by underscores).
- **`context`**: `CF` / `CA-R` / `CA-U`.
- **`seeds_by_model`**: the random seeds actually used per model (SDXL, SD3.5, FLUX.1,
  Qwen-Image), recovered directly from generated-image filenames (e.g.
  `image_06_seed_615284.jpg` / `.json` → seed `615284`). Every one of the 1,659 entries has
  recovered seeds.

## Sources, in the order they were used

1. `92_job_description_AllModels/<model>/` — the primary per-model dataset tree.
2. `92_job_description_AllModels_extracted_new/` — a second data drop (location
   substitution + paraphrase variants), unioned in per model.
3. Seeds for (1) and (2) come from each prompt folder's `descriptions/*.json` filenames.

Together, (1)+(2) alone reconstruct 1,639 prompts (99.0%) at exactly-or-near 18/role for 90
of the 92 roles. Two roles — `customer_service_representative` and `marketing_specialist` —
were badly short (5/18 and 16/18) because almost no folders for them survive in (1) or (2).

4. For those **two roles only**, additional prompts and seeds were pulled from
   `server/{SDXL_NEWPROMPT,SD3.5_NEWPRPMPTS,pipeline_flux_New,Qwen_NewPromptsAll}/roles/`
   — a broader, pre-dataset-curation prompt-exploration source. This source was **not** used
   for the other 90 roles: cross-checking it against roles with clean, complete data showed
   it's a superset of what actually shipped (more variants than the final curated benchmark),
   so applying it everywhere would have overcounted. It was only applied here because these
   two roles' real folder-based data was almost entirely absent, so filling the gap from this
   source is a net improvement rather than compounding an already-correct count. This is why
   these two roles land slightly above 18 (21 and 20 respectively) rather than exactly 18.

Also checked and found NOT to contain additional prompts beyond what's already included, or
found to be a different prompt family entirely: `generated_images_second_run/`,
`Backup/generated_images_stable3_new/`, `Backup/pipeline_final(XL)/`,
`MultimodalBiasDetection/.../generated_images/` and `.../Evaluation/`. The `Backup/pipeline_final*`
folders in particular are a **different prompt set** (explicit-attribute validation prompts
like "wearing a headset" / "with a monitor" — matching the paper's separate 220-prompt
quantitative-validation set, Table 1), not ContextBench, and were deliberately excluded here.

## Files

- `roles.json` — the 92 canonical role names (underscore form, e.g. `web_developer`)
- `prompts.json` — every reconstructed ContextBench prompt record with per-model seeds
  (1,659 entries)
- `counts_per_role.json` — CF / CA-R / CA-U / total prompt counts per role
- `validation_prompts.json` — a **separate, different prompt set**: the explicit-attribute
  quantitative-validation prompts from §4.2/Table 1 of the paper (e.g. "a photo of an
  accountant sitting"), reconstructed from `Backup/pipeline_final(XL)/`. These are NOT part
  of ContextBench's CF/CA-R/CA-U structure; they exist purely to validate the attribute
  extraction pipeline against explicit ground truth.

  **Trimmed to 220 to match the paper (Table 1: 220 prompts × 10 seeds = 2,200 images/model).**
  The raw recovered batch had 263 prompts; trimmed to the first 220 by the source data's own
  `index` field (0–219), recorded per entry as `source_index`. This is a disclosed selection
  rule, not a claim that these specific 220 are provably the paper's exact final set — no
  marker in the data distinguishes the "real" 220 from the other 43.

  **Seeds: only 5 of the expected 10 per prompt survive.** Every entry's raw metadata has
  exactly 5 seeds (`pipeline_final` and `pipeline_finalXL` share identical values — same
  underlying batch, not two independent seed sets). The other 5 seeds per prompt are
  **not recoverable and are not invented here** — each entry has `n_seeds_recovered: 5`,
  `n_seeds_expected: 10`, `n_seeds_missing: 5`, and `unique_seeds` listing exactly the 5 real
  values found, so downstream use can't mistake this for a complete 10-seed record.
