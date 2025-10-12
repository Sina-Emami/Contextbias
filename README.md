# Bias Detection Multimodal (CrewAI)

This project runs an asynchronous, resume-friendly pipeline that walks a dataset of prompt folders and generates structured bias analyses for every image. The workflow is driven by CrewAI agents and proceeds through three main stages:

1. **Schema-ready description capture** - The `Image Schema Describer` crew invokes the `DescribeImageFromFile` tool for every manifest entry and writes one `descriptions/<image_id>.json` `ImageAuditRecord` per image.
2. **Frequency counting & cleaning** - Run `python -m decription_pipeline.data_processing.pipeline` to count prompt attributes, normalise tokens, aggregate by role, and publish dataset-level roll-ups.
3. **Bias reasoning + reporting (optional)** - Once structured descriptions and frequency artifacts exist, downstream crews can generate bias analyses and narrative reports.

Each prompt directory is processed independently. The runner automatically skips folders that already contain all expected artifacts, resumes partially processed folders from the next missing image, and parallelises work to keep image agents busy.

---

## Dataset Layout

The pipeline expects dataset prompts under `dataset/` (configurable). A typical structure looks like:

```
dataset/
├─ Context-aware_Related_CA-R/
│  ├─ doctor/
│  │  ├─ a_photo_of_a_doctor...
│  │  │  ├─ image_0_seed_....jpg
│  │  │  ├─ ...
│  │  │  └─ manifest.json
│  │  └─ ...
│  └─ nurse/
│     └─ ...
└─ Context-free_CF/
   └─ ...
```

Each prompt directory must include a `manifest.json` (an array of image metadata). During processing the runner adds:

```
<prompt_dir>/
|- images_info.json
|- descriptions/
|  - <image_id>.json
|- frequency/
|  - frequencies.json
|  - frequencies.csv
- clean_frequency/
   - frequencies.json
   - frequencies.csv
```

The frequency pipeline also writes aggregated files under `dataset/role_counting/` (per role) and `dataset/role_counting/all_roles.*` (global). Resume logic honours existing structured JSON and only fills in missing work. All frequency CSV outputs share the columns `cohort`, `dimension`, `label`, `count`, and `bin` (the number of unique labels observed for that dimension).

## Agents, Tasks & Tools

- `decription_pipeline/crew.py` assembles a single CrewAI `Crew` composed of the Image Schema Describer agent and the describe-images task.
- The agent is defined in `decription_pipeline/agents/describer.py`. It calls the `DescribeImageFromFile` tool (`tools/vision_description_tool.py`) and reformats the evidence into an `ImageAuditRecord`.
- `decription_pipeline/tasks/describe_images.py` supplies the task prompt. It requires exactly one tool invocation, enforces enum tokens, and demands a single JSON object as output.
- `decription_pipeline/crew.build_image_description_crew()` wires the agent and task together with `Process.sequential` and `force_tool_output=True`, ensuring the saved JSON mirrors the tool observations.

---

## Requirements

* Python 3.10+
* `crewai`
* `openai`
* `pydantic` v2
* `python-dotenv`
* `replicate`
* Additional libraries listed in `requirements.txt` (e.g., `sentence-transformers`, `hdbscan`)

### Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` with at least:

```env
OPENAI_API_KEY=""          # required for the vision description tool
REPLICATE_API_KEY=""       # required for the bias reasoning agent
DESCRIBER_LLM=gpt-5-nano     # optional fallback if you re-enable LLM responses for the describer
SUMMARY_LLM=gpt-4o-mini     # optional override for summary crew
BIAS_REPLICATE_MODEL=openai/gpt-oss-20b
BIAS_REPLICATE_TEMPERATURE=0.0
SUMMARY_REPORT_LLM=gpt-5-mini
VISION_CHAT_MODEL=gpt-5-mini
PROMPT_STAGE_CONCURRENCY=3   # optional overrides for concurrency
RAW_STAGE_CONCURRENCY=10
STRUCT_STAGE_CONCURRENCY=10
```

Remove legacy keys from earlier experiments (image generation, consensus, etc.).

---

## Usage

Run the orchestrator to process any number of prompt folders:

```bash
python decription_pipeline/app.py
```

### Log Messages

* `[Skip] <prompt>` – All expected artifacts exist (structured JSON, frequency outputs, summary report); nothing runs.
* `[Start] <prompt>` – No prior work detected; the pipeline executes all stages.
* `[Resume] <prompt>` – Some artifacts are missing; the runner resumes from the first missing image and stops once it reaches an existing structured JSON (counts/summary regenerate when requested).

Successful runs end with `Queued N prompt folder(s) for processing.` followed by per-stage progress for each folder. Subsequent reruns pick up from the last incomplete image or skip completed prompts entirely.

---

## Dataset Rollup & Visualization

After the per-prompt pipeline finishes, generate aggregated counts and figures across the entire dataset with:

```bash
python -m decription_pipeline.data_processing.dataset_rollup
```

To run the entire frequency-processing flow end-to-end (count, clean, role roll-up, global roll-up):

```bash
python -m decription_pipeline.data_processing.pipeline
```

This utility walks every role/prompt, reads each `summary_report.json`, and emits:

- **Prompt aggregations** (`role/aggregation_counting/<prompt>_counts.json`) summarising all cohort/sub-key counts per prompt, along with `num_images`.
- **Prompt visuals** (`prompt/visualization_analysis/*.png`) including bar charts for each dimension and a `spatial_heatmap.png` when 3×3 spatial data exist. Filenames are hashed to stay within Windows path limits.
- **Role rollups** (`role/aggregation_counting/role_counts.json`) consolidating prompts under a role with `total_prompts`, `total_images`, combined counts, and prompt metadata.
- **Role heatmap** (`role/visualization_analysis/role_spatial_heatmap.png`) whenever any prompt contributes positional counts.

The script reuses the flattening helpers from `context_metrics.py`, so future aggregation tweaks automatically stay in sync. Rerunning the command refreshes existing outputs in place. Warnings about “tight layout not applied” are cosmetic and stem from very long labels.

---

## Pipeline Details

- **Description capture** - `process_prompt_directory()` prepares per-prompt folders, then calls `describe_images()`, which delegates to `_describe_images_async` and the image-description crew. Each run produces one `ImageAuditRecord` JSON per image under `descriptions/`.
- **Frequency counting & aggregation** - `decription_pipeline.data_processing.pipeline.run_pipeline()` orchestrates frequency counting, cleaning, per-role aggregation, and dataset roll-ups. It combines `frequency_counter.compute_frequencies`, `frequency_cleaner.clean_dataset`, `role_frequency_aggregator.aggregate_roles`, and `global_frequency_aggregator.aggregate_all_roles`.
- **Optional reporting** - Once structured descriptions and frequency artifacts exist, additional crews (bias analysis, narrative summaries, visualisations) can be executed as needed.

### Concurrency & Resume Behaviour

- `ROLE_CONCURRENCY` (default 2) bounds how many role directories are processed in parallel.
- `IMAGE_CONCURRENCY` (default 10) limits concurrent `DescribeImageFromFile` tool invocations inside a prompt.
- Manifest discovery is recursive; any `manifest.json` under the dataset root is treated as a prompt directory.
- Resume logic seeks the first missing structured description, restarts from that image, and halts as soon as the next existing JSON is encountered so downstream files remain untouched.

---

## Notes

- Relative paths inside manifests are resolved against the dataset root; absolute paths are honoured as-is.
- Ensure `manifest.json` entries include at least an `id` and `filename` (or `relpath`).
- The bias reasoning stage requires Replicate access; set `REPLICATE_API_KEY` before running the pipeline.
- You can override concurrency via environment variables to match your hardware/LLM quota.
- Summary reporting is idempotent; if counts are missing they are regenerated automatically.

---

## Aggregation Outputs (Standardised)

This repo standardises where aggregated files are written so downstream analysis is predictable:

- Per-prompt (inside each prompt folder)
  - ggregation_counting/<prompt>_counts.json - normalised counts for that single prompt.

- Per-role, per-context (inside each role folder)
  - ggregation_counting/role_counts.csv - combined counts for all prompts under that role in the current context.
  - ggregation_counting/role_counts.json - same data plus metadata (	otal_prompts, 	otal_images, prompt list).

- Per-role, all contexts (at dataset root)
  - dataset/role_count_aggregation/<role>.csv - merged counts for a role across Context-free and both Context-aware folders. Build with:

    `ash
    python -m decription_pipeline.data_processing.role_rollup
    `

- General attributes (optional)
  - dataset/general_attributes_rollup.csv - one file summarising high-level attributes (mood, lighting, camera, demographics, presence flags) aggregated over all roles. Build with:

    `ash
    python -m decription_pipeline.data_processing.role_rollup --general
    `

### Folder Layout Reference

`
dataset/
  Context-free_CF/ | Context-aware_Related_CA-R/ | Context-aware_Unrelated_CA-U/
    <role>/
      <prompt>/
        descriptions/summary/summary_report.json
        aggregation_counting/<prompt>_counts.json
      aggregation_counting/
        role_counts.csv
        role_counts.json
  role_count_aggregation/
    <role>.csv
  general_attributes_rollup.csv         # optional, if built with --general
`

All aggregations apply the same normalisation rules for keys and labels (e.g., clothing garment variants, gender presentation, presence flags, spatial plane/side), ensuring counts align across prompts, roles, and contexts.
