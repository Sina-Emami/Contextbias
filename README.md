# Bias Detection Multimodal (CrewAI)

This project runs an asynchronous, resume-friendly pipeline that walks a dataset of prompt folders and generates structured bias analyses for every image. The workflow is driven by CrewAI agents and proceeds through three main stages:

1. **Raw description capture** – A vision-enabled agent calls `DescribeImageFromFile` for each image referenced in a prompt manifest.
2. **Structured description + counts** – Raw narratives are converted into the `ImageAuditRecord` schema and frequency statistics are aggregated.
3. **Bias reasoning + reporting** – Structured outputs feed the bias analysis crew and a summary reporter produces the final dataset roll-up.

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
├─ raw_descriptions/
│  └─ raw_descriptions.json
├─ descriptions/
│  ├─ <image_id>.json
│  └─ summary/
│     ├─ counts.json
│     └─ summary_report.json
└─ biases/
   ├─ agg_state.json
   ├─ repeat_summary_full.json
   └─ bias_report.json
```

If raw descriptions or structured JSON already exist, the resume logic honours them and only fills in missing work.

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
DESCRIBER_LLM=gpt-5-mini    # optional override
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

* `[Skip] <prompt>` – All expected artifacts exist (`raw_descriptions`, structured JSON, counts, summary report); nothing runs.
* `[Start] <prompt>` – No prior work detected; the pipeline executes all stages.
* `[Resume] <prompt>` – Some artifacts are missing; the pipeline resumes from the first missing image (or regenerates counts/summary).

Successful runs end with `Queued N prompt folder(s) for processing.` followed by per-stage progress for each folder. Subsequent reruns pick up from the last incomplete image or skip completed prompts entirely.

---

## Dataset Rollup & Visualization

After the per-prompt pipeline finishes, generate aggregated counts and figures across the entire dataset with:

```bash
python -m decription_pipeline.analysis.dataset_rollup
```

This utility walks every role/prompt, reads each `summary_report.json`, and emits:

- **Prompt aggregations** (`role/aggregation_counting/<prompt>_counts.json`) summarising all cohort/sub-key counts per prompt, along with `num_images`.
- **Prompt visuals** (`prompt/visualization_analysis/*.png`) including bar charts for each dimension and a `spatial_heatmap.png` when 3×3 spatial data exist. Filenames are hashed to stay within Windows path limits.
- **Role rollups** (`role/aggregation_counting/role_counts.json`) consolidating prompts under a role with `total_prompts`, `total_images`, combined counts, and prompt metadata.
- **Role heatmap** (`role/visualization_analysis/role_spatial_heatmap.png`) whenever any prompt contributes positional counts.

The script reuses the flattening helpers from `context_metrics.py`, so future aggregation tweaks automatically stay in sync. Rerunning the command refreshes existing outputs in place. Warnings about “tight layout not applied” are cosmetic and stem from very long labels.

---

## Pipeline Details

- **Stage 1: Raw capture** – `_capture_raw_descriptions_async` schedules up to `RAW_STAGE_CONCURRENCY` images in parallel (default 10) via `asyncio.to_thread`, storing incremental results in `raw_descriptions/raw_descriptions.json`.
- **Stage 2: Structured schema + counts** – `_structure_descriptions_async` mirrors the map-reduce pattern for structuring; the `analysis.schema_counts` module then aggregates frequency statistics.
- **Stage 3: Bias reasoning and summary** – `run_analyze_bias` ingests structured descriptions and produces bias artifacts; `run_summary_report` saves `summary_report.json`, rebuilding counts when needed.

### Concurrency & Resume Behaviour

- `PROMPT_STAGE_CONCURRENCY` (default 3) limits how many prompt folders run simultaneously, regardless of role.
- `RAW_STAGE_CONCURRENCY` and `STRUCT_STAGE_CONCURRENCY` (default 10 each) control image-level fan-out.
- Manifest discovery is recursive; any `manifest.json` under the dataset root is considered a prompt directory.
- Resume checkpoints are based on existing raw descriptions, structured files, and report artifacts; only missing work is redone.

---

## Notes

- Relative paths inside manifests are resolved against the dataset root; absolute paths are honoured as-is.
- Ensure `manifest.json` entries include at least an `id` and `filename` (or `relpath`).
- The bias reasoning stage requires Replicate access; set `REPLICATE_API_KEY` before running the pipeline.
- You can override concurrency via environment variables to match your hardware/LLM quota.
- Summary reporting is idempotent; if counts are missing they are regenerated automatically.
