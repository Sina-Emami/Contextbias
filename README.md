# Bias Detection in Multimodal Image Generation

A research pipeline that detects and quantifies demographic and visual bias in AI-generated images. The system generates structured descriptions of images using a vision LLM, aggregates attribute frequencies across roles and contexts, runs statistical bias tests, and supports a human-study evaluation workflow.

---

## Overview

The project is split into four main areas:

| Area | Directory | Purpose |
|---|---|---|
| Description pipeline | `decription_pipeline/` | Generate structured JSON descriptions for every image via a vision LLM |
| Bias analysis | `analysis_visualization/` | Chi-square tests and p-value heatmaps across roles and attributes |
---

## Dataset Layout

Images are organised by context type, role, and prompt:

```
dataset/
├── Context-free_CF/
│   └── <role>/
│       └── <prompt_folder>/
│           ├── manifest.json          # list of image metadata entries
│           ├── descriptions/          # one JSON per image (output)
│           └── clean_frequency/       # normalised frequency counts (output)
├── Context-aware_Related_CA-R/
│   └── <role>/
│       └── <prompt_folder>/
└── Context-aware_Unrelated_CA-U/
    └── <role>/
        └── <prompt_folder>/
```

Each `manifest.json` is an array of objects with at least `id` and `filename` (or `relpath`). The pipeline creates `descriptions/`, `frequency/`, and `clean_frequency/` under each prompt folder, plus role-level and dataset-level rollup files under `dataset/role_counting/`.

---

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Environment variables** — create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here          # required for vision descriptions
REPLICATE_API_KEY=your_key_here       # required for Replicate-hosted LLMs
VISION_CHAT_MODEL=gpt-4o-mini         # OpenAI model used for image descriptions
DESCRIBER_LLM=gpt-4o-mini             # optional: LLM for the describer agent
BIAS_REPLICATE_TEMPERATURE=0.0
```

---

## 1. Description Pipeline

The pipeline walks the dataset, sends each image to a vision LLM, and saves a validated `ImageAuditRecord` JSON per image.

### Run

```bash
python decription_pipeline/app.py
```

The runner will print one of:
- `[Skip] <prompt>` — all artifacts already exist, nothing to do
- `[Start] <prompt>` — no prior work, full run
- `[Resume] <prompt>` — partial prior work, resumes from the first missing image

### How it works

1. **`app.py`** — entry point; discovers all `manifest.json` files and queues prompt folders.
2. **`crew.py`** — assembles a CrewAI `Crew` with the describer agent and describe-images task.
3. **`agents/describer.py`** — defines the `Image Schema Describer` agent.
4. **`tools/vision_description_tool.py`** — calls the OpenAI vision API, validates the response against `ImageAuditRecord` (Pydantic v2, `extra="forbid"`), and returns the JSON.
5. **`schemas/description.py`** — the full schema: all `Literal` enums (mood, skin tone, gender, clothing, etc.) and the `ImageAuditRecord` top-level model.

**Concurrency** (configurable via env vars):
- `ROLE_CONCURRENCY` (default 2) — parallel role directories
- `IMAGE_CONCURRENCY` (default 10) — parallel image requests within a prompt

### Frequency counting

After descriptions are generated, compute attribute frequencies:

```bash
python -m decription_pipeline.data_processing.pipeline
```

This runs the full chain:
1. `frequency_counter.py` — counts each label per cohort/dimension for a prompt
2. `frequency_cleaner.py` — normalises free-text tokens using semantic similarity (sentence-transformers + HDBSCAN clustering)
3. `role_frequency_aggregator.py` — merges prompts under each role
4. `global_frequency_aggregator.py` — global rollup across all roles

Outputs follow the schema: `cohort`, `dimension`, `label`, `count`, `bin`.

For role-level aggregation across all context types:

```bash
python -m decription_pipeline.data_processing.role_rollup
# add --general for a high-level attributes summary
python -m decription_pipeline.data_processing.role_rollup --general
```

---

## 2. Bias Analysis & Visualisation

Statistical tests over the frequency data to detect bias across roles and image contexts.

### Run

```bash
# Chi-square bias statistics
python analysis_visualization/bias_analysis_data.py

# P-value heatmaps
python analysis_visualization/bias_pvalue_heatmaps.py
```

**`bias_analysis_data.py`** loads `clean_frequency` CSVs, computes chi-square tests for each cohort/dimension pair across roles, and produces bias statistics with p-values.

**`bias_pvalue_heatmaps.py`** renders colour-coded heatmaps (one per cohort) showing which role × dimension combinations have statistically significant bias (p < 0.05). Outputs are saved as `.png` files.

---

## Aggregation Output Reference

| Path | Content |
|---|---|
| `<prompt>/descriptions/<id>.json` | One `ImageAuditRecord` per image |
| `<prompt>/frequency/frequencies.csv` | Raw label counts |
| `<prompt>/clean_frequency/frequencies.csv` | Normalised label counts |
| `<role>/aggregation_counting/role_counts.csv` | Per-role counts across prompts |
| `dataset/role_count_aggregation/<role>.csv` | Per-role counts across all contexts |
| `dataset/general_attributes_rollup.csv` | High-level attribute summary |

---

## Project Structure

```
decription_pipeline/
├── app.py                        # entry point
├── crew.py                       # CrewAI crew assembly
├── agents/describer.py           # describer agent
├── tasks/describe_images.py      # CrewAI task definition
├── tools/vision_description_tool.py  # OpenAI vision call + validation
├── schemas/description.py        # ImageAuditRecord schema
├── llm/replicate_llm.py          # Replicate LLM wrapper
├── data_processing/              # frequency counting & aggregation
│   ├── pipeline.py
│   ├── frequency_counter.py
│   ├── frequency_cleaner.py
│   ├── frequency_schema.py
│   ├── role_frequency_aggregator.py
│   ├── global_frequency_aggregator.py
│   └── semantic_utils.py
├── distribution_visualization/   # schema distribution plots
└── utils/
    ├── fs.py                     # directory initialisation helpers
    ├── add_structure_json.py     # manifest JSON builder
    └── export_artifacts.py

analysis_visualization/
├── bias_analysis_data.py         # chi-square bias statistics
└── bias_pvalue_heatmaps.py       # p-value heatmap figures

```
