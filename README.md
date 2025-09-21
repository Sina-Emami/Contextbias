# Bias Detection Multimodal (CrewAI)

This project now focuses on a lean three-stage pipeline built with CrewAI:

1. **Description capture** – call a vision-enabled agent to collect exhaustive raw narratives for each image.
2. **Structured analysis** – convert those narratives into the `ImageAuditRecord` schema, aggregate repetition statistics, and run the bias reasoning crew.
3. **Summary reporting** – turn aggregated counts into a structured summary that can feed downstream analytics.

The code has been streamlined to keep only this core flow. Image generation, question drafting, fact checking, and multi-LLM consensus have been removed.

---

## Output Layout

Stage outputs are written under `data/scenarios/<SCENARIO_ID>/`:

```
data/scenarios/<ID>/
├─ manifest.json
├─ images/
│  └─ <image files you provide>
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

You are responsible for supplying the images and a matching `images_info.json` manifest (see Usage below).

---

## Requirements

* Python 3.10+
* `crewai`
* `openai`
* `pydantic` v2
* `python-dotenv`
* `replicate`
* Any additional libraries referenced in `requirements.txt` for analysis (sentence-transformers, hdbscan, etc.)

### Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file with at least the following keys:

```env
OPENAI_API_KEY=""          # required for the vision description tool
REPLICATE_API_KEY=""       # required for the bias reasoning agent
DESCRIBER_LLM=gpt-5-mini    # optional override
SUMMARY_LLM=gpt-4o-mini     # optional override for the ingest crew
BIAS_REPLICATE_MODEL=openai/gpt-oss-20b
BIAS_REPLICATE_TEMPERATURE=0.0
SUMMARY_REPORT_LLM=gpt-5-mini
VISION_CHAT_MODEL=gpt-5-mini
```

Remove keys that referenced the deprecated features (e.g., image generation, Serper, consensus).

---

## Usage

1. **Prepare a scenario folder** – `python -m src.app` will call `setup_scenario`, which creates the directory tree shown above.
2. **Populate images and metadata** – place your images in `images/` and write `images_info.json` alongside them. Each entry should look like:
   ```json
   {
     "id": "img_001",
     "filename": "img_001.png",
     "relpath": "images/img_001.png"
   }
   ```
   You can also include an `abspath` key if the files live elsewhere.
3. **Capture raw descriptions** – uncomment `capture_raw_descriptions(paths)` in the `__main__` block or call it manually. This uses the vision tool to write `raw_descriptions/raw_descriptions.json`.
4. **Structure the descriptions** – run `structure_descriptions(paths)` to create one `ImageAuditRecord` per image under `descriptions/`.
5. **Aggregate counts** – optional helper `summarize_description_counts(paths)` produces `descriptions/summary/counts.json`.
6. **Analyze bias** – call `run_analyze_bias(paths)` to build `biases/repeat_summary_full.json` and `biases/bias_report.json`.
7. **Generate the summary report** – `run_summary_report(paths)` reads the counts JSON (regenerating it if missing) and produces `descriptions/summary/summary_report.json`.

Run the full script directly if you prefer to step through interactively:

```bash
python -m src.app
```

Inside `src/app.py` the relevant helper calls are already listed; uncomment the stages you want to execute.

---

## Pipeline Details

- **Stage 1: Raw capture** – `build_raw_description_crew` invokes the `DescribeImageFromFile` tool for each entry in `images_info.json` and persists the verbatim response.
- **Stage 2: Structured schema + counts** – `build_structured_description_crew` converts the raw text into `ImageAuditRecord` JSON. `analysis.schema_counts` then clusters tokens and aggregates repetition signals across the dataset.
- **Stage 3: Bias reasoning and summary** – the ingest crew writes cumulative repetition state, a Replicate-hosted model converts that state into a `BiasReport`, and the summary reporter agent distills aggregated counts into a final structured report.

---

## Notes

- Ensure `images_info.json` always uses paths that resolve on the current machine. Relative paths are stored relative to the scenario root.
- The bias reasoning step depends on Replicate availability; set `REPLICATE_API_KEY` before running `run_analyze_bias`.
- The summary report agent expects valid JSON counts; if counts are missing it will trigger regeneration automatically.
