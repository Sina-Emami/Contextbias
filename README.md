# Bias Detection Multimodal (CrewAI)

A step-by-step, **multi-agent** pipeline built with **CrewAI** that:

1. **Generates images** from a scenario
2. **Describes each image** in a structured JSON schema
3. **Aggregates descriptions** to detect **potential biases** (repetition, roles, attributes, symbols, etc.)
4. **Writes globally fact-checkable questions** from the bias report (uses **Replicate OSS-20B**)
5. **Fact-checks** those questions on the web (Serper + scraping, **gpt-5-mini**)
6. Gets a **5-LLM consensus** (via Replicate) on “what must be in the image”
7. (Planned) **Filters & scores** biases using fact-check + consensus

The pipeline is designed to run **incrementally**—you can execute up to any step and verify outputs before wiring the next one.

---

**Outputs** are written under: `data/scenarios/<SCENARIO_ID>/`

```
data/scenarios/<ID>/
├─ manifest.json
├─ images/
│  ├─ image_<8hex>.png
│  └─ images_info.json
├─ descriptions/
│  └─ <image_id>.json
├─ biases/
│  ├─ repeat_summary_full.json
│  └─ bias_report.json
├─ questions/
│  └─ questions.json
├─ research/
│  ├─ facts.jsonl
│  └─ summary.txt
└─ consensus/
   └─ consensus_output.json
```

---

## Requirements

* Python **3.10+** (tested on 3.12)
* **CrewAI** ≥ 0.76
* **OpenAI** Python SDK ≥ 1.35
* **Pydantic v2**
* **python-dotenv**
* **requests**
* **replicate** (used for OSS-20B + multiple models in consensus)
* **crewai-tools** (for Serper/Scraper if used)

### Install with `pip`

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### Install with `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> If you see `ModuleNotFoundError` for `dotenv` or `replicate`, ensure they’re in `requirements.txt`, then reinstall.

---

## Environment Variables

Create `.env` (copy from `.env.example`) and fill:

```env
# Required
# Copy this to .env and fill in your key(s)
OPENAI_API_KEY="" 
IMAGE_MODEL=Dall-e3
AGENT_LLM=gpt-5-mini
REPLICATE_API_KEY=""
BIAS_REPLICATE_MODEL=replicate/openai/gpt-oss-20b
BIAS_REPLICATE_TEMPERATURE=0.0
SUMMARY_LLM=gpt-5-mini
SERPER_API_KEY=""
```

**Replicate model slugs**

* CrewAI/LiteLLM LLM string: `replicate/openai/gpt-oss-20b`
* Direct `replicate.run()` in tools: `openai/gpt-oss-20b`

If OSS-20B returns “No output.”, verify token & slug match the usage above.

---

## Usage

`src/app.py` uses **hardcoded inputs** by design:

```python
SCENARIO   = "Prompt"
SCENARIO_ID= "Prompt_id"
N_IMAGES   = 10   # set 10 for the full run
```

Run:

```bash
python -m src.app
```

### Step-by-Step (toggle in `__main__`)

Inside `app.py`, uncomment steps gradually:

```python
paths = run_generate_images(SCENARIO, SCENARIO_ID, n=N_IMAGES)  # Step 1
run_describe_images(paths)                                      # Step 2

run_analyze_bias(paths)                                       # Step 3
qs = run_generate_questions(paths)                            # Step 4a (OSS-20B)
run_fact_check(paths, qs)                                     # Step 4b (search + scrape)
run_consensus(paths)                                          # Step 5 (Replicate multi-model)
```

**Sanity checks after each step:**

* **Step 1** → `images/` has PNGs and `images_info.json`
* **Step 2** → `descriptions/` has one `ImageAuditRecord` per image
* **Step 3** → `biases/` has `repeat_summary_full.json` + `bias_report.json`
* **Step 4a** → `questions/questions.json`
* **Step 4b** → `research/facts.jsonl` + `research/summary.txt`
* **Step 5** → `consensus/consensus_output.json`

---

## How the Pipeline Works

1. **Image Generation**

   * Agent: *Visionary Image Generator*
   * Tool: `generate_image` (OpenAI Images API)
   * Output: PNGs + `images_info.json` (stores `filename`, **relative** `relpath` to scenario root)

2. **Image Description (Structured)**

   * Agent: *Image Description Structuring Analyst*
   * Tools: `DescribeImageFromFile` (data URL) and `DescribeImageFromURL`
   * Model: **gpt-5-mini** (vision chat)
   * Output: strict `ImageAuditRecord` JSON per image

3. **Bias Analysis**

   * Deterministic aggregation of people/objects/environment tokens & phrases
   * Reasoner ( **Replicate OSS-20B** ) produces a **BiasReport** (findings + notes)
   * Output: `biases/repeat_summary_full.json`, `biases/bias_report.json`

4. **Question Generation & Fact-Checking**

   * **Writer** (OSS-20B): creates **global, atomic, verifiable** questions
   * **Fact Checker** (**gpt-5-mini** + **Serper** + **Scraper**): answers with % / one-word / YES-NO and a citation
   * Output: `questions/questions.json`, `research/facts.jsonl`, `research/summary.txt`

5. **5-LLM Consensus**

   * Tool `get_expected_elements_replicate` calls multiple Replicate models to list **required elements**
   * Aggregator keeps elements appearing in **≥70%** of models
   * Output: `consensus/consensus_output.json`:

     ```json
     {
       "prompt": "...",
       "consensus_elements": ["..."],
       "individual_predictions": [
         {"model_name": "moonshotai/kimi-k2-instruct", "required_elements": [...]},
         {"model_name": "meta/meta-llama-3-8b-instruct", "required_elements": [...]},
         {"model_name": "deepseek-ai/deepseek-v3", "required_elements": [...]},
         {"model_name": "ibm-granite/granite-3.3-8b-instruct", "required_elements": [...]},
         {"model_name": "microsoft/phi-3-mini-4k-instruct:...", "required_elements": [...]}
       ]
     }
     ```

---


## Things To Keep in Mind

- I should add metric regarding hallucination rate for the time it is doing the research.
- Not able to detect the right and left correctly by VLM.
- Need to make sure about the output of each LLM for the Consensus step, might be a problem with Replicate package and agent hallucinating the output of the merge Consensus.
- Work more on the categorising the thing in the image for evaluation (src/tool/vision_description_tool.py)