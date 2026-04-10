# jiutian-treb

TReB (Table Reasoning Benchmark) — evaluates LLMs on 26 table reasoning sub-tasks
across 7,790 samples. Paper: arxiv:2506.18421. Original repo: github.com/JT-LM/jiutian-treb.
Dataset: HuggingFace `JT-LM/JIUTIAN-TReB`.

We use this as a **second benchmark** (alongside MMTU at `~/research/MMTU`) to validate
whether T5Gemma's encoder-decoder bidirectional attention advantages hold across benchmarks.

## Setup

```bash
cd ~/research/jiutian-treb
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install accelerate sentencepiece protobuf openai
# GPU: pip install flash-attn --no-build-isolation
```

## Architecture

```
jiutian-treb/
├── src/
│   ├── run_eval.py                    # Entry point: --run_step reason|judge
│   ├── llm/
│   │   ├── vllmcaller.py             # Original vLLM caller (decoder-only models)
│   │   ├── hfseq2seqcaller.py        # OUR ADDITION: HF caller for encoder-decoder (T5Gemma)
│   │   └── openai_caller.py          # OUR ADDITION: OpenAI API caller for judge (GPT-5.4-mini)
│   ├── reason/
│   │   ├── TCoT_reason.py            # Text Chain-of-Thought (what we use)
│   │   ├── PoT_reason.py             # Program-of-Thought (code generation)
│   │   └── ICoT_reason.py            # Interleaved CoT (multi-turn plan→execute→reflect)
│   ├── judge/
│   │   ├── judger.py                 # Routes to metric-specific judgers
│   │   ├── ROUGE_judger.py           # ROUGE-L
│   │   ├── EM_judger.py              # Exact Match
│   │   └── llm_judger.py             # LLM-as-Judge (uses llmcaller.call_batch)
│   ├── sample_format/sample.py       # Sample class — data loading & serialization
│   ├── prompts/                      # System prompts per task/mode
│   ├── result_parsers/               # Parse model output (JSON, code, boxed)
│   └── util/file_op.py              # File I/O with safe-path checking
├── scripts/                          # OUR ADDITIONS
│   ├── sample_dataset.py             # Stratified sampling from HuggingFace
│   ├── generate_configs.py           # Generate experiment config JSONs
│   ├── analyze_results.py            # Aggregate judge scores per task
│   ├── compare_models.py             # Side-by-side Qwen vs T5Gemma
│   └── cross_benchmark.py            # Unified MMTU + TReB comparison
├── experiments/
│   └── bidir_attn/                   # Our experiment
│       ├── configs/                  # 4 JSON configs (smoke/large × qwen/t5gemma)
│       ├── data/{smoke,large}/       # Sampled datasets (JSON per task + manifest.csv)
│       └── output/                   # Results
├── docker/
│   ├── build.sh                      # Build all Docker images
│   ├── Dockerfile.base               # Shared base (TReB deps minus vLLM)
│   ├── Dockerfile.qwen               # Qwen backend (+ vLLM)
│   ├── Dockerfile.t5gemma            # T5Gemma backend (+ flash-attn)
│   ├── entrypoint-qwen.sh            # Clone repo, download weights, ready banner
│   └── entrypoint-t5gemma.sh         # Clone repo, download weights, ready banner
└── config/config_example.json        # TReB's original config template
```

## Our Modifications to TReB

We added 3 things to the upstream codebase:

1. **`src/llm/hfseq2seqcaller.py`** — HFSeq2SeqCaller for T5Gemma. Matches VLLMCaller
   interface (`call_batch(List[List[Dict]]) -> List[str]`). T5Gemma cannot run on vLLM
   (no encoder-decoder support in vLLM V1). Uses `AutoModelForSeq2SeqLM` + `AutoProcessor`.

2. **`src/llm/openai_caller.py`** — OpenAICaller for GPT-5.4-mini as LLM judge. TReB's
   default judge (Qwen2-72B via vLLM) requires too much GPU. Same `call_batch`/`call` interface.

3. **`src/run_eval.py`** — Modified `main()` to branch on `model_config.caller_type`
   (`vllm` | `hf_seq2seq`) and `model_config.judge_caller_type` (`vllm` | `openai`).

4. **`src/util/file_op.py`** — Added `../experiments/` to `is_safe_path()`.

## Key Concepts

### Caller Interface

All callers implement the same interface consumed by reasoners and judges:

```python
def call_batch(self, samples: list[list[dict]]) -> list[str]:
    """Input: list of conversations [[{"role": "system", ...}, {"role": "user", ...}], ...]
       Output: list of generated text strings"""

def call(self, messages: list[dict]) -> str:
    """Single conversation."""
```

### Inference Modes

- **TCoT** — pure text reasoning. NLU tasks (no table). Output: JSON `{"thought": ..., "answer": ...}`
- **TCoT_md** — text reasoning with markdown table injected. Table tasks. Same output format.
- **TCoT_html** — same but HTML table. We don't use this.
- **PoT** — model generates Python code, executed in sandbox. We don't use this (T5Gemma not code-tuned).
- **ICoT** — multi-turn iterative reasoning. We don't use this.

We run **TCoT / TCoT_md only** — closest to MMTU's text-in/text-out evaluation.

### Evaluation Metrics

- **ROUGE-L** — lexical overlap (computed locally, no LLM needed)
- **EM** (Exact Match) — string/value equality (computed locally)
- **LLM_score** — semantic correctness scored 0-10 by judge model (GPT-5.4-mini via API)
- **Instruction_Following** — special metric for Instruction_Following task only

### Data Format

Each sample is a dict with: `id`, `file_path`, `instruction`, `question`, `answer`,
`title`, `columnslable`, `Table_markdown`, `Table_html`, `number_answer`. Task name
is extracted from the `id` prefix (before `|`).

## Experiment Lifecycle

Same 4-step process as MMTU (see `~/research/MMTU/CLAUDE.md`):

1. **Develop** — write code, configs locally. Push to git.
2. **Build** — on the build server, build Docker images (one per backend to avoid
   dependency conflicts: vLLM for Qwen, HF transformers for T5Gemma). Push to Docker Hub.
3. **Run** — on **RunPod** (GPU), pull the image, sample datasets, run inference + judge.
4. **Commit** — commit results and analysis back to git.

Steps 1-2 and 4 happen locally. Step 3 happens on RunPod.
The judge step (GPT-5.4-mini via OpenAI API) can run anywhere with network access.

### Docker

One image per backend (vLLM and HF transformers have conflicting deps).

```bash
# Build all images locally
./docker/build.sh --no-push

# Build and push to Docker Hub
./docker/build.sh --repo achithanar --tag v1
```

Images:
- `treb-base` — shared base (TReB deps minus vLLM, not pushed)
- `treb-qwen` — adds vLLM for Qwen2.5-14B
- `treb-t5gemma` — adds flash-attn for T5Gemma-9B

### RunPod

```bash
# Qwen pod
docker run --gpus all -e TREB_GIT_URL=... -e TREB_GIT_REF=main \
    -e DEPLOY_KEY=... -e OPENAI_API_KEY=... \
    achithanar/treb-qwen:v1

# T5Gemma pod
docker run --gpus all -e TREB_GIT_URL=... -e TREB_GIT_REF=main \
    -e DEPLOY_KEY=... -e OPENAI_API_KEY=... \
    achithanar/treb-t5gemma:v1
```

Env vars:
- `TREB_GIT_URL` — git clone URL (default: upstream public repo)
- `TREB_GIT_REF` — branch/tag to checkout (default: main)
- `DEPLOY_KEY` — base64-encoded SSH key (only needed for private forks)
- `OPENAI_API_KEY` — for GPT-5.4-mini judge
- `QWEN_MODEL_PATH` / `T5GEMMA_MODEL_PATH` — override HuggingFace model IDs

## Workflow (detailed)

### 1. Sample datasets (run once, locally or on RunPod)

```bash
python scripts/sample_dataset.py --size smoke --output experiments/bidir_attn/data/smoke/ --seed 42
python scripts/sample_dataset.py --size large --output experiments/bidir_attn/data/large/ --seed 42
```

- **Smoke:** 3 samples/task = ~78 total (pipeline validation)
- **Large:** 1000 total, stratified with floor 10/task

### 2. Generate configs (run once, or after changing model settings)

```bash
python scripts/generate_configs.py
```

Produces 4 configs in `experiments/bidir_attn/configs/`:
- `config_qwen_smoke.json`, `config_qwen_large.json`
- `config_t5gemma_smoke.json`, `config_t5gemma_large.json`

### 3. Run inference (on GPU machine)

```bash
cd src

# Smoke first to validate
python run_eval.py --config ../experiments/bidir_attn/configs/config_qwen_smoke.json --run_step reason
python run_eval.py --config ../experiments/bidir_attn/configs/config_t5gemma_smoke.json --run_step reason

# Large after smoke validates
python run_eval.py --config ../experiments/bidir_attn/configs/config_qwen_large.json --run_step reason
python run_eval.py --config ../experiments/bidir_attn/configs/config_t5gemma_large.json --run_step reason
```

Results go to `eval_output/<version>/<model_name>/`.

### 4. Run judge (needs OPENAI_API_KEY)

```bash
export OPENAI_API_KEY=...
python run_eval.py --config ../experiments/bidir_attn/configs/config_qwen_smoke.json --run_step judge
python run_eval.py --config ../experiments/bidir_attn/configs/config_t5gemma_smoke.json --run_step judge
```

### 5. Analyze & compare

```bash
cd ..
python scripts/analyze_results.py --version bidir_attn_smoke --model Qwen2.5-14B-Instruct --save_json results/qwen.json
python scripts/analyze_results.py --version bidir_attn_smoke --model t5gemma-9b-9b-ul2-it --save_json results/t5gemma.json
python scripts/compare_models.py --qwen_json results/qwen.json --t5gemma_json results/t5gemma.json
```

### 6. Cross-benchmark comparison (MMTU + TReB)

```bash
python scripts/cross_benchmark.py \
    --mmtu_qwen ~/research/MMTU/.../qwen_analysis.json \
    --mmtu_t5gemma ~/research/MMTU/.../t5gemma_analysis.json \
    --treb_qwen results/qwen.json \
    --treb_t5gemma results/t5gemma.json
```

## Models

| Model | Architecture | Params | Caller | Config key |
|---|---|---|---|---|
| Qwen2.5-14B-Instruct | Decoder-only | 14B active | VLLMCaller | `caller_type: vllm` |
| T5Gemma-9B-9B-UL2-IT | Encoder-decoder | ~18B total, ~9B active decoding | HFSeq2SeqCaller | `caller_type: hf_seq2seq` |
| GPT-5.4-mini | (judge only) | — | OpenAICaller | `judge_caller_type: openai` |

## Config Keys (our additions to TReB's config format)

```json
{
  "model_config": {
    "caller_type": "vllm | hf_seq2seq",
    "judge_caller_type": "vllm | openai",
    "max_new_tokens": 512,
    "hf_batch_size": 4
  }
}
```

- `caller_type` — which caller to use for the reason step (default: `vllm`)
- `judge_caller_type` — which caller to use for the judge step (default: `vllm`)
- `max_new_tokens` — max generation length for HF seq2seq (default: 512)
- `hf_batch_size` — GPU sub-batch size for HF seq2seq to avoid OOM (default: 4)

## TReB's 26 Tasks (6 categories)

| Category | Tasks |
|---|---|
| **NLU** (6) | Understanding, Instruction Following, Hallucination, Robustness, Code Gen, Math Reasoning |
| **Table Understanding** (6) | Retrieval, Summary, Column Naming, Title Naming, Fact Checking, Plausibility |
| **Table Basic Ops** (2) | Query, Selection |
| **Table Computational Ops** (2) | General Ops, Domain-Specific Ops |
| **Data Analysis** (4) | Outlier Detection, Correlation, Hypothesis Testing, Distribution Testing |
| **Advanced Data Analysis** (6) | Multi-step: Retrieval, Fact Checking, Ops, Correlation, Hypothesis, Conditional Calc |

## Related Work

- **MMTU experiments:** `~/research/MMTU/projects/tabular-llms-research/`
  - `encoder_vs_decoder_baseline` — same model pair on 15 MMTU tasks (861 samples)
  - `insights/20260401-imputation-list-deep-dive/` — per-question analysis of Data-Imputation + List-to-table
  - Key finding: T5Gemma wins annotation/positional tasks, Qwen wins reasoning-heavy tasks, overall tied

## Code Conventions

- Python 3.11+
- `snake_case` for functions/variables
- TReB's existing code doesn't follow strict conventions — our additions follow the same style for consistency
- All new callers must implement `call_batch(List[List[Dict]]) -> List[str]` and `call(List[Dict]) -> str`
