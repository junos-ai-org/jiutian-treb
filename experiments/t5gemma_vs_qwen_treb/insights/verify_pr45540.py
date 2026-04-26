"""Verify huggingface/transformers PR #45540 fixes the T5Gemma 2 long-input
shape bug we filed in #45521.

Runs the exact failure-length sweep from the bug report (table in #45521)
against whatever transformers is currently installed and prints a
pass/fail summary. Run it twice — once on stock transformers (control),
once after installing the PR branch (treatment) — see verify_pr45540.sh
for the full before/after harness.

Pass criterion (treatment run):
  - All lengths above 4094 generate without RuntimeError.
  - Specifically the "tensor a (4097) must match tensor b (N)" mismatch
    from #45521 must be gone.

Needs a GPU with ~40 GB VRAM (H100/A100) and HF auth for
`google/t5gemma-2-4b-4b` (gated).
"""
import importlib.metadata
import subprocess
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "google/t5gemma-2-4b-4b"

# Lengths from the failure table in transformers#45521 — one known-good
# control (3525) plus the seven that previously raised RuntimeError.
LENGTHS = [3525, 5015, 6499, 7493, 9997, 14967, 19808, 25135]


def transformers_provenance() -> str:
    version = importlib.metadata.version("transformers")
    try:
        import transformers
        path = transformers.__path__[0]
        sha = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{version} (git {sha})"
    except Exception:
        return f"{version} (no git checkout)"


def make_prompt(target: int) -> str:
    filler = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |\n"
    return (
        "Answer concisely.\n\nTable: "
        + (filler * max(1, target // 20))
        + "\n\nQuestion: sum?"
    )


def run_one(model, tokenizer, target: int) -> tuple[str, int, str]:
    prompt = make_prompt(target)
    ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=target
    ).input_ids
    actual = int(ids.shape[-1])
    try:
        with torch.no_grad():
            model.generate(ids.to(model.device), max_new_tokens=8, do_sample=False)
        return ("OK", actual, "")
    except RuntimeError as e:
        return ("FAIL", actual, str(e).splitlines()[0])


def main() -> int:
    print(f"transformers: {transformers_provenance()}")
    print(f"torch:        {torch.__version__}")
    print(f"cuda:         {torch.version.cuda}  available={torch.cuda.is_available()}")
    print(f"device:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"model:        {MODEL}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    ).eval()

    print(f"{'target':>8}  {'actual':>8}  {'status':>6}  detail")
    print(f"{'-'*8}  {'-'*8}  {'-'*6}  {'-'*60}")
    results = []
    for target in LENGTHS:
        status, actual, detail = run_one(model, tokenizer, target)
        results.append((target, actual, status, detail))
        print(f"{target:>8}  {actual:>8}  {status:>6}  {detail[:60]}")

    failures = [r for r in results if r[2] == "FAIL"]
    long_failures = [r for r in failures if r[1] > 4094]
    print()
    print(f"summary: {len(results) - len(failures)}/{len(results)} OK; "
          f"{len(long_failures)} failures at length > 4094")
    # Treatment-run pass criterion: no failures above the SWA boundary.
    return 0 if not long_failures else 1


if __name__ == "__main__":
    sys.exit(main())
