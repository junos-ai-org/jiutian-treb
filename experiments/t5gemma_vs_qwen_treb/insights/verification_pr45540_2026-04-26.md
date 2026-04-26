# transformers#45540 verification — T5Gemma 2 long-input bug fix

**Date**: 2026-04-26
**PR**: [huggingface/transformers#45540](https://github.com/huggingface/transformers/pull/45540) ("Fix cross-attention cache layer type for T5Gemma2 long inputs")
**Issue**: [huggingface/transformers#45521](https://github.com/huggingface/transformers/issues/45521) (filed 2026-04-20)
**Verdict**: **PR fixes the bug.** 8/8 lengths pass in the treatment phase.

## Setup

- Pod: RunPod EU-NL-1, 1× NVIDIA H100 80GB HBM3, image `achithanar/treb-eval-t5gemma:latest` on `runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404`
- Stack: torch 2.8.0 + CUDA 12.8, Python 3.12
- Model: `google/t5gemma-2-4b-4b`, `attn_implementation="eager"`, `batch=1`, `max_new_tokens=8`
- Lengths: same 8 sample lengths from the original bug-report table (one control + 7 known-failing)
- Cost / runtime: ~13 min wall, ~$0.65 (H100 80GB at $2.99/hr)

## Results

| input tokens | BEFORE (`transformers 5.5.4`) | AFTER (`#45540` @ `b8c3dff`) |
|-------------:|:-----------------------------:|:----------------------------:|
| 3525  | OK                     | OK |
| 5015  | FAIL (a=4097, b=5018)  | OK |
| 6499  | FAIL (a=4097, b=6502)  | OK |
| 7493  | FAIL (a=4097, b=7496)  | OK |
| 9997  | FAIL (a=4097, b=10000) | OK |
| 14967 | FAIL (a=4097, b=14970) | OK |
| 19808 | FAIL (a=4097, b=19811) | OK |
| 25135 | FAIL (a=4097, b=25138) | OK |

**Control phase**: 1/8 OK; the constant `a=4097` (`= 4 × sliding_window + 1`) appeared on every failure, exactly as in the original report — bug reproduced cleanly.

**Treatment phase**: 8/8 OK; the cross-attention cache is now built as a plain `DynamicLayer` (not `DynamicSlidingWindowLayer`), so the merged self+cross-attention mask shape lines up with the KV cache at every input length we tested.

## Reproducer

- `verify_pr45540.py` — single-phase verifier (runs against whatever transformers is currently installed)
- `verify_pr45540.sh` — control/treatment harness; uninstalls + reinstalls transformers from `git+https://github.com/Beichen-Ma/transformers@fix-cross-attention-cache-not-sliding` between phases

To re-run on a fresh pod:

```bash
HF_TOKEN=hf_xxx bash insights/verify_pr45540.sh
```

The raw pod-side log (`verify_pr45540_20260426-035016.log`, 21 KB, mostly weight-load progress bars) stays gitignored locally; the table above is the canonical extract.

## Process notes (for future verification runs)

- The `runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404` image's `.bashrc` injects HF_TOKEN above the standard `[ -z "$PS1" ] && return` non-interactive guard, so `bash -c` and `bash -l -c` from a fresh SSH session do **not** see HF env vars. Pass `HF_TOKEN` explicitly with `env HF_TOKEN=… bash …`.
- Ubuntu 24.04's PEP 668 blocks system-pip; the harness uses `pip install --break-system-packages` for the PR-branch reinstall. Acceptable on a throwaway pod; would need a venv for a long-lived environment.
