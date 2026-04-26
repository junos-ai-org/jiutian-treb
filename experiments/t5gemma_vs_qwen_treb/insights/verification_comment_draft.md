Confirmed — PR #45540 fixes this on `google/t5gemma-2-4b-4b`. Re-ran the same length sweep from the original repro at batch=1, `attn_implementation="eager"`, on 1× H100 80GB HBM3 (RunPod EU-NL-1).

| input tokens | BEFORE (`transformers 5.5.4`) | AFTER (`#45540` @ `b8c3dff`) |
|-------------:|:------------------------------:|:----------------------------:|
| 3525  | OK                  | OK |
| 5015  | FAIL (a=4097, b=5018)  | OK |
| 6499  | FAIL (a=4097, b=6502)  | OK |
| 7493  | FAIL (a=4097, b=7496)  | OK |
| 9997  | FAIL (a=4097, b=10000) | OK |
| 14967 | FAIL (a=4097, b=14970) | OK |
| 19808 | FAIL (a=4097, b=19811) | OK |
| 25135 | FAIL (a=4097, b=25138) | OK |

The `a=4097` constant from the original report was reproduced exactly in the BEFORE column; treatment phase passed all 8 lengths including five that exceed the previous 4× sliding-window ceiling. Stack: torch 2.8.0 + CUDA 12.8 on `runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404`. Reproducer: [`verify_pr45540.{py,sh}`](https://github.com/junos-ai-org/jiutian-treb/tree/experiment-setup/experiments/t5gemma_vs_qwen_treb/insights).

Thanks @Beichen-Ma for the fix and @vasqu for the ping!
