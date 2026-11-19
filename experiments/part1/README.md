# DDA5001 Part I experiment results

Six runs from this repository completed on November 19, 2026. Each run trained the randomly initialized 302.4M-parameter GPT preset on the Shakespeare character dataset for 5,000 updates and 92,160,000 prediction tokens. The runs used the same seed, training budget, data, model settings and RTX 5090 D v2. The recorded source and dataset hashes match the repository. The endpoint table was checked against the final CSV entries, the run summaries and the metadata in each `last.pt`.

- [Endpoint metrics](endpoints.csv)
- [Training and validation curves](training-validation.png)
- [Per-run configurations, environments, CSV metrics, summaries, logs and plots](records)
- [SGDM endpoint text sample](sgdm-other-sample.txt), generated from `sgdm-other/last.pt` at peak LR 0.05 using CPU inference, a newline prompt, seed 1234, temperature 0.8 and 200 generated characters

## Endpoint comparison

Loss is measured in nats per character. Training minutes come from the training timer. Wall minutes cover the full run, including evaluation and checkpoint writes.

| Optimizer | Peak LR | Validation loss | Training min | Wall min |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 0.0003 | 3.9869 | 21.12 | 22.25 |
| AdamW | 0.00006 | 2.9572 | 21.01 | 22.15 |
| Muon | 0.01 | 4.0923 | 25.33 | 26.18 |
| Muon | 0.002 | 4.2958 | 25.31 | 26.18 |
| SGDM | 0.1 | 1.6582 | 21.08 | 22.13 |
| SGDM | 0.05 | 1.5359 | 21.11 | 22.27 |

## Notes for the report

- AdamW at 0.00006 improved the endpoint by 1.0297. Its validation loss overtook the default between updates 1,000 and 1,500. Both curves rose after their minima while training loss kept falling, consistent with overfitting.
- Lowering Muon's peak LR to 0.002 made the endpoint 0.2034 worse. The lower-rate run briefly led at update 1,000, then fell behind again at 1,500. The auxiliary AdamW rate stayed at 0.0003.
- SGDM at 0.05 had the lowest endpoint validation loss. The 0.1 run led at update 3,000 and the 0.05 run led at 3,500. The 0.1 run still had the lowest observed validation point across all six runs, 1.5163 at update 2,500.
- The first AdamW alternate attempt hit the account disk quota during a checkpoint write. After space was freed, the successful run restarted from random weights. All rows above use the six completed runs. Each setting was measured with one seed.

Checkpoints are stored locally for spot checks.
