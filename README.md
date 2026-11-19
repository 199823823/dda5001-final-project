# DDA5001 — Personal Final Project

Part I preparation based on the official course repository. The seven exercise
TODOs are completed. Official checks and all three CPU smoke runs passed locally;
classroom CUDA and the six full experiments still need to be run.

## Classroom Ubuntu quick start

```bash
git clone https://github.com/199823823/dda5001-final-project.git
cd dda5001-final-project
bash lab-linux.sh
bash lab-linux.sh train default
```

The first script command activates `/opt/dda5001/venv` and verifies the GPU plus
three tiny end-to-end runs. Run full experiments only after verification passes.
The script uses project-relative paths and needs no Windows environment files.
For a private repository, sign in through GitHub's normal authentication flow.

Read [the Chinese lab guide](LAB_GUIDE_CN.md) for choosing the other learning rates,
collecting results, and backing up before the classroom machine reboots.
See [local validation evidence](LOCAL_VALIDATION.txt) for exactly what was tested.
Read [the Lecture 10 implementation notes](LECTURE_NOTES_CN.md) for AdamW/Muon
details and interpreting the endpoint table. `python lab.py audit` checks that
only the permitted learning-rate settings vary across the experiments.

## Official course information

Course code for DDA5001 at CUHK-Shenzhen. See the handbook and Blackboard for deadlines and submission requirements.

| Part | Topic |
| --- | --- |
| [Part I](p1/README.md) | GPT pre-training: optimizers and learning rates |
| Part II | Coming later |
| Part III | Coming later |

Based on [nanoGPT](https://github.com/karpathy/nanoGPT) and the [2025 course code](https://github.com/CUHKSZ-Course/DDA5001-25Fall). See [LICENSE](LICENSE).
