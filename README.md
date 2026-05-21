# Abstention Geometry: Probing the Residual Stream for Calibrated Uncertainty

We investigate whether **Llama 3.1 8B Instruct** internally distinguishes prompts it can
answer from prompts it should hedge or refuse — i.e., whether there is a linear subspace
of the residual stream encoding *calibrated abstention*. This is a study of the model's
own knows / doesn't-know signal on the SelfAware dataset. I was inspired by recent work on
the Assistant Axis, which describes the structure of the vector space within which
LLM activations live, and uses it to make predictions about the *persona* it will inhabit.

<p align="center">
  <img src="assets/persona_axes.png" alt="Persona axes" width="600">
</p>

## Research questions

1. Is there a linear direction in the residual stream that separates answerable from
   unanswerable prompts (as defined by SelfAware)?
2. At which layer is this direction most clearly identifiable?
3. Does the direction generalise across question topics, or is it topic-specific?

## Repo structure

```
.
├── data/selfaware/             SelfAware.json (drop in manually; gitignored)
├── notebooks/
│   ├── 00_setup.ipynb          env checks, HF login, GPU verification
│   ├── 01_cache_activations.ipynb   forward passes, cache resid_post per layer
│   ├── 02_probe_analysis.ipynb      probes + headline figure
│   └── 03_generalization.ipynb      cross-topic eval
├── src/
│   ├── data.py                 dataset loading + prompt templating
│   ├── model.py                model load, generation, activation caching
│   ├── judge.py                LLM-as-judge label pipeline
│   ├── probes.py               diff-of-means, logistic probes, PCA
│   └── generalization.py       cross-topic holdouts
├── results/                    activations, labels (gitignored)
└── figures/                    plots (gitignored)
```

## Setup

1. `pip install -r requirements.txt`
2. `huggingface-cli login` (needed for gated Llama weights)
3. Download SelfAware.json from the SelfAware repo and place it at
   `data/selfaware/SelfAware.json`.

## Colab Pro+ workflow

1. `!git clone <repo-url>` in the runtime, then `cd` into it.
2. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`.
3. Point `RESULTS_DIR` at a Drive path so cached activations survive runtime restarts.
   The full 32-layer cache for 3.7k prompts is ~14GB in bf16.
4. Use an A100 runtime for activation caching and generation; CPU is fine for the probe
   analysis once activations are cached.

Large artifacts (`results/*.pt`, `results/*.csv`, `figures/*`) are gitignored. Treat
Drive as the source of truth for cached outputs; only code and the README live in git.
