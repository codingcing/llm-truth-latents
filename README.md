# Abstention Geometry: Knowledge and Behaviour Are Dissociable in Llama 3.1 8B

We probe **Llama 3.1 8B Instruct**'s residual stream on the SelfAware dataset and ask a calibrated-uncertainty question: does the model's internal state distinguish prompts it can answer from prompts it should refuse and, separately, does that internal state drive its behaviour?

**Findings.**
- A linear probe on the residual stream classifies answerable vs. unanswerable questions at **0.974 balanced accuracy** (peak layer 15), saturating by layer ~10.
-  Of 1,031 unanswerable questions, the model correctly abstains on only **252 (24%)** and hallucinates on the other 779.
- The model is *more* internally certain the question is unanswerable exactly when it ignores that certainty and answers. 
- Most religious theories of eschatological justice seperate ignorance into two substreams: *vincible* and *invincible* forms of ignorance. Llama 3.1 participates in vincible ignorance - when you know the right course of action but you still do the wrong thing.
- I can imagine two main sources of this problem. 1, RHLF is a lossy way to intent-align instruct models. 2, (courtesy of my little brother), the data upon which LLM's are trained is likely compatible with real-looking, completely false 'facts'.

The implication: knowledge and behaviour live on largely separate, near-orthogonal directions in the residual stream. This is a calibration failure of the *act-on-it* circuit, not a failure of the underlying representation.

<p align="center">
  <img src="figures/probe_accuracy_comparison.png" alt="knowledge vs behaviour decodability by layer" width="640">
</p>

This work was largely inspired by Lu. et al's work on the *persona axis* within LLM's, the idea that the activations of language models can be inspected and decoded to uncover cogent persona's associated with different areas of activation space. I initally wanted to extend this idea to map out a subspace corresponding to abstention, but this finding I thought too interesting not to document.

<p align="center">
  <img src="assets/persona_axes.png" alt="persona axes" width="640">
</p>

## Pipeline

Four notebooks. All run; figures and tables in `figures/` and `results/`.

### 1. Data and prompts — [src/data.py](src/data.py)
- **SelfAware**, 3,369 questions, 69%/31% answerable/unanswerable. Loaded from `data/selfaware/SelfAware.json`.
- **Llama 3.1 Instruct chat template** with a system prompt that explicitly encourages hedging:
  > *"Answer the question if you can. If you do not know or are not sure, say so clearly."*

### 2. Generation + activation caching — [notebooks/01_cache_activations.ipynb](notebooks/01_cache_activations.ipynb), [src/model.py](src/model.py)
- Llama 3.1 8B Instruct loaded in **bfloat16** via `HookedTransformer.from_pretrained_no_processing` (Colab A100).
- **Greedy decoding**, `max_new_tokens=150`, **left-padded batched generation**.
- **Residual stream `resid_post` cached at the final prompt token for all 32 layers**, with left-padding so position `-1` is the true last prompt token. Shard-checkpointed, ~1.7 GB of activations for 3,369 prompts.
- Output: `responses.json`, `acts_layer_0..31.pt`.

### 3. LLM-as-judge labelling — [src/judge.py](src/judge.py), [notebooks/02_probe_analysis.ipynb](notebooks/02_probe_analysis.ipynb) (Step 1)
- **Qwen2.5-3B-Instruct** classifies every Llama response as `answered` or `abstained`, run through Qwen's chat template.
- The deciding rule: *is there a concrete answer anywhere in the response?* If yes → `answered`; if not (refusals, "I don't know", pure hedges) → `abstained`.
- Manually spot-checked 15 random (question, Llama answer, judge label) triples; agreement perfect.
- Output: `labels.csv` — `answered: 2,476  |  abstained: 893`.

### 4. The 2 × 2: calibration scorecard

| | gold: **answerable** | gold: **unanswerable** |
|---|---|---|
| **model answered** | 1696 — correct | 779 — hallucinated |
| **model abstained** | 642 — over-refused | 252 — correctly abstained |

Llama answers 76% of unanswerable questions (hallucination) and over-refuses 27% of answerable ones. Of 893 abstentions, only 252 were on questions where abstention was the correct behaviour.

### 5. Probe sweep, two targets — [notebooks/02_probe_analysis.ipynb](notebooks/02_probe_analysis.ipynb), [src/probes.py](src/probes.py)

For each of two binary targets (`answerable` — gold; `abstained_correctly` — behaviour ∧ unanswerable), at every layer:

- **Difference-of-means** unit vector, saved as `dom_vectors_{label}.pt`.
- **Cross-layer cosine similarity** of those vectors — `figures/dom_cosine_{label}.png`. Heatmap shows the abstention direction is layer-specific in the mid-network, only stabilising in layers ~25–30.
- **Logistic probe**, `C=0.1`, `max_iter=1000`, 5-fold CV, scored with **balanced accuracy** (class imbalance handled). `figures/probe_accuracy_by_layer_{label}.png`, `results/probe_accuracy_{label}.csv`.
- **PCA**: top-2 variance per layer (`figures/pca_variance_{label}.png`) + a PC1/PC2 scatter at the peak layer (`figures/pca_peak_layer_{label}.png`).

The headline comparison overlays the two probe curves: [figures/probe_accuracy_comparison.png](figures/probe_accuracy_comparison.png).

### 6. Dissociation analysis — [notebooks/03_dissociation.ipynb](notebooks/03_dissociation.ipynb), [src/dissociation.py](src/dissociation.py)

Characterises the 0.97 / 0.74 gap.

- **Quadrant split** (gold answerability × judged behaviour) gives four behavioural outcomes.
- **Projection onto the answerability direction** at layer 15 for every activation. Per-quadrant means:

| quadrant | n | mean projection |
|---|---|---|
| `answerable_abstained` (over-refusal) | 642 | **+1.555** |
| `answerable_answered` (correct) | 1696 | +1.433 |
| `unanswerable_abstained` (correct) | 252 | −1.472 |
| `unanswerable_answered` (hallucinated) | 779 | **−1.819** |

Gold answerability moves the projection by ~3.0; behaviour, within a fixed answerability class, moves it by 0.12–0.35. Behaviour explains roughly **10% as much variance on the answerability axis as answerability itself does** — the two are largely separate directions.

The failure cases (`unanswerable_answered`) sit *further* on the unanswerable side than correct abstentions. This is the opposite of the naive "weak signal causes failure" hypothesis: the model is *more* internally certain, not less, when it hallucinates.

[figures/quadrant_distributions.png](figures/quadrant_distributions.png) visualises this — the two answerability clusters separate cleanly; the behavioural distributions within each cluster overlap heavily, with the small offset described above.

- **Conditional behavioural probe** (`src/probes.py:train_conditional_probe`) — restricted to the 1,031 unanswerable questions, predicts `answered` vs `abstained`. Peaks at **0.74 balanced accuracy** by layer ~13 and plateaus through layer 31. Behaviour *is* linearly decodable, just on a separate, weaker direction from knowledge.

[figures/conditional_vs_answerability.png](figures/conditional_vs_answerability.png) overlays the conditional and full-answerability probes.

## Repo structure

```
.
├── data/selfaware/SelfAware.json    raw dataset (gitignored)
├── notebooks/
│   ├── 00_setup.ipynb               env checks, HF login, GPU verification
│   ├── 01_cache_activations.ipynb   forward passes, cache resid_post per layer
│   ├── 02_probe_analysis.ipynb      judge + both probe sweeps + headline comparison
│   └── 03_dissociation.ipynb        quadrants, projections, conditional probe
├── src/
│   ├── data.py            SelfAware loading + Llama 3.1 chat template
│   ├── model.py           HookedTransformer load, generation, residual caching
│   ├── judge.py           Qwen2.5-3B binary judge via chat template
│   ├── probes.py          difference-of-means, logistic probe, conditional probe, PCA
│   ├── dissociation.py    quadrant assignment + projection onto a direction
│   └── paths.py           Drive-aware results-dir resolver
├── results/   acts_layer_0..31.pt, responses.json, labels.csv,
│              dom_vectors_{label}.pt, probe_accuracy_{label}.csv,
│              conditional_probe_accuracies.csv  (all gitignored)
└── figures/   probe_accuracy_comparison.png, quadrant_distributions.png,
              conditional_vs_answerability.png, and per-label probe / PCA /
              cosine plots  (gitignored)
```

## Setup

1. `pip install -r requirements.txt` (`transformer-lens`, `torch`, `transformers`, `scikit-learn`, …).
2. `huggingface-cli login` with an account that has access to `meta-llama/Llama-3.1-8B-Instruct` (gated).
3. Drop `SelfAware.json` at `data/selfaware/SelfAware.json`.

### Colab workflow (used here)

1. Open the notebook from a Drive-synced copy of the repo (`%cd "/content/drive/.../abstention-geometry"`).
2. Mount Drive; `paths.results_dir()` resolves results next to the repo so they sync back to your machine automatically.
3. A100 runtime for **notebook 01** (generation + caching, ~2–3 hr). CPU is sufficient for **02** and **03** once the activations are cached.
