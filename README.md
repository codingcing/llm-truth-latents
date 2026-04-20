# Truth and Falsity in LLM's

Recent publications have successfully extracted cogent 'personas' from large language models, that is, certain components of activation space corresponding to traits in the responses of these models.

<div align="center">

[![Persona Axes][persona-axes-img]][persona-axes-src]

[persona-axes-img]: assets/persona_axes.png
[persona-axes-src]: <https://arxiv.org/abs/2601.10387>

</div>

In this repo, I investigate the following questions:

1) What conception, if any, do language models have of truth and falsity?

2) Do certain subspaces of persona space correspond to higher and lower propensity for truth-telling? 

3) What is the *degree* (e.g. model layer) at which we can clearly identify the concept of truthfulness?

More broadly, the hypothesis I want to invesigate is that there exists a subspace of LLM activation space corresponding to a higher density of truthful model responses. 

By **truthful**, I mean either an answer which is factually correct, in a case where an LLM *could* ascertain the correct answer, or an abstension in the case where a correct answer either does not exist or is not knowable to the LLM. A **false** response is any response which is *not* truthful.
---

## Repository Structure

**/assets** — figures, images, and other static files referenced in the README or notebooks.

**/data** — datasets of answerable and unanswerable questions, raw model responses, and LLM-judge labels.

**/models** — model configs and any downloaded open-weights checkpoints used for analysis.

**/notebooks** — exploratory Jupyter notebooks for dataset inspection, activation visualisation, and hypothesis testing.

**/src** — source code: dataset generation, inference pipelines, LLM-as-judge, and vector extraction utilities.

**/results** — saved activation vectors, PCA/SAE outputs, K-means fits, and plots produced by the analysis pipeline.
