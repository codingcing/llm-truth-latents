"""LLM-as-judge: classify model responses as answered, abstained, or uncertain."""
from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Qwen2.5-3B follows short label-only instructions reliably and is cheap to load
# alongside Llama 3.1 8B. Swap for a larger judge if label noise turns out to matter.
JUDGE_PROMPT = """You are evaluating whether a model's response answers a question or abstains.

Classify the response as exactly one of:
- "abstained": the response refuses, says it doesn't know, or hedges without committing
- "answered": the response gives a direct factual answer
- "uncertain": neither clearly answered nor clearly abstained

Question: {question}
Response: {response}

Reply with one word only: abstained, answered, or uncertain.
Label:"""


def load_judge(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the judge model and tokenizer in bfloat16.

    Args:
        model_name: HF model id of the judge.

    Returns:
        Tuple of (model, tokenizer), model in eval mode on the best available device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tok


@torch.no_grad()
def judge_responses(
    judge_model,
    judge_tokenizer,
    questions: list[str],
    responses: list[str],
    batch_size: int = 8,
) -> list[str]:
    """Run the judge on each (question, response) pair.

    Args:
        judge_model: Judge LM.
        judge_tokenizer: Matching tokenizer.
        questions: Original questions.
        responses: Model responses to evaluate.
        batch_size: Pairs per forward pass.

    Returns:
        List of labels, each in {"abstained", "answered", "uncertain", "unknown"}.
        "unknown" is returned when the judge output cannot be parsed to a valid label.
    """
    assert len(questions) == len(responses)
    valid = {"abstained", "answered", "uncertain"}
    labels: list[str] = []

    for i in tqdm(range(0, len(questions), batch_size), desc="judging"):
        bq = questions[i : i + batch_size]
        br = responses[i : i + batch_size]
        prompts = [JUDGE_PROMPT.format(question=q, response=r) for q, r in zip(bq, br)]
        inputs = judge_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(judge_model.device)
        out = judge_model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )
        gen = out[:, inputs.input_ids.shape[1]:]
        decoded = judge_tokenizer.batch_decode(gen, skip_special_tokens=True)
        for text in decoded:
            token = text.strip().lower().split()[0] if text.strip() else ""
            labels.append(token if token in valid else "unknown")
    return labels


def save_labels(
    questions_df: pd.DataFrame,
    responses: list[str],
    labels: list[str],
    path: str = "results/labels.csv",
) -> None:
    """Save the full labelled dataset.

    Args:
        questions_df: Output of data.load_selfaware (aligned row-wise to responses).
        responses: Model responses.
        labels: Judge labels.
        path: Output CSV path.
    """
    out = pd.DataFrame({
        "question_id": questions_df["question_id"].values,
        "question": questions_df["question"].values,
        "topic": questions_df["topic"].values,
        "answerable": questions_df["answerable"].values,
        "model_response": responses,
        "judge_label": labels,
    })
    out.to_csv(path, index=False)
