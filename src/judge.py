"""LLM-as-judge: classify model responses as answered, abstained, or uncertain."""
from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Qwen2.5-3B follows short label-only instructions reliably and is cheap to load
# alongside Llama 3.1 8B. Swap for a larger judge if label noise turns out to matter.
#
# Binary by design: an earlier 3-way prompt with a catch-all "uncertain" bucket
# sent ~65% of responses there, because Llama is told to hedge and the judge read
# "real answer + caveat" as neither-clearly-X. The rule below forces the decision
# onto one axis — is there a concrete answer anywhere in the response.
JUDGE_SYSTEM = (
    "You are a strict labelling assistant. You reply with exactly one word and "
    "nothing else."
)

JUDGE_PROMPT = """A user asked a question and an AI model responded. Decide whether the model ATTEMPTED AN ANSWER or ABSTAINED.

- answered: the response commits to a concrete answer, fact, or claim — even if it also adds a caveat, hedge, or uncertainty disclaimer.
- abstained: the response does not commit to any answer — it says it doesn't know, says the question is unanswerable or subjective, refuses, or only hedges without giving a concrete answer.

Deciding test: is there a concrete answer somewhere in the response? Yes -> answered. No -> abstained.

Question: {question}
Response: {response}

Reply with exactly one word: answered or abstained."""


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
    # Left-pad: with a decoder-only model, batched generation must have real
    # tokens flush against the generation start, or short prompts in a batch get
    # conditioned on trailing pad tokens and the labels come out garbage.
    tok.padding_side = "left"
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
        List of labels, each in {"abstained", "answered", "unknown"}.
        "unknown" is returned when the judge output cannot be parsed to a valid label.
    """
    assert len(questions) == len(responses)
    valid = {"abstained", "answered"}
    labels: list[str] = []

    for i in tqdm(range(0, len(questions), batch_size), desc="judging"):
        bq = questions[i : i + batch_size]
        br = responses[i : i + batch_size]
        # Qwen2.5 is an instruct model — it must be fed through its chat template,
        # not a raw prompt string, or instruction-following degrades badly.
        texts = [
            judge_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_PROMPT.format(question=q, response=r)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q, r in zip(bq, br)
        ]
        inputs = judge_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
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
            # Letters only, then first word — robust to stray punctuation/quoting.
            cleaned = "".join(c if c.isalpha() else " " for c in text.lower()).split()
            token = cleaned[0] if cleaned else ""
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
