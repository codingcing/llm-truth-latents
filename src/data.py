"""Dataset loading and prompt construction for the abstention-geometry project."""
from __future__ import annotations

import json
import pandas as pd

# Llama 3.1 Instruct chat template. The system prompt frames abstention as a valid
# response so the model is not pushed to answer when it shouldn't.
LLAMA_INSTRUCT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Answer the question if you can. If you do not know or are not sure, say so clearly."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def load_selfaware(path: str) -> pd.DataFrame:
    """Load SelfAware JSON into a DataFrame.

    Args:
        path: Path to SelfAware.json (raw or topic-enriched).

    Returns:
        DataFrame with columns [question_id, question, answerable, topic]. If the
        source JSON has no 'topic' field, rows are tagged 'uncategorized'.
    """
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for i, item in enumerate(raw["example"]):
        rows.append({
            "question_id": item.get("question_id", item.get("id", i)),
            "question": item["question"],
            "answerable": bool(item["answerable"]),
            "topic": item.get("topic", "uncategorized"),
        })
    return pd.DataFrame(rows)


def build_prompts(df: pd.DataFrame) -> list[str]:
    """Format each question with the Llama 3.1 Instruct chat template.

    Args:
        df: DataFrame from load_selfaware.

    Returns:
        List of formatted prompt strings, aligned row-wise to df.
    """
    return [LLAMA_INSTRUCT_TEMPLATE.format(question=q) for q in df["question"].tolist()]
