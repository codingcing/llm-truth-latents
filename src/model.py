"""Model loading, generation, and residual-stream caching via TransformerLens."""
from __future__ import annotations

import os

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer


def load_model(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
) -> HookedTransformer:
    """Load a HookedTransformer in bfloat16 on the best available device.

    bfloat16 keeps the 8B model under ~16GB VRAM, which fits an A100 40GB
    comfortably without needing 8-bit or 4-bit quantisation.

    Args:
        model_name: HF model id.

    Returns:
        HookedTransformer in eval mode.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device=device,
    )
    model.eval()
    return model


@torch.no_grad()
def generate_responses(
    model: HookedTransformer,
    prompts: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 150,
) -> list[str]:
    """Generate one greedy completion per prompt.

    Greedy decoding (do_sample=False) keeps responses reproducible and removes
    sampling noise from the judge labels.

    Args:
        model: A HookedTransformer.
        prompts: Formatted (templated) prompt strings.
        batch_size: Number of prompts per forward pass.
        max_new_tokens: Cap on generated tokens.

    Returns:
        List of decoded response strings (prompt stripped), aligned to prompts.
    """
    responses: list[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="generating"):
        batch = prompts[i : i + batch_size]
        out = model.generate(
            batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
        )
        for prompt, full in zip(batch, out):
            responses.append(full[len(prompt):].strip())
    return responses


@torch.no_grad()
def cache_residual_stream(
    model: HookedTransformer,
    prompts: list[str],
    batch_size: int = 4,
    save_dir: str = "results/",
    layers: list[int] | None = None,
) -> None:
    """Cache last-token resid_post activations at each layer.

    For each prompt, runs a forward pass with hooks on `blocks.{i}.hook_resid_post`
    and saves the activation at the final prompt token. Per-layer tensors are
    concatenated across prompts and written to `{save_dir}/acts_layer_{i}.pt`
    with shape (N, d_model).

    Note: For Llama 3.1 8B (32 layers, d_model=4096) at bf16, the full cache for
    ~3.7k prompts is ~14GB. In Colab, set save_dir to a Google Drive path.

    Args:
        model: A HookedTransformer.
        prompts: Formatted prompt strings.
        batch_size: Prompts per forward pass.
        save_dir: Output directory (created if missing).
        layers: Subset of layer indices to cache. Defaults to all layers.
    """
    os.makedirs(save_dir, exist_ok=True)
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    hook_names = [f"blocks.{i}.hook_resid_post" for i in layers]
    acc: dict[int, list[torch.Tensor]] = {i: [] for i in layers}

    for i in tqdm(range(0, len(prompts), batch_size), desc="caching"):
        batch = prompts[i : i + batch_size]
        # prepend_bos=False because LLAMA_INSTRUCT_TEMPLATE already includes <|begin_of_text|>.
        tokens = model.to_tokens(batch, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        for layer in layers:
            acts = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :].float().cpu()
            acc[layer].append(acts)
        del cache

    for layer in layers:
        stacked = torch.cat(acc[layer], dim=0)
        torch.save(stacked, os.path.join(save_dir, f"acts_layer_{layer}.pt"))
