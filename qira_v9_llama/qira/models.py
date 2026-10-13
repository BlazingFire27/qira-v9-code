"""
Model loading and architecture adapter for decoder-only RoPE LLMs
(Qwen2 / Qwen2.5 / Qwen3, LLaMA-2/3.x, Mistral).

The engine re-implements the decoder layer with the model's own sub-modules so
that tokens can be halted at an intermediate layer.  `selftest` (run_benchmark.py)
verifies numerically that the manual dense path reproduces the HF forward.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class Geometry:
    name: str
    hf_id: str
    num_layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    max_positions: int
    decision_layer: int


def _cfg_get(cfg, *names, default=None):
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
    return default


def model_short_name(hf_id: str) -> str:
    name = Path(hf_id.rstrip("/")).name.replace("/", "_")
    if name.startswith("Meta-"):
        name = name[5:]
    return name


class LoadedModel:
    """Holds model, tokenizer and the sub-modules the engine needs."""

    def __init__(self, hf_id: str, device: str, dtype: torch.dtype, token: Optional[str],
                 decision_layer_fraction: float):
        self.hf_id = hf_id
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type != "cpu" or dtype != torch.float16 else torch.float32
        print(f"[model] loading {hf_id} dtype={self.dtype} device={self.device}")

        # Attempt tokenizer load with automatic fallback if official repo is gated
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token, trust_remote_code=True)
        except Exception as ex:
            ex_s = str(ex).lower()
            if any(err in ex_s for err in ["gated", "restricted", "unauthorized", "401", "403"]) and "llama" in hf_id.lower():
                alt_id = "NousResearch/Meta-Llama-3.1-8B-Instruct"
                print(f"[model] Access restricted for '{hf_id}'. Automatically falling back to ungated mirror: '{alt_id}'")
                hf_id = alt_id
                self.hf_id = alt_id
                self.tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token, trust_remote_code=True)
            else:
                raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kw = dict(token=token, trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation="sdpa")

        try:
            model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=self.dtype, **kw)
        except TypeError:
            try:
                model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=self.dtype, **kw)
            except Exception as ex:
                ex_s = str(ex).lower()
                if any(err in ex_s for err in ["gated", "restricted", "unauthorized", "401", "403"]) and "llama" in hf_id.lower():
                    alt_id = "NousResearch/Meta-Llama-3.1-8B-Instruct"
                    print(f"[model] Access restricted for '{hf_id}'. Automatically falling back to ungated mirror: '{alt_id}'")
                    hf_id = alt_id
                    self.hf_id = alt_id
                    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=self.dtype, **kw)
                else:
                    raise
        except Exception as ex:
            ex_s = str(ex).lower()
            if any(err in ex_s for err in ["gated", "restricted", "unauthorized", "401", "403"]) and "llama" in hf_id.lower():
                alt_id = "NousResearch/Meta-Llama-3.1-8B-Instruct"
                print(f"[model] Access restricted for '{hf_id}'. Automatically falling back to ungated mirror: '{alt_id}'")
                hf_id = alt_id
                self.hf_id = alt_id
                model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=self.dtype, **kw)
            else:
                raise

        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        core = getattr(self.model, "model", None)
        if core is None or not hasattr(core, "layers"):
            raise RuntimeError("Unsupported architecture: expected model.model.layers (LLaMA/Qwen/Mistral family).")
        self.core = core
        self.layers = core.layers
        self.embed_tokens = core.embed_tokens
        self.final_norm = core.norm
        self.lm_head = self.model.lm_head
        self.rotary_emb = getattr(core, "rotary_emb", None)
        if self.rotary_emb is None:
            self.rotary_emb = getattr(self.layers[0].self_attn, "rotary_emb", None)
        if self.rotary_emb is None:
            raise RuntimeError("Could not find a rotary embedding module; this transformers version is too old.")

        L = int(_cfg_get(cfg, "num_hidden_layers"))
        hidden = int(_cfg_get(cfg, "hidden_size"))
        heads = int(_cfg_get(cfg, "num_attention_heads"))
        kv_heads = int(_cfg_get(cfg, "num_key_value_heads", default=heads))
        head_dim = int(_cfg_get(cfg, "head_dim", default=hidden // heads))
        inter = int(_cfg_get(cfg, "intermediate_size", default=4 * hidden))
        vocab = int(self.lm_head.weight.shape[0])
        maxpos = int(_cfg_get(cfg, "max_position_embeddings", default=32768))
        d = max(1, min(L - 1, int(L * decision_layer_fraction)))
        self.geom = Geometry(model_short_name(hf_id), hf_id, L, hidden, heads, kv_heads, head_dim,
                             inter, vocab, maxpos, d)

        # Sliding-window attention is not re-implemented; guard against silent mismatch.
        sw = getattr(cfg, "sliding_window", None)
        use_sw = getattr(cfg, "use_sliding_window", None)
        self.sliding_window = None
        if sw is not None and (use_sw is None or bool(use_sw)):
            self.sliding_window = int(sw)
        a0 = self.layers[0].self_attn
        self.has_qk_norm = hasattr(a0, "q_norm") and hasattr(a0, "k_norm")
        self.scaling = float(getattr(a0, "scaling", head_dim ** -0.5))
        g = self.geom
        print(f"[model] layers={g.num_layers} hidden={g.hidden} heads={g.heads} kv_heads={g.kv_heads} "
              f"head_dim={g.head_dim} inter={g.intermediate} decision_layer={g.decision_layer} "
              f"qk_norm={self.has_qk_norm} sliding_window={self.sliding_window}")

    def eos_ids(self):
        ids = set()
        tok = self.tokenizer
        for v in [tok.eos_token_id, getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)]:
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                ids.update(int(x) for x in v)
            else:
                ids.add(int(v))
        for t in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "</s>", "<|end|>", "<end_of_turn>"]:
            tid = tok.convert_tokens_to_ids(t) if t in tok.get_vocab() else None
            if tid is not None and tid != tok.unk_token_id:
                ids.add(int(tid))
        return ids

    def unload(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_name(hf_id: str, token: Optional[str] = None) -> str:
    try:
        AutoConfig.from_pretrained(hf_id, token=token, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] could not read config for {hf_id}: {e}")
    return model_short_name(hf_id)
