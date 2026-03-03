from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig

# MPT support depends on private Transformers internals that can change across
# releases. Keep LLaMA-based usage available even when MPT cannot be imported.
try:
    from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
except Exception:  # pragma: no cover
    LlavaMPTForCausalLM = None
    LlavaMPTConfig = None
