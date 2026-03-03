from .language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM

# MPT support depends on private Transformers internals that can change across
# releases. Keep LLaMA-based usage available even when MPT cannot be imported.
try:
    from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM
except Exception:  # pragma: no cover
    LlavaMPTConfig = None
    LlavaMPTForCausalLM = None
