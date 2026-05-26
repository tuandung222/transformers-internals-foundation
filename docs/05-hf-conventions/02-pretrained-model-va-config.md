---
title: PreTrainedModel và Config
---

# `PreTrainedModel` và `PreTrainedConfig`

Hai class này là backbone của mọi model HF. Chương này đọc cấu trúc, thấy chúng cung cấp gì cho subclass.

## `PreTrainedConfig`

File: `src/transformers/configuration_utils.py`. Class này là dataclass cho hyper-parameter của một model. Mỗi model có một subclass: `LlamaConfig`, `T5Config`, `WhisperConfig`, ...

```python
class PreTrainedConfig(PushToHubMixin):
    model_type: str = ""
    keys_to_ignore_at_inference: list[str] = []
    attribute_map: dict[str, str] = {}

    def __init__(self, **kwargs):
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)
        ...
        # Model-specific kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
```

Subclass:

```python
class LlamaConfig(PreTrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        ...
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        ...
```

Hai use chính:

**Use 1**: serialize/deserialize. `config.to_json_file("config.json")` và `LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")`. Hub cache config riêng từ weight.

**Use 2**: dispatch class. `AutoConfig.from_pretrained(...)` đọc `model_type` từ JSON, lookup class trong registry, instantiate đúng config class.

## `model_type` là gì?

```python
class LlamaConfig(PreTrainedConfig):
    model_type = "llama"
```

String identifier unique. Trong `config.json`:

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  ...
}
```

`AutoConfig` đọc field này, lookup `CONFIG_MAPPING["llama"] = LlamaConfig`. Tương tự `AutoModel` đọc và lookup model class.

## `attribute_map`

```python
class LlamaConfig(PreTrainedConfig):
    attribute_map = {
        "num_attention_heads": "num_attention_heads",
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_hidden_layers",
    }
```

Cho phép alias: `config.n_layer` map sang `config.num_hidden_layers`. Mục đích: legacy compat (GPT-2 dùng `n_layer`, T5 dùng `num_layers`, ... HF chuẩn hoá thành `num_hidden_layers`, alias còn lại).

## `PreTrainedModel`

File: `src/transformers/modeling_utils.py`, line ~1186. Class này là base class cho mọi model.

```python
class PreTrainedModel(nn.Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin):
    config_class: type[PreTrainedConfig] | None = None
    _auto_class = None
    base_model_prefix: str = ""
    _is_stateful: bool = False
    model_tags: list[str] | None = None

    # Loading-specific
    _tied_weights_keys: dict[str, str] = None
    _keys_to_ignore_on_load_missing: set[str] | list[str] | None = None
    _keys_to_ignore_on_load_unexpected: set[str] | list[str] | None = None
    _keys_to_ignore_on_save: set[str] | list[str] | None = None

    # Attention support
    _supports_sdpa: bool = False
    _supports_flash_attn: bool = False
    _supports_flex_attn: bool = False
    _compatible_flash_implementations: list[str] | None = None
```

Một số attribute key:

- `config_class`: class config tương ứng. `LlamaPreTrainedModel.config_class = LlamaConfig`.
- `base_model_prefix`: tên attribute trỏ tới base model. Với `LlamaForCausalLM`, `base_model_prefix = "model"` vì `self.model = LlamaModel(...)`. Dùng để gather state_dict đúng cách.
- `_tied_weights_keys`: dict map "target -> source" cho weight tied. Ví dụ `{"lm_head.weight": "model.embed_tokens.weight"}` nghĩa là `lm_head.weight` được tie với `embed_tokens.weight`.
- `_supports_sdpa`: True nếu model hỗ trợ SDPA backend. `from_pretrained` check flag này để raise error sớm nếu user request SDPA mà model không hỗ trợ.

## Multiple inheritance

```python
class PreTrainedModel(nn.Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin):
```

- `nn.Module`: PyTorch base.
- `EmbeddingAccessMixin`: cung cấp `get_input_embeddings()`, `set_input_embeddings()`, `tie_weights()`.
- `ModuleUtilsMixin`: utility (extend_attention_mask, get_extended_attention_mask, num_parameters, ...).
- `PushToHubMixin`: `push_to_hub("user/repo")` upload weight + config.
- `PeftAdapterMixin`: hỗ trợ load/save PEFT adapter (LoRA, ...).

Mixin pattern này thay vì single inheritance giúp tách concern. Mỗi mixin handle một loại behavior.

## Subclass cụ thể: hai tầng

Mọi model có pattern hai class:

```python
class LlamaPreTrainedModel(PreTrainedModel):
    """Base class declaring config + flags."""
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            ...


class LlamaModel(LlamaPreTrainedModel):
    """Body model, output last_hidden_state."""
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size)
        ...

    def forward(self, ...):
        ...
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, ...)


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """Task head: causal LM."""
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, ...):
        outputs = self.model(...)
        logits = self.lm_head(outputs.last_hidden_state)
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values, ...)
```

Tách `LlamaPreTrainedModel` chỉ chứa **metadata** (config_class, flags, _init_weights). Tách `LlamaModel` chứa **body** (embed + layers). Tách `LlamaForCausalLM` chứa **task head** (lm_head).

Pattern này cho phép task head khác nhau dùng chung body. `LlamaForSequenceClassification` cũng có `self.model = LlamaModel(config)`, chỉ thêm classification head.

## `post_init()`

```python
class LlamaForCausalLM(...):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(...)
        self.post_init()  # <- important
```

`post_init()` (định nghĩa trong `PreTrainedModel`) làm:

1. Apply `_init_weights` cho mọi module.
2. Tie weights (lm_head.weight = embed_tokens.weight nếu config bảo tie).
3. Apply `_keep_in_fp32_modules` cho stability.

Quên `post_init` -> weight bị init random nhưng không tie -> generate ra junk.

## `config_class` mapping

```python
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
```

Khi `from_pretrained("path")`:

1. Đọc `config.json` -> instantiate `LlamaConfig`.
2. Lookup class model qua `architectures` field hoặc `_auto_class`.
3. Instantiate `LlamaForCausalLM(config)`.
4. Load weight.

Nếu model class không có `config_class` chuẩn (ví dụ class tạm), `from_pretrained` fail validation.

Chương sau ta đi sâu vào modular files.
