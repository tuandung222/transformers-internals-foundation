---
title: Cheatsheet
---

# Cheatsheet

Tra cứu nhanh các pattern và snippet.

## Cấu trúc một modeling file

```
# 1. Imports
from ...modeling_utils import PreTrainedModel
from ...cache_utils import Cache, DynamicCache
from .configuration_X import XConfig

# 2. Utilities
def rotate_half(x), apply_rotary_pos_emb, eager_attention_forward, ...

# 3. Modules
class XRMSNorm, XRotaryEmbedding, XMLP, XAttention, XDecoderLayer

# 4. Base
class XPreTrainedModel(PreTrainedModel):
    config_class = XConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_flash_attn = True

# 5. Body
class XModel(XPreTrainedModel)

# 6. Task heads
class XForCausalLM(XPreTrainedModel, GenerationMixin)
class XForSequenceClassification(XPreTrainedModel)
```

## Attention dispatch

```python
attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)
attn_output, attn_weights = attention_interface(
    self, q, k, v, attention_mask,
    scaling=self.scaling, dropout=0.0, **kwargs,
)
```

## Backend signature

```python
def my_backend_forward(module, query, key, value, attention_mask,
                       scaling, dropout=0.0, **kwargs):
    ...
    return attn_output, attn_weights
```

## Register backend custom

```python
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
ALL_ATTENTION_FUNCTIONS.register("my_backend", my_backend_forward)
```

## Cache use

```python
# Forward
key, value = past_key_values.update(key_new, value_new, layer_idx)

# Generate-side setup
model.generation_config.cache_implementation = "static"  # hoặc "quantized", "offloaded"
```

## EncoderDecoderCache

```python
past_key_values = EncoderDecoderCache(
    DynamicCache(config=config),  # self attention
    DynamicCache(config=config),  # cross attention
)

# Cross attention check is_updated
if is_cross_attention and past_key_values.is_updated.get(layer_idx):
    k, v = past_key_values.cross_attention_cache.layers[layer_idx].keys, ...values
else:
    k = self.k(encoder_states); v = self.v(encoder_states)
    past_key_values.cross_attention_cache.update(k, v, layer_idx)
    past_key_values.is_updated[layer_idx] = True
```

## Generate options

```python
out = model.generate(
    input_ids,
    max_new_tokens=200,           # cap số token mới
    do_sample=True,                # multinomial sampling
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
    num_beams=1,                   # > 1 cho beam search
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=True,
    cache_implementation="static",
    return_dict_in_generate=True,
    output_scores=False,
)
```

## Compile + StaticCache + Speedup

```python
from transformers import GenerationConfig, CompileConfig

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")

# Warm-up compile
_ = model.generate(input_ids, max_new_tokens=2,
    cache_implementation="static",
    compile_config=CompileConfig(fullgraph=True, mode="reduce-overhead"))

# Fast inference
out = model.generate(input_ids, max_new_tokens=200,
    cache_implementation="static",
    compile_config=CompileConfig(fullgraph=True, mode="reduce-overhead"))
```

## Backend chọn theo case

| Case | Backend |
|---|---|
| Debug, check attn_weights | `eager` |
| Default, đơn giản | `sdpa` |
| Long context, GPU Ampere+ | `flash_attention_2` |
| Sliding window, document mask | `flex_attention` |
| Continuous batching | `paged|flash_attention_2` |

## Cache chọn theo case

| Case | Cache |
|---|---|
| Default inference | `DynamicCache` |
| Compile, low latency | `StaticCache` |
| Long context, RAM hạn chế | `QuantizedCache` (HQQ 4-bit) |
| Cực dài, multi-GPU | `OffloadedCache` |
| Encoder-decoder | `EncoderDecoderCache(Dynamic, Dynamic)` |

## Sample strategy chọn theo task

| Task | Strategy |
|---|---|
| Chat creative | `do_sample=True, top_p=0.9, temperature=0.7` |
| Code generation | `do_sample=True, top_p=0.95, temperature=0.2` |
| Translation, summarization | `num_beams=4, do_sample=False` |
| Greedy QA | `do_sample=False, num_beams=1` |
| Speculative | `assistant_model=draft, do_sample=False` |

## Tied weights khai báo

```python
class XForCausalLM(XPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

## TP plan khai báo

```python
class XPreTrainedModel(PreTrainedModel):
    _tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
    }
```

## `from_pretrained` arguments thường dùng

| Argument | Mục đích |
|---|---|
| `torch_dtype=torch.bfloat16` | Load với dtype này |
| `device_map="auto"` / `"cuda"` | Placement |
| `attn_implementation="sdpa"` | Pin backend attention |
| `quantization_config=BitsAndBytesConfig(load_in_4bit=True)` | Quantize lúc load |
| `tp_plan="auto"` | Apply TP plan declared |
| `trust_remote_code=True` | Allow custom code từ Hub |
| `revision="main"` | Branch/tag hub |
| `weights_only=True` | Safe load |

## Debug warnings phổ biến

| Warning | Nguyên nhân | Fix |
|---|---|---|
| "Some weights were not initialized" | Missing key checkpoint | Check `_keys_to_ignore_on_load_missing` hoặc retrain |
| "Some weights of pretrained model not used" | Unexpected key checkpoint | Check `_keys_to_ignore_on_load_unexpected` hoặc đúng model class |
| "sdpa does not support output_attentions" | Backend không support | Fallback `eager` |
| "Setting pad_token_id = eos_token_id" | Không pad token | Set tokenizer.pad_token = tokenizer.eos_token |
| "Recompiling..." | Shape thay đổi với compile | Dùng StaticCache + fixed batch size |

## Output dataclass thường gặp

| Class | Fields | Khi nào |
|---|---|---|
| `BaseModelOutputWithPast` | last_hidden_state, past_key_values | Body model output |
| `CausalLMOutputWithPast` | loss, logits, past_key_values | ForCausalLM forward |
| `SequenceClassifierOutputWithPast` | loss, logits | Sequence classification |
| `GenerateDecoderOnlyOutput` | sequences, scores, attentions | generate with return_dict_in_generate |
| `GenerateEncoderDecoderOutput` | sequences, scores, encoder_attentions, cross_attentions | encoder-decoder generate |
| `GenerateBeamDecoderOnlyOutput` | sequences, sequences_scores, beam_indices | beam search |

## Quick check model có generate được không

```python
model.can_generate()  # True nếu có lm_head + GenerationMixin
```

## Quick get input/output embedding

```python
embedding = model.get_input_embeddings()  # nn.Embedding
output_emb = model.get_output_embeddings()  # nn.Linear (lm_head)
```

## Force re-tie sau swap embedding

```python
model.set_input_embeddings(new_embedding)
model.tie_weights()  # re-tie nếu config bảo tie
```
