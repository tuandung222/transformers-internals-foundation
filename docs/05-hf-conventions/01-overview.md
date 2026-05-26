---
title: Tổng quan Phần 5
---

# Phần 5: Convention thiết kế model của HF

Bốn phần trước tập trung vào "vertical" (đào sâu vào một mảng cụ thể như attention, cross-attention, KV cache, generate). Phần 5 là "horizontal": các convention chung mà mọi modeling file của HF tuân theo. Biết các convention này giúp:

1. Đọc bất kỳ file `modeling_*.py` mới đến 10x nhanh hơn.
2. Tự tin khi viết model custom, biết "vị trí" đúng cho từng thứ.
3. Debug nhanh khi `from_pretrained` warning về missing/unexpected key.

## Mục tiêu

Sau Phần 5, bạn:

1. Hiểu `PreTrainedModel` cung cấp gì cho subclass.
2. Phân biệt `PreTrainedConfig` (metadata) và `PreTrainedModel` (weight + logic).
3. Hiểu `modular_*.py` và `modeling_*.py`, cách HF generate code từ modular file.
4. Hiểu Auto class registry: `AutoModel`, `AutoModelForCausalLM`, ... map từ config string đến class.
5. Hiểu mixin cho task heads: `CausalLMOutputWithPast`, generation mixin, ...
6. Biết tied weights, attention support flags, TP/FSDP plans.
7. Đọc `from_pretrained` end-to-end.

## Cấu trúc Phần 5

8 chương:

- Chương 2: `PreTrainedModel` và `PreTrainedConfig`.
- Chương 3: `modular_*.py` vs `modeling_*.py`.
- Chương 4: Auto class registry, `AutoModelForCausalLM.from_pretrained`.
- Chương 5: Generic mixin cho task heads.
- Chương 6: Tied weights và attention support flags.
- Chương 7: TP/FSDP plan declaration trong PreTrainedModel.
- Chương 8: `from_pretrained` flow chi tiết.

## Vì sao học conventions cuối cùng?

Có thể bạn nghĩ: chúng ta nên học conventions **trước**, rồi đào sâu cụ thể? Argument ngược: convention dễ nhớ hơn khi đã thấy chúng cụ thể.

Ví dụ: nói "mọi model có `_supports_sdpa` flag" trừu tượng. Sau khi đã đọc `LlamaAttention` và biết SDPA backend hoạt động ra sao, ta hiểu flag này để làm gì (validate ở `from_pretrained` thay vì lỗi runtime khi forward).

Tương tự, `_tied_weights_keys` chỉ rõ ý nghĩa khi đã biết model có lm_head share weight với embed_tokens.

Đây là phương pháp **bottom-up** (từ cụ thể lên trừu tượng). Chậm hơn nhưng nắm chắc.

## Triết lý thiết kế của HF

Một số nguyên tắc lan toả trong code base:

**1. Một model, một file**. Mọi logic của Llama nằm trong `modeling_llama.py`. Không phân tán. Khi đọc một model, ta đọc một file thôi.

**2. Self-contained class**. `LlamaAttention` đầy đủ thông tin để hiểu, không phụ thuộc nhiều utility xa.

**3. Convention over configuration**. Naming consistent: `q_proj`, `k_proj`, `v_proj`, `o_proj`; `embed_tokens`, `lm_head`; `LlamaModel`, `LlamaForCausalLM`. Đọc một model là quen mọi model.

**4. Composition over inheritance**. Đa thừa kế hạn chế. Wrapper (`AutoModel`) thay vì subclass. Mixin chỉ cho behavior shared (generation, embedding access, push_to_hub).

**5. Modular files cho duplication**. Để tránh phải sửa 200 model khi refactor, có `modular_*.py` từ đó generate `modeling_*.py`. Ta sẽ thấy ở chương 3.

**6. Backward compatibility**. Code base có nhiều legacy. Method deprecate qua warning trước, không break ngay. Pre-trained checkpoint nhiều năm tuổi vẫn load được.

## Cấu trúc đọc

Mọi modeling file của HF có cấu trúc gần giống:

```
# 1. Imports
import torch
from ...modeling_utils import PreTrainedModel
from ...cache_utils import Cache, DynamicCache
from .configuration_llama import LlamaConfig

# 2. Utility functions (RoPE, repeat_kv, ...)
def rotate_half(x): ...
def apply_rotary_pos_emb(q, k, cos, sin): ...
def eager_attention_forward(...): ...

# 3. Module classes
class LlamaRMSNorm(nn.Module): ...
class LlamaRotaryEmbedding(nn.Module): ...
class LlamaMLP(nn.Module): ...
class LlamaAttention(nn.Module): ...
class LlamaDecoderLayer(nn.Module): ...

# 4. Base PreTrainedModel
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_flash_attn = True
    ...

# 5. Body model
class LlamaModel(LlamaPreTrainedModel):
    def forward(...): ...

# 6. Task-specific heads
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

class LlamaForSequenceClassification(LlamaPreTrainedModel): ...
class LlamaForTokenClassification(LlamaPreTrainedModel): ...
class LlamaForQuestionAnswering(LlamaPreTrainedModel): ...
```

Khi mở model mới, ta scan theo cấu trúc này. Skip phần đã quen (RoPE, RMSNorm), tập trung vào phần đặc thù (attention variant, custom head).

Sẵn sàng, ta sang chương 2.
