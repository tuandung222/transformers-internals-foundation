---
title: Modular và modeling files
---

# `modular_*.py` và `modeling_*.py`

Một trong những convention mới (2024+) của HF: tách file model thành hai. `modular_llama.py` là **source of truth** với inheritance; `modeling_llama.py` là **generated** file, flat, không inheritance. Chương này giải thích vì sao và cách hoạt động.

## Bài toán: duplication giữa model

Có ~200 model trong transformers. Nhiều model rất giống nhau (Mistral khác Llama vài chỗ, Qwen khác Llama vài chỗ, ...). Trước đây, mỗi model có file `modeling_*.py` riêng, copy code từ Llama rồi modify. Hậu quả:

1. **Bug fix lan toả khó**: sửa RoPE ở Llama, phải mở 50 model khác kiểm tra.
2. **Inconsistency**: model A đã update theo refactor, model B chưa, hành vi khác.
3. **Onboard khó**: new model đóng góp phải copy 1500 dòng.

Hai giải pháp HF thử:

**Giải pháp 1**: `# Copied from ...` comment. Code marker:

```python
# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        ...
```

`make fix-repo` quét comment này, sync code từ source. Sửa LlamaRMSNorm, MistralRMSNorm sync tự động.

Hạn chế: chỉ sync **identical** code. Không hỗ trợ "Mistral giống Llama nhưng có thêm sliding window".

**Giải pháp 2**: `modular_*.py` với inheritance Python thực. Source file dùng Python `class MistralAttention(LlamaAttention)` để override. Compiler HF generate flat file `modeling_mistral.py`.

## Cấu trúc `modular_*.py`

Ví dụ giả định `modular_mistral.py`:

```python
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)
from .configuration_mistral import MistralConfig


class MistralAttention(LlamaAttention):
    """Inherit Llama, override sliding window support."""
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.sliding_window = config.sliding_window


class MistralDecoderLayer(LlamaDecoderLayer):
    """Inherit, but use MistralAttention instead."""
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = MistralAttention(config, layer_idx)


class MistralModel(LlamaModel):
    pass  # Layer list được override khi build qua MistralConfig + DecoderLayer


class MistralForCausalLM(LlamaForCausalLM):
    pass
```

Sau `make fix-repo`, HF compiler sinh `modeling_mistral.py` là file **flat** (không inheritance), copy tất cả code Llama, apply transformation (rename, override method), sinh `MistralAttention` thừa kế trực tiếp `nn.Module`.

## Quy tắc

1. **Modular file** là source of truth. Sửa modular -> regenerate modeling.
2. **Modeling file** không được sửa tay. Sửa sẽ bị overwrite ở lần `make fix-repo` kế tiếp.
3. **Modular inherit chỉ trong model HF**. Không inherit từ third-party lib (vì compile cần biết source code).
4. **Override method = override toàn bộ method**. Không có super() trong generated file (vì flat). Trong modular, super() được resolve compile-time.

## Vì sao file flat?

Sao không giữ inheritance trong production code?

1. **One model, one file invariant**. User đọc Mistral chỉ cần đọc `modeling_mistral.py`. Không phải mở Llama để hiểu Mistral.
2. **Refactor an toàn**: nếu Llama refactor (đổi tên attribute), Mistral generated với attribute mới. Không có dangling reference.
3. **Performance**: Python attribute lookup trên inheritance chain hơi chậm. Flat class nhanh hơn microsecond.
4. **Backward compat**: nếu user fork Llama, Mistral không bị ảnh hưởng vì code đã được copy.

## Khi nào dùng?

Mọi model mới đóng góp lên HF hiện đều ưu tiên modular. Document hướng dẫn ở `docs/source/en/modular_transformers.md` của repo transformers.

Một số model đã refactor sang modular:

- Gemma (modular dùng Llama base).
- Cohere (modular dùng Llama base).
- Olmo (modular dùng Llama base).
- Persimmon, Falcon, Mistral, Qwen, ...

Llama vẫn là **standalone** (không modular) vì nó là base. Một số model phức tạp (T5 với relative bias, Whisper với conv encoder) cũng standalone.

## Implementation compiler

Compiler nằm ở `utils/modular_model_converter.py`. Workflow:

1. Parse `modular_*.py` bằng AST.
2. Resolve mọi import từ model HF khác (recursively).
3. Inline class parent vào subclass.
4. Rename theo prefix (`Llama` -> `Mistral`).
5. Apply override (method body từ subclass thay method body parent).
6. Dump ra `modeling_*.py`.

Vì compiler text-based, có giới hạn:

- Không support metaclass.
- Không support multiple inheritance phức tạp.
- Một số Python feature (decorator stack, descriptor) khó preserve.

Tuy nhiên, đủ tốt cho 90% use case của model HF.

## Pitfall

**1. Quên `make fix-repo` sau sửa modular**: modeling không update, CI fail.

**2. Sửa modeling trực tiếp**: bị overwrite. Nếu phát hiện model có modular, sửa modular.

**3. Import từ model khác sai path**: compiler không resolve được. Fail compile.

**4. Override method nhưng quên gọi super()-like behavior**: vì generated file flat, không có super(). Phải copy toàn bộ logic.

## Đọc modeling khi không có modular

Một số model (Llama, T5, Whisper, ...) không có modular. Đọc trực tiếp `modeling_*.py` là source of truth. Pattern code vẫn tuân theo Phần 5 Chương 1: imports, utilities, modules, PreTrainedModel base, Body, Task heads.

Khi đọc modeling sinh từ modular, ta vẫn đọc nó như source độc lập. Chỉ khi cần **sửa** mới quan tâm có modular hay không.

## Kết luận chương

Modular giải bài toán duplication không đơn giản. Hiện đang là tương lai của codebase HF. Khi đóng góp model mới, ưu tiên modular. Khi đọc model, modeling file vẫn là first-class.

Chương sau ta đi vào Auto class registry.
