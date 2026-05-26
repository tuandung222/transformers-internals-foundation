---
title: AttentionInterface và dispatch
---

# `AttentionInterface` và cơ chế dispatch

Chương trước, ta thấy `LlamaAttention.forward` gọi `ALL_ATTENTION_FUNCTIONS.get_interface(...)`. Chương này mở `modeling_utils.py` để hiểu interface này được xây ra sao, vì sao dùng global dict, và làm thế nào để thêm backend custom.

## Định nghĩa trong `modeling_utils.py`

Tìm tới class `AttentionInterface`:

```python
class GeneralInterface(MutableMapping):
    _global_mapping: dict[str, Callable] = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        if key in self._local_mapping:
            return self._local_mapping[key]
        if key in self._global_mapping:
            return self._global_mapping[key]
        raise KeyError(key)

    def register(self, key: str, value: Callable):
        self._local_mapping[key] = value

    def __contains__(self, key):
        return key in self._local_mapping or key in self._global_mapping
```

`GeneralInterface` là một `MutableMapping` hai tầng. Tầng `_global_mapping` là dict class-level chứa các backend mà thư viện ship sẵn (eager, SDPA, FlashAttention, ...). Tầng `_local_mapping` là dict instance-level mà user có thể register thêm để override hoặc thêm backend mới.

Vì sao tách hai tầng? Để user có thể experiment với backend mới mà không cần monkey-patch dict global. Khi instance bị garbage collect, mọi backend tạm thời biến mất. Class-level dict đảm bảo các backend built-in luôn có sẵn cho mọi instance.

## Subclass `AttentionInterface`

```python
class AttentionInterface(GeneralInterface):
    _global_mapping = {
        "flash_attention_4": flash_attention_forward,
        "flash_attention_3": flash_attention_forward,
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
        "paged|flash_attention_4": paged_attention_forward,
        ...
    }

    def get_interface(self, attn_implementation: str, default: Callable) -> Callable:
        if attn_implementation != "eager" and attn_implementation not in self:
            raise KeyError(f"`{attn_implementation}` is not a valid attention implementation")
        return super().get(attn_implementation, default)
```

Một số quan sát:

1. Tất cả các phiên bản FlashAttention (`_2`, `_3`, `_4`) cùng map tới một function. Function này phía trong gọi đúng kernel theo phiên bản. Tách key giúp config string vẫn rõ ràng.
2. Key `paged|sdpa`, `paged|flash_attention_2`, ... là pattern cho paged attention. Dấu `|` đóng vai trò namespace, không phải toán tử logic. Chương 5 phân tích paged attention.
3. `get_interface` raise nếu key không hợp lệ, **trừ** `"eager"`. Vì sao `"eager"` được trừ ra? Eager là implementation mặc định nằm ngay trong modeling file (`eager_attention_forward`), không cần register. Nếu user chỉ định `"eager"`, function `default` (chính là `eager_attention_forward`) được dùng.

## Singleton global

```python
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()
```

Một instance global. Mọi modeling file import biến này. Khi user muốn register backend custom:

```python
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def my_attention_forward(module, q, k, v, mask, scaling, dropout, **kwargs):
    ...
    return attn_output, attn_weights

ALL_ATTENTION_FUNCTIONS.register("my_backend", my_attention_forward)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation="my_backend")
```

Vì `register` ghi vào `_local_mapping`, nên backend custom vẫn shadow được backend built-in nếu trùng key. Nhưng cẩn thận: vì `ALL_ATTENTION_FUNCTIONS` là singleton global, register cũng có hiệu ứng global.

## Vì sao dispatch bằng dict, không phải if-else?

Câu hỏi thiết kế kinh điển. Nếu viết:

```python
if config._attn_implementation == "sdpa":
    out = sdpa_forward(...)
elif config._attn_implementation == "flash_attention_2":
    out = flash_forward(...)
...
```

Mọi backend mới đều phải sửa file modeling. Có khoảng 200 model trong transformers; thêm một backend phải sửa 200 file. Không scale được.

Với dict global, backend mới chỉ cần một file `integrations/my_backend.py` và một dòng `ALL_ATTENTION_FUNCTIONS._global_mapping["my_backend"] = my_attention_forward`. Modeling file không thay đổi.

## Vì sao function, không phải class?

Backend là **function** (`Callable`), không phải class. Lý do:

1. Stateless: backend không cần lưu state riêng. State (Q/K/V projection, dropout rate, scaling) thuộc về `LlamaAttention`, được pass qua tham số `module` và `**kwargs`.
2. Tránh `nn.Module` overhead: nếu là Module, mỗi layer cần một instance, làm `state_dict` phồng và logic init phức tạp hơn.
3. Dễ test: function thuần tuý dễ unit test (input -> output), không phụ thuộc constructor.

## Backend signature chuẩn

Mọi backend nhận signature giống nhau:

```python
def some_backend_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    return attn_output, attn_weights
```

`module` được pass vào để backend đọc attribute (ví dụ `module.num_key_value_groups`, `module.is_causal`). Mọi thông tin còn lại nằm trong các tensor và `**kwargs`. Backend trả về `(attn_output, attn_weights)`; nếu backend không tính `attn_weights` (như SDPA, FlashAttention), trả về `None` cho phần tử thứ hai.

Chương 4 mở từng backend cụ thể.
