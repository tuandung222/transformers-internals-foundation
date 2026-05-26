---
title: Tổng quan Phần 1
---

# Phần 1: Attention internals

Attention là cốt lõi của Transformer, và cũng là phần thư viện trải qua nhiều iteration nhất. Từ implementation manual (eager) sang `F.scaled_dot_product_attention` của PyTorch, rồi FlashAttention 2, 3, 4, rồi FlexAttention, rồi paged attention. Mỗi backend có ưu điểm riêng. HuggingFace transformers đóng gói tất cả bằng một interface dispatch chung.

## Mục tiêu

Sau Phần 1, bạn có thể:

1. Đọc `LlamaAttention.forward` từng dòng và giải thích vì sao đoạn code được viết theo cách đó.
2. Hiểu `AttentionInterface` global dispatch: vì sao nó là dict thay vì if-else, làm sao để register backend mới.
3. So sánh trade-off giữa eager, SDPA, FlashAttention, FlexAttention: kernel, performance, support level.
4. Biết khi nào nên dùng paged attention (continuous batching) và cấu trúc của nó.

## Cấu trúc Phần 1

Phần 1 có 6 chương:

- Chương 2 (`02-llama-attention-walkthrough`) mở `modeling_llama.py` và đi từng dòng của `LlamaAttention.forward`. Đây là chương dài nhất, đặt nền cho mọi thứ sau.
- Chương 3 (`03-attention-interface-va-dispatch`) phân tích `AttentionInterface` trong `modeling_utils.py`. Vì sao là dict global, cách user thêm backend custom, mechanism `get_interface`.
- Chương 4 (`04-eager-vs-sdpa-vs-flash`) so sánh ba backend chính. Eager để debug, SDPA mặc định cho hầu hết user, FlashAttention cho production. Mỗi backend có một file trong `integrations/`, ta đọc kỹ.
- Chương 5 (`05-paged-attention`) giới thiệu paged attention cho continuous batching. Đây là kỹ thuật của vLLM, đã được HuggingFace tích hợp với các function `paged_attention_forward`, `flash_paged`, `sdpa_paged`, `eager_paged`.
- Chương 6 (`06-tong-ket-attention`) tổng kết các quyết định thiết kế và pitfall.

## Cốt lõi: AttentionInterface

Trong `modeling_utils.py`:

```python
class AttentionInterface(GeneralInterface):
    _global_mapping = {
        "flash_attention_4": flash_attention_forward,
        "flash_attention_3": flash_attention_forward,
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
        "paged|flash_attention_4": paged_attention_forward,
        "paged|flash_attention_3": paged_attention_forward,
        "paged|flash_attention_2": paged_attention_forward,
        "paged|sdpa": sdpa_attention_paged_forward,
        "paged|eager": eager_paged_attention_forward,
    }

    def get_interface(self, attn_implementation: str, default: Callable) -> Callable:
        if attn_implementation != "eager" and attn_implementation not in self:
            raise KeyError(f"`{attn_implementation}` is not a valid attention implementation")
        return super().get(attn_implementation, default)

ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()
```

Trong `LlamaAttention.forward`, dispatch như sau:

```python
attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)

attn_output, attn_weights = attention_interface(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=...,
    scaling=self.scaling,
    **kwargs,
)
```

Một dòng `config._attn_implementation = "sdpa"` đủ để mọi `LlamaAttention` chuyển sang dùng SDPA. Không sửa modeling file.

## Vì sao dispatch quan trọng

Trước version 4.45, mỗi model có nhiều class attention: `LlamaAttention`, `LlamaSdpaAttention`, `LlamaFlashAttention2`. User phải config đúng class. Khi thêm backend mới, phải thêm class mới vào mọi modeling file.

Sau version 4.45, chỉ một class `LlamaAttention`. Dispatch qua `AttentionInterface`. Thêm backend chỉ cần thêm function vào `_global_mapping`. Đơn giản hóa cực lớn.

Đây là một ví dụ điển hình của decision design tốt: chuyển từ inheritance sang composition (dict dispatch), thư viện sạch hơn và dễ extend.

## Ngữ cảnh PyTorch

Để hiểu sự khác biệt giữa các backend, bạn cần biết:

- **Eager**: implementation Python thuần, từng phép `matmul`, `softmax` riêng lẻ. Chậm nhưng dễ debug và hỗ trợ mọi feature (output_attentions, custom mask).
- **SDPA** (`scaled_dot_product_attention`): PyTorch 2.0 native, tự chọn kernel tốt nhất (math, memory_efficient, flash). Mặc định cho hầu hết user.
- **FlashAttention** (Dao-AILab): kernel CUDA tối ưu, không materialize attention matrix, tiết kiệm memory cho sequence dài. v2 (Ampere), v3 (Hopper), v4 (B100).
- **FlexAttention** (PyTorch 2.5+): cho phép viết custom attention pattern (sliding window, document masking, ...) mà vẫn được compile thành kernel hiệu quả.

Chương 4 đi vào chi tiết từng backend.

## Một chú ý về GQA

Llama-3, Mistral, Qwen dùng GQA (Grouped Query Attention): K/V có ít head hơn Q. Trong file source:

```python
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
```

Khi tính attention, K và V phải được repeat để khớp số head Q:

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(...)
    return hidden_states.reshape(...)
```

Eager attention làm `repeat_kv` manually. SDPA mới (PyTorch 2.5+) hỗ trợ GQA natively qua flag `enable_gqa=True`, không cần repeat. Chương 4 phân tích.

Sẵn sàng, ta mở `modeling_llama.py`.
