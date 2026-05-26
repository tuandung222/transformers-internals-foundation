---
title: Tied weights và attention flags
---

# Tied weights và attention support flags

Hai topic kỹ thuật của `PreTrainedModel` đáng đào sâu: tied weights (chia sẻ tensor giữa các module) và attention support flags (cờ cho biết model hỗ trợ backend nào).

## Tied weights: ý tưởng

Một số weight được **share** giữa hai module:

- `lm_head.weight` thường share với `embed_tokens.weight`. Embedding matrix `(vocab_size, hidden_size)` cũng dùng làm output projection `(hidden_size, vocab_size)` (transposed).
- Cross-attention K/V projection có thể share với encoder self-attention (hiếm).

Lý do share embedding-output:

1. **Memory**: vocab thường lớn (Llama-3: 128k). Embedding matrix 1.0 GB cho bf16. Tied giúp tiết kiệm.
2. **Generalization**: Share weight tạo ra inductive bias, giúp model tổng quát hơn. Token "the" có embedding gần với hướng output "the".

Tied chỉ khi `config.tie_word_embeddings = True`. Llama-3 8B: False. Llama-3.2 1B: True. Phụ thuộc model.

## `_tied_weights_keys`

Khai báo trên model class:

```python
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

Dict map "target -> source". Khi `tie_weights()` chạy (gọi từ `post_init`), HF set `lm_head.weight = embed_tokens.weight` (chung tensor).

Tại sao cần khai báo? Ba lý do:

1. **Save state_dict**: chỉ save **một** tensor (source), không save tensor target (tránh duplicate). Khi load, restore source rồi tie.
2. **`from_pretrained` validation**: kiểm tra state_dict load có đúng các key bị thiếu (`lm_head.weight` không có trong state_dict là OK nếu tied).
3. **PEFT/LoRA**: adapter biết tensor nào share để apply consistently.

## `tie_weights()` implementation

Trong `PreTrainedModel`:

```python
def tie_weights(self):
    if getattr(self.config, "tie_word_embeddings", True):
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()
        if output_embeddings is not None and input_embeddings is not None:
            output_embeddings.weight = input_embeddings.weight
            if hasattr(output_embeddings, "bias") and output_embeddings.bias is not None:
                output_embeddings.bias.data = nn.functional.pad(
                    output_embeddings.bias.data,
                    (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                    "constant", 0,
                )
```

`get_input_embeddings` và `get_output_embeddings` là method trên model (override trong subclass). Llama:

```python
class LlamaForCausalLM(...):
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def get_output_embeddings(self):
        return self.lm_head
```

`tie_weights` set `lm_head.weight = embed_tokens.weight`. Đây là **assignment** Python, không copy data. Hai module sau đó cùng trỏ tới một tensor object. Update một -> update cái còn lại.

## Validation lúc save

Khi save state_dict, HF check để chỉ ghi tensor một lần:

```python
shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
```

Group tensor theo địa chỉ memory. Nếu hai tensor chia memory (do tie), HF chỉ save một, sau đó tie lại lúc load.

Nếu khai báo `_tied_weights_keys` sai (ví dụ `lm_head.weight` không thực sự tied), save sẽ raise warning vì có shared memory nhưng không khai báo.

## `_keys_to_ignore_on_load_missing`

```python
class LlamaPreTrainedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["model.embed_tokens.weight", "lm_head.weight"]
```

Khi `from_pretrained`, một số key có thể missing (vì tied, vì optional). HF warning user về missing key. Nếu key nằm trong `_keys_to_ignore_on_load_missing`, suppress warning.

Đối ngược: `_keys_to_ignore_on_load_unexpected` cho key thừa trong checkpoint mà model không có (ví dụ legacy buffer).

## Attention support flags

```python
class LlamaPreTrainedModel(PreTrainedModel):
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
```

Mỗi flag báo: model hỗ trợ backend này hay không.

Trong `from_pretrained`, HF check flag khi user request:

```python
def _check_and_enable_sdpa(self, hard_check_only=False):
    if not self._supports_sdpa:
        raise ValueError(
            f"{self.__class__.__name__} does not support an attention implementation through "
            f"torch.nn.functional.scaled_dot_product_attention yet."
        )
```

Vì sao raise sớm, không để runtime fail? Vì user request "tôi muốn dùng SDPA, model load với SDPA mode". Nếu model không hỗ trợ, fail ngay ở load, không phải đợi đến forward.

Một số case model không hỗ trợ:

- Model có position bias kiểu T5 (relative bias). SDPA không hỗ trợ additive bias kiểu T5 hiệu quả.
- Model có custom attention pattern (Flamingo's perceiver). SDPA dispatch không đúng.
- Model legacy chưa được test với SDPA.

## `_supports_cache_class`

```python
_supports_cache_class = True
```

Báo model dùng `Cache` API (Phần 3). Một số model legacy vẫn dùng tuple `past_key_values=((k, v), (k, v), ...)`. Flag này phân biệt để `generate` chọn path đúng.

Tất cả model HF mới đều `_supports_cache_class = True`. Class wrapper backward compat (convert tuple `<->` Cache) chỉ kick in nếu False.

## `_no_split_modules`

```python
_no_split_modules = ["LlamaDecoderLayer"]
```

Cho `device_map="auto"` (Accelerate). Báo `LlamaDecoderLayer` không được split (mỗi layer phải nằm trên một device). Tránh split tensor parameter giữa GPU, gây slow communication.

Khi `device_map="auto"` placement, Accelerate đọc `_no_split_modules` để chọn boundary split. Layer là boundary tự nhiên.

## `_skip_keys_device_placement`

```python
_skip_keys_device_placement = ["past_key_values"]
```

Báo `past_key_values` (cache) không cần move device ở dispatch. Lý do: cache có thể được manage bởi user (offload, ...). HF không tự move.

## `_keep_in_fp32_modules`

```python
_keep_in_fp32_modules = ["lm_head"]
```

Báo `lm_head` luôn ở fp32 dù model load fp16. Vì lm_head output đi qua softmax, fp16 dễ overflow.

`post_init` apply: với mọi module match name pattern, cast weight về fp32.

## Tổng hợp các flag

| Flag | Mục đích |
|---|---|
| `_supports_sdpa` | SDPA backend OK |
| `_supports_flash_attn` | FlashAttention 2/3/4 OK |
| `_supports_flex_attn` | FlexAttention OK |
| `_supports_cache_class` | Cache API (vs tuple legacy) |
| `_no_split_modules` | device_map boundary |
| `_skip_keys_device_placement` | Không auto-move key này |
| `_keep_in_fp32_modules` | Giữ fp32 dù model fp16 |
| `_tied_weights_keys` | Map tied weights |
| `_keys_to_ignore_on_load_missing` | Suppress missing warning |
| `_keys_to_ignore_on_load_unexpected` | Suppress unexpected warning |
| `_keys_to_ignore_on_save` | Skip save |
| `supports_gradient_checkpointing` | Gradient checkpointing OK |
| `_is_stateful` | Model có state riêng (RNN, SSM) |

Khi viết model mới, set các flag đúng. Default phần lớn False (conservative). Bật flag khi đã test backend tương ứng.

Chương sau ta đi vào TP/FSDP plan.
