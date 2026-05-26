---
title: EncoderDecoderCache
---

# `EncoderDecoderCache` chi tiết

Chương trước, ta thấy `T5Attention` đọc `past_key_values.is_updated` và `past_key_values.cross_attention_cache`. Chương này mở `cache_utils.py` để hiểu class này.

## Định nghĩa

Trong `src/transformers/cache_utils.py`:

```python
class EncoderDecoderCache(Cache):
    def __init__(self, *caches) -> None:
        if len(caches) == 2:
            if not isinstance(caches[0], Cache) or not isinstance(caches[1], Cache):
                raise TypeError(...)
            self.self_attention_cache = caches[0]
            self.cross_attention_cache = caches[1]
        ...

        self.is_updated = {}
        for layer_idx in range(len(self.cross_attention_cache)):
            self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)
```

Một `EncoderDecoderCache` đóng gói hai `Cache` riêng:

- `self_attention_cache`: K/V của self-attention decoder, đầy lên qua từng step decoding (như `DynamicCache` thường).
- `cross_attention_cache`: K/V của cross-attention, được fill **một lần** ở step đầu tiên và reuse mãi.

Cộng với một dict `is_updated`: cờ đánh dấu mỗi layer đã có K/V cross attention chưa.

## Vì sao tách hai cache?

Tại sao không một `DynamicCache` chung? Vài lý do:

1. **Semantic khác**: K/V self thay đổi theo decode step. K/V cross không đổi. Trộn lẫn dễ nhầm.
2. **Length khác**: K/V cross có length `T_src` (cố định). K/V self có length tăng dần (`t` step). Lưu chung trong một tensor không khớp shape.
3. **Reorder cho beam search**: khi beam search reorder, cả hai cache phải reorder theo cùng `beam_idx`. Tách biệt cho phép gọi `reorder_cache` trên từng cache.
4. **Reset độc lập**: trong một số use case (như Whisper khi xử lý chunk audio tiếp theo), self_attention_cache cần reset trong khi cross_attention_cache reset độc lập theo audio chunk mới.

## Flag `is_updated`

```python
self.is_updated = {}
for layer_idx in range(len(self.cross_attention_cache)):
    self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)
```

Khởi tạo: nếu cache đã có dữ liệu sẵn (ví dụ user pass cache đã prefill), set `True`. Nếu cache rỗng, `False`.

Trong `T5Attention.forward`:

```python
if isinstance(past_key_values, EncoderDecoderCache):
    is_updated = past_key_values.is_updated.get(self.layer_idx)
    ...
if is_cross_attention and past_key_values is not None and is_updated:
    # Path 1: reuse
    key_states = curr_past_key_values.layers[self.layer_idx].keys
    value_states = curr_past_key_values.layers[self.layer_idx].values
else:
    # Path 2: compute và update
    ...
    past_key_values.is_updated[self.layer_idx] = True
```

Step đầu (`is_updated[i] = False`): vào path 2, compute K/V cross và set flag `True`.

Step sau (`is_updated[i] = True`): vào path 1, reuse.

Đơn giản. Nhưng hiệu quả: tiết kiệm `2 * num_layers * decode_steps` matmul.

## Các method khác

```python
def __len__(self):
    return len(self.self_attention_cache)

def get_seq_length(self, layer_idx: int = 0) -> int:
    return self.self_attention_cache.get_seq_length(layer_idx)
```

Length của cache (số layer hoặc seq_length) dựa trên self_attention_cache. Quan trọng vì các logic decoding khác (mask, position_id) dựa trên seq_length của self attention, không phải cross.

```python
def reset(self):
    self.self_attention_cache.reset()
    self.cross_attention_cache.reset()
    for layer_idx in self.is_updated:
        self.is_updated[layer_idx] = False
```

Reset cả hai cache và clear flag.

```python
def reorder_cache(self, beam_idx: torch.LongTensor):
    self.self_attention_cache.reorder_cache(beam_idx)
    self.cross_attention_cache.reorder_cache(beam_idx)
```

Beam search reorder cả hai. Một subtlety: khi reorder, cross attention K/V cho mỗi beam phải khớp với decoder state của beam đó. Vì K/V cross đến từ encoder, và encoder cũng đã chạy cho mỗi beam (hoặc broadcast), `reorder_cache` đảm bảo consistency.

## Khởi tạo trong forward decoder

Trong `T5Stack.forward` (hoặc tương đương Bart, mBart):

```python
if self.is_decoder:
    if use_cache and past_key_values is None:
        if self.config.is_encoder_decoder:
            past_key_values = EncoderDecoderCache(
                DynamicCache(config=self.config), DynamicCache(config=self.config)
            )
        else:
            past_key_values = DynamicCache(config=self.config)
```

Quan sát: model encoder-decoder (`is_encoder_decoder=True`) khởi tạo `EncoderDecoderCache` với hai `DynamicCache` rỗng. Model decoder-only chỉ cần một `DynamicCache`.

Phần khởi tạo này được thực hiện **tự động** trong `T5Stack.forward`. User chỉ cần truyền `use_cache=True`, không cần manual.

## Use case: Whisper

Whisper là speech-to-text encoder-decoder. Encoder lấy mel spectrogram chunk 30s. Decoder generate text. Mỗi audio chunk:

1. Encoder run một lần, ra `encoder_hidden_states`.
2. Decoder generate token by token, dùng `EncoderDecoderCache`. Sau token đầu tiên, `is_updated[i] = True` cho mọi layer, cross-attention K/V reuse mãi.
3. Nếu xử lý audio chunk tiếp theo, encoder run lại, cross-attention cache phải reset.

Mọi logic này được Whisper generation script handle tự động khi pass `past_key_values=None` cho chunk mới.

Chương kế tiếp đi sâu hơn vào Whisper và Vision Encoder-Decoder.
