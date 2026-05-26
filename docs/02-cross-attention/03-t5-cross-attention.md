---
title: Walkthrough T5 cross-attention
---

# Walkthrough cross-attention trong T5

Ta mở `src/transformers/models/t5/modeling_t5.py` và đọc `T5Attention.forward` (khoảng line 240). Đây là class được dùng cho **cả** self và cross attention. Ta tập trung vào code path cross-attention.

## Đầu vào

```python
def forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_values=None,
    output_attentions=False,
    **kwargs,
):
```

- `hidden_states`: decoder hidden, shape `(B, T_q, D)`.
- `mask`: additive mask cho attention. Cross attention: `(B, 1, 1, T_src)`.
- `key_value_states`: encoder output, shape `(B, T_src, D)`. **Tham số then chốt** phân biệt self/cross.
- `past_key_values`: cache (loại `EncoderDecoderCache` khi model là encoder-decoder).

## Phân loại self vs cross

```python
is_cross_attention = key_value_states is not None

query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
```

Query luôn được tính từ `hidden_states` (decoder), không phụ thuộc cross/self. Còn K/V phụ thuộc.

## Chọn cache đúng

```python
is_updated = False
if isinstance(past_key_values, EncoderDecoderCache):
    is_updated = past_key_values.is_updated.get(self.layer_idx)
    if is_cross_attention:
        curr_past_key_values = past_key_values.cross_attention_cache
    else:
        curr_past_key_values = past_key_values.self_attention_cache
else:
    curr_past_key_values = past_key_values
```

`EncoderDecoderCache` chứa hai sub-cache: `self_attention_cache` và `cross_attention_cache`. Cross-attention path đọc `cross_attention_cache`, self-attention path đọc `self_attention_cache`. Cả hai đều là `DynamicCache` bên trong, nhưng được tách biệt để tránh trộn K/V của hai loại.

`is_updated` là dict `{layer_idx: bool}`. Đánh dấu rằng layer này đã có K/V cross encode rồi, ở các step decoder sau không cần tính lại.

## Reuse hoặc tính K/V mới

```python
current_states = key_value_states if is_cross_attention else hidden_states
if is_cross_attention and past_key_values is not None and is_updated:
    # reuse k,v, cross_attentions
    key_states = curr_past_key_values.layers[self.layer_idx].keys
    value_states = curr_past_key_values.layers[self.layer_idx].values
else:
    kv_shape = (*current_states.shape[:-1], -1, self.key_value_proj_dim)
    key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
    value_states = self.v(current_states).view(kv_shape).transpose(1, 2)

    if past_key_values is not None:
        key_states, value_states = curr_past_key_values.update(key_states, value_states, self.layer_idx)
        if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
            past_key_values.is_updated[self.layer_idx] = True
```

Đây là chỗ tinh tế nhất của cross-attention. Hai code path:

**Path 1: đã có K/V cross attention (đã update từ step trước).** Đọc thẳng `keys` và `values` từ cache. Không gọi `self.k`, `self.v`. Tiết kiệm hai matmul mỗi step decoder, mỗi layer. Với 32 layer, decode 100 token, ta tiết kiệm `32 * 100 * 2 = 6400` lần matmul.

**Path 2: chưa update (step đầu tiên decode).** Project K/V từ `current_states` (chính là `key_value_states = encoder_output`). Sau đó `curr_past_key_values.update(...)` ghi vào cache, và set `is_updated[layer_idx] = True`.

Ở các step sau, flag `is_updated` đảm bảo ta vào path 1, reuse K/V.

So sánh với self-attention: K/V mới mỗi step (vì decoder token mới mỗi step), nên luôn rơi vào path 2 và append vào cache.

## Tính scores

```python
scores = torch.matmul(query_states, key_states.transpose(3, 2))
```

Phép `Q @ K^T`. Shape `(B, num_heads, T_q, T_kv)`. Với cross: `T_kv = T_src`. Với self: `T_kv = T_q + past_seen_tokens`.

## Position bias

```python
if position_bias is None:
    key_length = key_states.shape[-2]
    if not self.has_relative_attention_bias:
        position_bias = torch.zeros(
            (1, query_states.shape[1], input_shape[1], key_length), device=scores.device, dtype=scores.dtype
        )
    else:
        position_bias = self.compute_bias(input_shape[1], key_length, ...)

    if mask is not None:
        causal_mask = mask[:, :, :, : key_states.shape[-2]]
        position_bias = position_bias + causal_mask

position_bias_masked = position_bias
scores += position_bias_masked
```

Vài quan sát:

1. Cross-attention layer có `has_relative_attention_bias = False`, nên `position_bias` là tensor zero. Không có bias.
2. Mask được merge **vào** `position_bias` trước khi cộng vào scores. Cách này tránh phải pass mask riêng tới mọi layer dưới (tiết kiệm bộ nhớ).
3. `mask[:, :, :, : key_states.shape[-2]]` slice mask đúng `T_kv`. Cần thiết vì self attention với cache, `T_kv` thay đổi theo step.

## Softmax và output

```python
attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

attn_output = torch.matmul(attn_weights, value_states)
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(*input_shape, -1)
attn_output = self.o(attn_output)
```

Phần này giống self-attention: softmax (float32 cho stability), dropout, matmul V, reshape, output projection.

## So sánh với LlamaAttention

Bạn có thể thắc mắc: vì sao `T5Attention` không dùng `AttentionInterface` dispatch như `LlamaAttention`? Lý do lịch sử: T5 viết ra trước cuộc refactor 4.45, có vài đặc thù (relative position bias, layer-wise reuse bias) khó fit signature backend chung. Trong version mới của HF, một số model encoder-decoder đã được refactor. Nhưng T5 vẫn dùng eager-style.

Bài học chung: cross-attention không yêu cầu pattern khác biệt cơ bản về backend. Mọi backend (eager, SDPA, Flash) đều support nếu signature K/V được pass đúng. Chỉ logic cache và mask thay đổi.

Chương kế tiếp đi vào `EncoderDecoderCache` chi tiết.
