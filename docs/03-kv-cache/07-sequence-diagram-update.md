---
title: Sequence diagram update flow
---

# Sequence diagram: một step decode

Chương cuối Phần 3 tổng hợp: theo dõi flow một step decode end-to-end, từ `generate` xuống `Cache.update`. Mục tiêu: vẽ ra sequence diagram tinh thần, chốt lại các invariant đã học.

## Setup

Giả sử model Llama-3 8B, dùng `DynamicCache`, eager backend (cho dễ đọc), batch=1, đang ở step decode thứ `t`.

State tại bắt đầu step `t`:

- `past_key_values`: `DynamicCache` chứa K/V của `t-1` token trước đó cho 32 layer.
- `input_ids`: tensor shape `(1, 1)`, là token đã sample ở step `t-1`.
- `attention_mask`: tensor shape `(1, t)`, toàn 1 (giả sử không pad).
- `position_ids`: tensor shape `(1, 1)` chứa `[t-1]`.

## Bước 1: `generate` gọi `model(...)`

```python
outputs = self(
    input_ids=input_ids,
    past_key_values=past_key_values,
    attention_mask=attention_mask,
    position_ids=position_ids,
    use_cache=True,
    cache_position=cache_position,
)
```

`cache_position` là `tensor([t-1])`, vị trí trong cache của token mới.

## Bước 2: `LlamaModel.forward` lặp qua layer

```python
hidden_states = self.embed_tokens(input_ids)  # (1, 1, 4096)
position_embeddings = self.rotary_emb(hidden_states, position_ids)  # cos, sin

for layer in self.layers:
    hidden_states = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
        past_key_values=past_key_values,
        ...
    )
```

Quan sát: `position_embeddings` (cos, sin) compute một lần ở model level, pass xuống mọi layer. `past_key_values` là cùng object, mọi layer share.

## Bước 3: layer `k` gọi attention

`LlamaDecoderLayer.forward` (đã skip residual + layernorm cho gọn):

```python
hidden_states, _ = self.self_attn(
    hidden_states=normed_hidden_states,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
)
```

## Bước 4: `LlamaAttention.forward`

```python
input_shape = hidden_states.shape[:-1]  # (1, 1)
hidden_shape = (*input_shape, -1, self.head_dim)

query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
# Q shape: (1, 32, 1, 128)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
# K shape: (1, 8, 1, 128)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

Sau RoPE, Q và K mới đã rotate theo position `t-1`.

## Bước 5: `past_key_values.update`

```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

Inside `DynamicCache.update`:

```python
def update(self, key_states, value_states, layer_idx, *args, **kwargs):
    if self.layer_class_to_replicate is not None:
        while len(self.layers) <= layer_idx:
            self.layers.append(self.layer_class_to_replicate())

    keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)
    return keys, values
```

Layer-level update (`DynamicLayer.update`):

```python
def update(self, key_states, value_states, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)

    self.keys = torch.cat([self.keys, key_states], dim=-2)
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values
```

Trước update: `self.keys` shape `(1, 8, t-1, 128)`. Sau update: shape `(1, 8, t, 128)`. K mới được append.

Return `(self.keys, self.values)`, là tensor full chứa cả lịch sử + mới.

## Bước 6: dispatch backend

```python
attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)
attn_output, attn_weights = attention_interface(
    self, query_states, key_states, value_states, attention_mask,
    scaling=self.scaling, dropout=0.0,
)
```

`key_states` ở đây là tensor return từ `update`, shape `(1, 8, t, 128)`. Đầy đủ history.

## Bước 7: `eager_attention_forward`

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)  # (1, 32, t, 128)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # attn_weights shape: (1, 32, 1, t)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask  # broadcast

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)  # (1, 32, 1, 128)
    attn_output = attn_output.transpose(1, 2).contiguous()  # (1, 1, 32, 128)
    return attn_output, attn_weights
```

Q nhỏ `(1, 32, 1, 128)` cross với K lớn `(1, 32, t, 128)` ra attn_weights `(1, 32, 1, t)`. Đây là decode "skinny": Q chỉ 1 row, K nhiều cột.

## Bước 8: output projection + return

```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # (1, 1, 4096)
attn_output = self.o_proj(attn_output)
return attn_output, attn_weights
```

Hết. Output `(1, 1, 4096)` đi qua FFN của layer, residual, layernorm, sau đó vào layer kế tiếp.

## Bước 9: lm_head và sample

Sau khi qua 32 layer, hidden states `(1, 1, 4096)` qua `lm_head` ra `(1, 1, vocab_size)`. `generate` sample logit, ra token mới. Step `t+1` bắt đầu.

## Sequence diagram tổng

```
generate -> model.forward -> for layer in layers:
    LlamaDecoderLayer -> self_attn (LlamaAttention)
        -> q_proj, k_proj, v_proj
        -> apply_rotary_pos_emb
        -> past_key_values.update(k_new, v_new, layer_idx)
            -> DynamicLayer.update -> torch.cat (append)
        -> attention_interface (eager/sdpa/flash)
            -> repeat_kv (if GQA)
            -> matmul Q K^T
            -> add mask
            -> softmax
            -> matmul V
        -> o_proj
    -> FFN, residual, norm
-> lm_head
-> sample
```

## Các invariant đã thấy

1. **Cache lưu K/V đã rotate**: RoPE áp dụng trước `update`. Không re-rotate cache cũ.
2. **`update` return full**: tensor return chứa cả history, attention dùng để compute.
3. **Layer indep**: mỗi layer có cache riêng (`cache.layers[layer_idx]`), update độc lập.
4. **One-token Q**: ở decode, Q luôn shape `(B, num_heads, 1, head_dim)`. K, V grow.
5. **Position embed pre-computed**: `cos`, `sin` cho token mới được compute ở model level, share xuống layer.

## Cầu nối sang Phần 4

Bạn đã thấy `generate` gọi `model.forward` mỗi step. Phần 4 đi vào `generate`: nó orchestrate thế nào, sample strategy hoạt động ra sao, logits processor và stopping criteria là gì. Đây là phần cao nhất của stack inference.
