---
title: Self vs cross-attention
---

# So sánh self-attention và cross-attention

Trên giấy, sự khác biệt rất nhỏ: chỉ là nguồn của K và V. Trong code, sự khác biệt lan ra mask, cache, và position bias. Chương này làm rõ.

## Formula song song

Self-attention:

$$Q = h W_Q,\quad K = h W_K,\quad V = h W_V$$

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}} + M_\text{causal}\right) V$$

Cross-attention:

$$Q = h_\text{dec} W_Q,\quad K = h_\text{enc} W_K,\quad V = h_\text{enc} W_V$$

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}} + M_\text{pad}\right) V$$

Khác biệt:

1. K và V được phép chiếu từ `hidden_states` khác. Self: cùng `h`. Cross: K/V từ `h_enc`.
2. Mask. Self decoder thường causal. Cross dùng padding mask (che các vị trí `pad` của source). Không có causal vì decoder Q được phép nhìn mọi K encoder.

## Implementation: cùng class

Nhiều thư viện hiện đại dùng **cùng class** cho cả hai loại attention, phân biệt qua sự có mặt của tham số `key_value_states`. T5 là ví dụ điển hình. Đọc constructor của `T5Attention`:

```python
class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, layer_idx=None):
        super().__init__()
        self.is_decoder = config.is_decoder
        ...
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
```

Linear projection y hệt self-attention. Sự khác biệt nằm ở `forward`. Trong T5:

```python
def forward(self, hidden_states, mask=None, key_value_states=None, ...):
    is_cross_attention = key_value_states is not None
    query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
    ...
    current_states = key_value_states if is_cross_attention else hidden_states
    ...
    key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
    value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
```

Một dòng `current_states = key_value_states if is_cross_attention else hidden_states` quyết định: K/V được tính từ encoder output (cross) hay từ chính decoder hidden (self).

`T5LayerSelfAttention` và `T5LayerCrossAttention` (file `modeling_t5.py`) wrap `T5Attention` với hai cách gọi khác nhau:

```python
class T5LayerSelfAttention(nn.Module):
    def forward(self, hidden_states, ...):
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask, ...)
        ...

class T5LayerCrossAttention(nn.Module):
    def forward(self, hidden_states, key_value_states, ...):
        attention_output = self.EncDecAttention(normed_hidden_states, mask=attention_mask,
                                                 key_value_states=key_value_states, ...)
```

Ở `T5LayerCrossAttention`, `key_value_states` (encoder hidden) được truyền vào. Ở self attention, không có. Cùng class `T5Attention`, hai cách gọi.

## Shape

Self-attention decoder, khi decode token `t`:

- `hidden_states`: `(B, 1, D)` (chỉ token mới)
- `Q`: `(B, num_heads, 1, head_dim)`
- `K, V`: `(B, num_kv_heads, t, head_dim)` sau cache update
- attn_weights: `(B, num_heads, 1, t)`

Cross-attention decoder, mỗi step:

- `hidden_states`: `(B, 1, D)` (decoder token mới)
- `key_value_states`: `(B, T_src, D_enc)` (encoder output, **không đổi** qua các step)
- `Q`: `(B, num_heads, 1, head_dim)`
- `K, V`: `(B, num_kv_heads, T_src, head_dim)` (constant qua các step)
- attn_weights: `(B, num_heads, 1, T_src)`

Quan sát: K và V của cross-attention có `seq_len = T_src` cố định. Nếu prefill encoder một lần, K, V có thể compute một lần và reuse. Đây là motivation của `EncoderDecoderCache.is_updated`, ta sẽ thấy ở chương 4.

## Mask

Self causal decoder: mask `(B, 1, q_len, k_len)` với `mask[i,0,q,k] = 0` nếu `k <= q + offset`, `-inf` nếu `k > q + offset` (offset = past_seen_tokens).

Cross: mask chỉ che padding của source. Shape `(B, 1, 1, T_src)`, `mask[i,0,0,k] = 0` nếu token `k` là valid, `-inf` nếu pad. Không có thông tin causal.

Trong code, HF dùng các helper:

```python
from ...masking_utils import create_bidirectional_mask, create_causal_mask
```

T5Stack gọi `create_bidirectional_mask` cho encoder và cross-attention path, `create_causal_mask` cho self-attention path. Mỗi mask là một tensor additive.

## Position bias đặc thù T5

T5 không dùng RoPE hay sinusoidal. Thay vào đó: **relative position bias** học được. Bias là một tensor `(num_heads, q_len, k_len)` được cộng vào scores **trước softmax**. Bias chỉ học ở layer đầu (`has_relative_attention_bias=True`), các layer sau dùng lại bias đó (truyền qua tham số `position_bias`).

Cross-attention trong T5 **không** dùng relative bias (`has_relative_attention_bias=False`), vì khoảng cách "vị trí" giữa decoder token và encoder token không có ý nghĩa rõ. Bias bị zero ở cross-attention layer.

Tóm lại: hai loại attention dùng chung primitive nhưng khác về nguồn K/V, mask, và cách handle position. Chương sau ta đọc cụ thể `T5Attention.forward`.
