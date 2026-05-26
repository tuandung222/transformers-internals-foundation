---
title: Walkthrough LlamaAttention
---

# Walkthrough `LlamaAttention.forward`

Chương này đi từng dòng của `LlamaAttention.forward` trong `src/transformers/models/llama/modeling_llama.py`. Trước khi đọc backend, ta cần hiểu cái mà backend được gọi từ đâu, nhận input gì, và output gì.

## Constructor: bốn linear projection

```python
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
```

Bốn projection: `q_proj`, `k_proj`, `v_proj`, `o_proj`. Chú ý hai điểm. Thứ nhất, `q_proj` ra `num_attention_heads * head_dim`, trong khi `k_proj` và `v_proj` ra `num_key_value_heads * head_dim`. Llama-3 dùng GQA với `num_attention_heads = 32` và `num_key_value_heads = 8`, nghĩa là Q có 32 head còn K/V chỉ có 8 head. `num_key_value_groups = 4` báo: mỗi K/V head phục vụ 4 Q head.

Thứ hai, `is_causal = True` là attribute lớp, không phải config. Một số backend (SDPA, FlashAttention) dùng flag này để chọn mask path. Nếu một subclass cross-attention, attribute này sẽ được override thành `False`.

## Reshape và transpose

```python
def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

`hidden_states` có shape `(batch, seq_len, hidden_size)`. Sau `q_proj`, ta được `(batch, seq_len, num_heads * head_dim)`. Bước `.view(hidden_shape)` reshape thành `(batch, seq_len, num_heads, head_dim)`. Cuối cùng `.transpose(1, 2)` đổi sang `(batch, num_heads, seq_len, head_dim)`. Đây là layout chuẩn cho attention, vì các tensor operation sau sẽ xử lý batch của các head độc lập.

`-1` trong `hidden_shape` cho phép cùng một dòng code chạy được cả cho Q (32 head) và K/V (8 head). Đây là tinh tế nhỏ giúp tránh ba `view` khác nhau.

## Áp dụng RoPE

```python
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

`position_embeddings` không được tính trong `LlamaAttention` mà được tính một lần ở `LlamaModel` và truyền vào mọi layer. Đây là một quyết định performance quan trọng: với 32 layer, tính RoPE 32 lần là lãng phí; tính một lần và cache giúp tiết kiệm khoảng vài phần trăm thời gian forward.

`apply_rotary_pos_emb` rotate Q và K theo `cos`, `sin` của vị trí từng token. Quan trọng: V không bị rotate. RoPE chỉ ảnh hưởng tới attention score (dot product Q-K), không ảnh hưởng giá trị được aggregate.

## Cache update

```python
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

Nếu có cache (inference mode), K và V mới được append vào cache, và `update` trả về toàn bộ K, V của layer này (bao gồm cả phần cũ). `layer_idx` chỉ định slot cache của layer. Phần 3 sẽ đi sâu vào `Cache` interface.

Lưu ý: K/V được update **sau** khi áp RoPE. Điều này nghĩa là cache chứa các vector đã rotate. Khi prefill xong, mọi K mới chỉ cần rotate theo vị trí của nó rồi append, không phải re-rotate K cũ.

## Dispatch attention backend

```python
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states, key_states, value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
```

Toàn bộ tính toán attention được delegate cho function nằm trong `ALL_ATTENTION_FUNCTIONS`. `self` (module) được pass vào để function backend đọc các attribute như `num_key_value_groups`, `is_causal`, `training`. `kwargs` chứa các flag như `output_attentions`, `sliding_window`, ...

Một câu hỏi tự nhiên: vì sao không gọi method? Câu trả lời nằm ở phần 5: HF muốn dispatch theo config string, không theo class hierarchy, để giảm số class trong modeling file. Một function rời, một dict global. Đơn giản, nhanh, và mỗi backend kernel có thể được test độc lập.

## Output projection

```python
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
```

Sau backend, `attn_output` có shape `(batch, seq_len, num_heads, head_dim)` (đã transpose lại bên trong backend). Ta reshape về `(batch, seq_len, hidden_size)` rồi qua `o_proj` để được output cuối cùng.

`.contiguous()` là cần thiết vì `transpose` không đảm bảo memory contiguous, mà `o_proj` (matmul) thường mong đợi contiguous tensor để tối ưu kernel.

## Tổng kết các invariant

Một số invariant bạn cần nhớ khi đọc các backend ở chương sau:

1. Q có `num_attention_heads`, K/V có `num_key_value_heads`. Nếu khác nhau, backend phải `repeat_kv` hoặc dùng GQA-aware kernel.
2. Q và K đã được rotate trước khi vào backend. V không bị rotate.
3. Layout là `(batch, num_heads, seq_len, head_dim)` khi vào backend.
4. `attention_mask` được pass nguyên si. Backend tự quyết định xử lý thế nào (additive mask cho eager, boolean mask cho SDPA, ...).
5. `is_causal` là attribute module, không nằm trong `attention_mask`.

Với nền tảng này, chương 3 mở `AttentionInterface` để hiểu cơ chế dispatch.
