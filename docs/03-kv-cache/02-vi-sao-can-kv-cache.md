---
title: Vì sao cần KV Cache
---

# Vì sao cần KV Cache

Ý tưởng KV Cache rất đơn giản, nhưng tác động sâu rộng. Chương này phân tích bằng số liệu vì sao không có cache thì generation không khả thi cho LLM hiện đại.

## Bài toán: autoregressive decoding

Model muốn sinh ra một sequence dài N token, mỗi step điều kiện trên các token đã sinh trước đó. Mathematically:

$$P(y_1, y_2, \ldots, y_N \mid x) = \prod_{t=1}^{N} P(y_t \mid y_{<t}, x)$$

Để tính `P(y_t | y_{<t}, x)`, ta cần đưa toàn bộ `(x, y_1, ..., y_{t-1})` qua model một lần. Naive implementation: mỗi step decode, gọi `model(input_ids)` với toàn bộ context, lấy logit cuối, sample.

## Cost không có cache

Mỗi step decode `t`, ta đưa `t` token qua model. Mỗi layer attention:

- Linear projection Q, K, V: `3 * t * D²` FLOPs (`D = hidden_size`).
- Q @ K^T: `t² * num_heads * head_dim = t² * D` FLOPs.
- Softmax: `t² * num_heads`.
- @ V: `t² * D`.
- Output projection: `t * D²`.

Tổng FLOPs cho một step: `O(t * D² + t² * D)`. Cộng dồn N step: `sum_{t=1}^N (t * D² + t² * D) = O(N² * D² + N³ * D)`.

Với N = 1000, D = 4096: `1e6 * 16e6 + 1e9 * 4e3 = 1.6e13 + 4e12 = 2e13 FLOPs` chỉ riêng attention. Cho 32 layer: `6.4e14 FLOPs`. Trên A100 (312 TFLOPS bf16), mất hơn 2 giây. Cho 8k token: 64x hơn, tức 2 phút mỗi prompt. Không khả thi.

## Cost có cache

Với KV cache, mỗi step decode chỉ cần đưa **một token mới** qua model. Q của một token. K, V mới của một token được append vào cache. Attention tính trên cache:

- Q, K, V mới: `3 * 1 * D² = 3 D²` FLOPs.
- Q @ K^T (Q shape `(1, T)`, K shape `(T,)`): `T * D` FLOPs.
- @ V: `T * D` FLOPs.
- Output: `D²` FLOPs.

Tổng cho một step: `O(D² + T * D)`. Cộng dồn N step:

$$\sum_{t=1}^N (D^2 + t \cdot D) = O(N \cdot D^2 + N^2 \cdot D)$$

So sánh:

- Không cache: `O(N² * D² + N³ * D)`.
- Có cache: `O(N * D² + N² * D)`.

Tỷ lệ: hơn N lần. Với N = 1000, **1000 lần nhanh hơn**. Đây là không phải optimization nhỏ; đây là khác biệt giữa khả thi và không khả thi.

## Cost memory

KV cache chiếm memory. Với mỗi layer, K và V có shape `(B, num_kv_heads, seq_len, head_dim)`. Total cache size:

```
M_cache = 2 * L * B * H_kv * T * d_h * bytes_per_element
```

trong đó `L` = số layer, `H_kv` = num_kv_heads, `T` = context length, `d_h` = head_dim.

Llama-3 8B: `L=32`, `H_kv=8`, `d_h=128`, bf16 (2 bytes), `B=1`.

| Context | KV Cache size |
|---|---|
| 1k | 134 MB |
| 8k | 1.07 GB |
| 32k | 4.29 GB |
| 128k | 17.18 GB |
| 1M | 134 GB |

Một A100 80GB còn cỡ 64GB sau khi load weight. Cache 128k chiếm 17 GB, còn 47 GB cho activation và scratch. Cache 1M không khả thi trên một GPU. Đây là lý do paged attention và offloading ra đời.

## Cost prefill vs decode

Khi `generate`, hai phase tách biệt:

- **Prefill**: đưa prompt (T_prompt token) qua model một lần. Đây là forward "fat", Q, K, V đều có seq_len = T_prompt. Sau prefill, cache có K/V của T_prompt token.
- **Decode**: từng token mới một lần. Q shape `(B, 1, D)`. K, V mới `(B, 1, D)` per layer, append vào cache.

Prefill **compute-bound**: GPU bận làm matmul lớn. Decode **memory-bound**: GPU phải load entire weight + cache để xử lý một token. Tỷ lệ FLOPs/byte rất thấp.

Đây là vì sao **batch decode** hiệu quả: batching không tăng nhiều FLOPs (vẫn dominate bởi weight load), nhưng tận dụng GPU tốt hơn. Continuous batching (paged attention) là extreme version: batch không cố định, các request có context length khác nhau cùng chạy.

## Khi nào cache **không** hữu ích?

Một số case:

1. **Inference một lần** (không generate iterative): chỉ cần forward một lần, không cần cache.
2. **Training**: cache vô nghĩa vì backward pass cần activation đầy đủ.
3. **Greedy với prompt = 1 token**: prefill bằng decode đầu tiên.

Trong hầu hết các trường hợp practical của LLM serving, cache là essential.

## Kết luận

KV Cache giảm cost attention từ O(N²) xuống O(N) per token, với cost memory tuyến tính theo context length. Chương sau ta đọc interface `CacheLayerMixin` mà mọi cache class kế thừa.
