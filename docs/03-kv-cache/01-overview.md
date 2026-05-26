---
title: Tổng quan Phần 3
---

# Phần 3: KV Cache

KV Cache là kỹ thuật then chốt giúp autoregressive generation nhanh. Không có nó, mỗi step decode phải tính lại attention với toàn bộ context, dẫn tới complexity O(N²) cho mỗi token. Với cache, mỗi token decode chỉ tốn O(N) cho attention plus O(D²) cho linear projection. Đây là lý do generation của LLM nhanh được như ngày nay.

Phần này đi sâu vào cách HF tổ chức class cache, từ `CacheLayerMixin` (interface), `DynamicCache` (mặc định), `StaticCache` (cho `torch.compile`), `QuantizedCache` (tiết kiệm memory), tới `OffloadedCache` (offload CPU).

## Mục tiêu

Sau Phần 3, bạn:

1. Giải thích tại sao KV Cache cần thiết bằng cách so sánh complexity có/không cache.
2. Đọc `CacheLayerMixin` và hiểu interface `update`, `get_seq_length`, `get_mask_sizes`.
3. So sánh `DynamicCache` (grow `torch.cat`) và `StaticCache` (preallocate, in-place mutate).
4. Biết khi nào dùng quantized cache (2-bit, 4-bit) và offloaded cache (CPU RAM).
5. Vẽ được sequence diagram của một `update` call: nhận K/V mới, append/write, return tensor cho attention backend.

## Cấu trúc Phần 3

7 chương:

- Chương 2: Vì sao cần KV Cache. So sánh chi phí.
- Chương 3: `CacheLayerMixin` interface và pattern subclass.
- Chương 4: `DynamicCache` cho generation thông thường.
- Chương 5: `StaticCache` cho `torch.compile` và `torch.export`.
- Chương 6: `QuantizedCache` (HQQ, quanto) và `OffloadedCache`.
- Chương 7: Sequence diagram chi tiết flow update của cache xuyên qua các backend attention.

## Mệnh đề trung tâm

Mỗi layer attention tự nhiên sinh ra một cặp K, V tensor cho mỗi token. Nếu ta lưu lại, decode token tiếp theo chỉ cần Q của một token mới `(B, 1, D)`. Attention chỉ tốn `Q @ K^T` shape `(B, num_heads, 1, T+1)`, không phải `(B, num_heads, T+1, T+1)`.

Câu hỏi tiếp theo: lưu ở đâu, theo dạng nào? Câu trả lời định hình toàn bộ chương Phần 3.

## Các trade-off

| Cache type | Memory | Speed | Compile-friendly | Use case |
|---|---|---|---|---|
| DynamicCache | grow theo seq_len | nhanh (cat reallocation) | không (shape dynamic) | inference thông thường |
| StaticCache | preallocate max_len | nhanh nhất (in-place) | có | torch.compile, export |
| SlidingWindowCache | bounded sliding window | nhanh | có (per layer) | sliding window model (Mistral) |
| QuantizedCache | giảm 2-4x | hơi chậm (quant overhead) | không | long context, RAM hạn chế |
| OffloadedCache | CPU RAM | rất chậm (prefetch giúp) | một phần | extreme long context |
| Paged variant | bounded per-page | rất nhanh, complex | có (vLLM kernel) | continuous batching |

Hai chiều cần cân: memory (chiều cache lớn lên, GPU hết VRAM) và speed (đọc/ghi cache có thể là bottleneck nếu dispatch không tối ưu).

## Quan sát số liệu

Llama-3 8B, batch=1, sequence=8k, bf16:

- Memory cho parameters: ~16 GB.
- Memory cho KV cache: `2 * 32 layer * 8 num_kv_heads * 8192 * 128 head_dim * 2 bytes = 1.07 GB`.

KV cache với 8k context không đáng kể so với weight. Nhưng với 128k context: `2 * 32 * 8 * 131072 * 128 * 2 bytes = 17.1 GB`. **Vượt qua kích thước weight**. Đây là lý do KV cache trở thành bottleneck cho long context.

Quantize K/V xuống 4-bit: tiết kiệm 4x. 17 GB còn 4.3 GB. Offload K/V của các layer xa ra CPU: gần như không tốn VRAM nhưng chậm decode.

Hai mục tiêu (memory và speed) cần thực sự cân nhắc cho từng use case. Phần 3 sẽ giúp bạn hiểu các option.

## Cầu nối

Phần 1 đã thấy `LlamaAttention.forward` gọi `past_key_values.update(...)`. Phần 2 đã thấy `EncoderDecoderCache` đóng gói hai DynamicCache. Phần 3 mở chi tiết: `update` thực sự làm gì, lưu data ở đâu, và các variant Cache có gì khác.

Sẵn sàng, ta sang chương 2.
