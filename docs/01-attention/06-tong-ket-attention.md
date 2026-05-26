---
title: Tổng kết Phần 1
---

# Tổng kết Phần 1

Ta đã đi qua năm chương về attention. Chương cuối này tóm lại các bài học thiết kế quan trọng và các pitfall thường gặp.

## Bài học thiết kế

**Một class attention duy nhất cho mỗi model.** Trước version 4.45, mỗi model có `LlamaAttention`, `LlamaSdpaAttention`, `LlamaFlashAttention2`. Bây giờ chỉ một class, dispatch backend qua dict. Decision tốt: ít code duplication, dễ thêm backend.

**Backend là function, không phải class.** Stateless function, nhận `module` và tensor, trả `(attn_output, attn_weights)`. Đơn giản, dễ test, dễ register. Không có overhead `nn.Module`, không vướng `state_dict`.

**Position embeddings tính ngoài Attention.** `cos`, `sin` được tính một lần ở `LlamaModel`, truyền vào mọi layer. Tiết kiệm 32 lần tính cho Llama-3 8B. Đây là pattern bạn nên áp dụng nếu viết model riêng.

**Cache update trong Attention, không ngoài.** `past_key_values.update(k, v, layer_idx)` được gọi ngay sau RoPE và trước backend. K/V trong cache là phiên bản đã rotate. Khi decode chỉ cần rotate một token mới.

**`is_causal` ở mức attribute, không phải kwarg.** `LlamaAttention.is_causal = True`. Cross-attention sẽ override `is_causal = False`. Backend đọc attribute này để chọn fast path.

## Bảng tổng hợp dispatch

```
config._attn_implementation        function được gọi
---------------------------------  -----------------------------------
"eager"                            eager_attention_forward (trong modeling file)
"sdpa"                             sdpa_attention_forward (integrations/sdpa_attention.py)
"flash_attention_2"                flash_attention_forward (integrations/flash_attention.py)
"flash_attention_3"                flash_attention_forward
"flash_attention_4"                flash_attention_forward
"flex_attention"                   flex_attention_forward (integrations/flex_attention.py)
"paged|sdpa"                       sdpa_attention_paged_forward
"paged|flash_attention_2"          paged_attention_forward
```

Mọi key được khai báo trong `_global_mapping`. Nếu key không có (và không phải `"eager"`), `get_interface` raise `KeyError`.

## Pitfall khi đọc/viết model

**1. Quên `repeat_kv` khi backend không hỗ trợ GQA native.** SDPA của PyTorch < 2.5 không hiểu `enable_gqa`. Phải repeat manually. Quên dẫn tới shape mismatch.

**2. Nhầm thứ tự RoPE và cache.** RoPE phải áp **trước** cache update. Nếu áp sau, K cũ trong cache chưa rotate theo vị trí cũ, K mới rotate theo vị trí mới, kết quả sai.

**3. Truyền `attention_mask` cho SDPA khi muốn `is_causal=True`.** SDPA chỉ pick fast path khi `attn_mask=None` và `is_causal=True`. Pass cả hai sẽ rơi vào path math chậm. HF đã handle (`is_causal = query.shape[2] > 1 and attention_mask is None and is_causal`), nhưng khi viết model custom phải nhớ.

**4. `output_attentions=True` với SDPA/Flash.** Không khả thi vì các backend này không expose intermediate. HF chỉ warning. Khi cần inspect attn_weights, switch về eager.

**5. `attention_dropout` không kích hoạt vì `model.training=False`.** Khi gọi `model.train()` quên, dropout không chạy. Backend đọc `module.training` để quyết.

**6. Mask shape không khớp.** Eager mong đợi mask `(batch, num_heads, q_len, k_len)` hoặc broadcast được. SDPA chấp nhận `(batch, 1, q_len, k_len)`. Flash chỉ chấp nhận causal hoặc sliding pattern built-in. Sai shape có thể gây error hoặc tệ hơn, sai im lặng.

## Khi nào nên đọc lại Phần 1

- Khi thấy hiệu năng attention tệ, cần kiểm backend đang dùng.
- Khi viết một model architecture mới, cần kế thừa pattern của HF.
- Khi tích hợp một kernel attention mới (FlashAttention 5, RingAttention, ...), cần biết chỗ register.
- Khi debug bug numeric: fallback eager để so sánh.

## Cầu nối sang Phần 2

Mọi thứ ở Phần 1 nói về **self-attention**: Q, K, V cùng đến từ một sequence. Phần 2 mở rộng sang **cross-attention**: Q từ decoder, K/V từ encoder. Cấu trúc class gần giống `LlamaAttention`, nhưng `is_causal=False`, K/V không update mỗi step (vì encoder chạy một lần duy nhất), và cache có cấu trúc đặc biệt là `EncoderDecoderCache`.

Mời bạn tiếp tục Phần 2.
