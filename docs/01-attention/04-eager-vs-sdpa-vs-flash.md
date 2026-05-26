---
title: Eager vs SDPA vs Flash
---

# Eager vs SDPA vs FlashAttention vs FlexAttention

Bốn backend mainstream. Mỗi backend là một file Python trong `integrations/`. Ta đi từng cái, đọc code, hiểu khi nào dùng.

## Eager: implementation tham chiếu

Eager nằm ngay trong `modeling_llama.py`:

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights
```

Đây là implementation tham chiếu. Mọi thứ rõ ràng:

1. `repeat_kv` để K/V có cùng số head với Q (GQA).
2. `matmul(Q, K^T) * scale` cho ra `attn_weights` shape `(batch, num_heads, q_len, k_len)`.
3. Mask được cộng thêm dưới dạng additive (tức `-inf` cho vị trí cần che, `0` cho vị trí còn lại).
4. Softmax theo float32 để tránh underflow (đặc biệt với bf16/fp16), rồi cast về dtype gốc.
5. Dropout chỉ kích hoạt khi `module.training`.
6. Matmul với V để aggregate, rồi `transpose` đưa head dim ra sau seq dim.

Ưu điểm: dễ debug, hỗ trợ `output_attentions=True` (trả attn_weights), hỗ trợ mọi mask shape. Nhược điểm: chậm và tốn memory vì materialize `attn_weights` shape `(batch, num_heads, q_len, k_len)`. Với `q_len = k_len = 32k`, tensor này là `1 * 32 * 32k * 32k * 2 bytes = ~64 GB`. Không khả thi cho long context.

## SDPA: kernel native của PyTorch

Đọc `src/transformers/integrations/sdpa_attention.py`:

```python
def sdpa_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
```

Ba điểm tinh tế:

**Điểm 1: GQA path.** Nếu PyTorch >= 2.5 và `attention_mask is None`, SDPA hỗ trợ GQA natively qua flag `enable_gqa=True`, tránh phải `repeat_kv` (tiết kiệm memory K/V gấp `num_key_value_groups` lần). Ngược lại, phải `repeat_kv` trước.

**Điểm 2: `is_causal` logic.** SDPA có hai mode mask. Nếu `is_causal=True`, kernel tự sinh causal mask trong kernel (rất nhanh, không materialize). Nếu pass `attn_mask`, kernel dùng mask từ user (chậm hơn vì phải đọc memory). HF check `query.shape[2] > 1` để phân biệt prefill (toàn bộ prompt, q_len > 1) vs decode (q_len = 1). Khi decode, một token mới attend tới mọi token cũ; không cần causal mask, set `is_causal=False`.

**Điểm 3: không trả `attn_weights`.** SDPA không cho phép truy cập intermediate, nên trả `None`. Đây là lý do `output_attentions=True` không tương thích với SDPA. HF có warning rõ ràng.

SDPA tự chọn kernel tốt nhất: `math` (fallback), `memory_efficient` (Xformers-style), `flash` (FlashAttention 2 nếu hỗ trợ), hoặc `cudnn`. User có thể ép kernel qua `torch.backends.cuda.sdp_kernel(...)`.

## FlashAttention 2/3/4: kernel CUDA tối ưu

FlashAttention không materialize `attn_weights`. Ý tưởng: chia Q, K, V thành block, đưa vào SRAM, tính softmax trên block trong SRAM, accumulate output bằng running max/sum trick. Memory complexity giảm từ O(N²) xuống O(N).

File: `integrations/flash_attention.py`. Bên trong dispatch tới `flash_attn_func` hoặc `flash_attn_varlen_func` (cho variable-length batching) tuỳ packing format.

Ưu điểm: nhanh nhất cho long sequence, tiết kiệm memory cực mạnh. Nhược điểm: yêu cầu Ampere+ (v2), Hopper (v3), Blackwell (v4); không hỗ trợ mọi mask custom (chỉ causal, sliding window built-in); không có `attn_weights`.

Khi chọn `flash_attention_2`, HF kiểm tra môi trường (CUDA version, GPU compute capability) ở `from_pretrained`. Nếu không khả dụng, raise error rõ ràng.

## FlexAttention: custom kernel compiled

FlexAttention (PyTorch 2.5+) cho phép user viết function mask/score modify Python, được compile thành kernel CUDA tối ưu.

```python
def causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))

attn_out = flex_attention(q, k, v, score_mod=causal_score_mod)
```

`integrations/flex_attention.py` wrap function này, cung cấp các mask pattern built-in (causal, sliding window, document mask, prefix LM). HF tận dụng FlexAttention cho các model có mask phức tạp như Gemma-2 (sliding window mỗi 2 layer), Mistral (sliding window).

Ưu điểm: linh hoạt + nhanh. Nhược điểm: cần `torch.compile`, chậm warmup, đôi khi recompile khi mask shape thay đổi.

## Bảng so sánh

| Backend | Memory | Speed | output_attentions | GQA native | Custom mask | GPU yêu cầu |
|---|---|---|---|---|---|---|
| eager | O(N²) | chậm | có | repeat_kv | mọi shape | bất kỳ |
| sdpa | O(N) | nhanh | không | torch ≥ 2.5 | mọi shape | bất kỳ (CUDA tốt nhất) |
| flash_attention_2 | O(N) | nhanh nhất | không | có | hạn chế | Ampere+ |
| flash_attention_3 | O(N) | rất nhanh | không | có | hạn chế | Hopper |
| flex_attention | O(N) | nhanh (compiled) | không | có | tùy biến | torch ≥ 2.5 |

## Khi nào chọn cái nào?

- Debug: `eager` để check `attn_weights`, kiểm tra correctness.
- Default cho user thông thường: `sdpa`. PyTorch tự pick kernel tốt nhất.
- Production long context, GPU hiện đại: `flash_attention_2` hoặc `3`.
- Model có mask custom (sliding window, document mask): `flex_attention`.
- Continuous batching (multi-request server): `paged|*` backend, xem chương 5.

Một nguyên tắc: nếu code chạy được với `eager`, output đúng numeric. Mọi backend khác phải khớp với eager trong tolerance. Khi nghi ngờ bug numeric, hãy fallback về eager để kiểm.
